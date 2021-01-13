import matplotlib.pyplot as plt
import numpy as np
import pymunk
import weakref
from mm2d.util import bound_array, rotation_matrix


# TODO: idea is that this would be used for Jacobian calculations
# I think I'll need to keep track of relationships between bodies in the robot
# class: want to know dependencies on the inputs
# This means that I probably will want to replace the pymunk body and shape
# classes, to make things as I see fit
class Marker:
    """A single point attached to a rigid body."""

    def __init__(self, body, offset=np.zeros(2)):
        self.body = body
        self.offset = offset

    def get_position(self):
        R = rotation_matrix(self.body.angle)
        return np.array(self.body.position) + self.offset @ R.T


# NOTE: only used for rendering
# TODO: could be made into a Trajectory for convenient plotting afterward
class MarkerPathRenderer:
    """Records the path of a `Marker` over time."""

    def __init__(self, marker):
        self.marker = marker

    def render(self, ax):
        p = self.marker.get_position()
        self.xs = [p[0]]
        self.ys = [p[1]]
        self.plot, = ax.plot(self.xs, self.ys)

    def update_render(self):
        p = self.marker.get_position()
        self.xs.append(p[0])
        self.ys.append(p[1])
        self.plot.set_xdata(self.xs)
        self.plot.set_ydata(self.ys)


# TODO: a slightly cleaner interface would be to separate update and render
# logic, so updating only needs to be done when requested/needed
# If I end up wrapping all of these things in my own classes, then I can be
# even cleaner about adding things to the renderer
class PymunkRenderer:
    def __init__(self, space, markers):
        self.space = space
        self.patches = weakref.WeakKeyDictionary()
        self.marker_paths = [MarkerPathRenderer(marker) for marker in markers]

    def _create_patch(self, shape, ax):
        type_ = type(shape)
        facecolor = shape.facecolor if hasattr(shape, 'facecolor') else (1, 1, 1, 1)
        edgecolor = shape.edgecolor if hasattr(shape, 'edgecolor') else (0, 0, 0, 1)

        if type_ == pymunk.shapes.Circle:
            R = rotation_matrix(shape.body.angle)
            p = np.array(shape.body.position) + np.array(shape.offset) @ R.T
            r = shape.radius
            patch = plt.Circle(p, r, facecolor=facecolor, edgecolor=edgecolor)
            ax.add_patch(patch)
        elif type_ == pymunk.shapes.Poly:
            p = np.array(shape.body.position)
            R = rotation_matrix(shape.body.angle)
            v = p + np.array(shape.get_vertices()) @ R.T
            patch = plt.Polygon(v, facecolor=facecolor, edgecolor=edgecolor, closed=True)
            ax.add_patch(patch)
        elif type_ == pymunk.shapes.Segment:
            p = np.array(shape.body.position)
            R = rotation_matrix(shape.body.angle)
            v = p + np.array([shape.a, shape.b]) @ R.T
            patch = plt.Line2D(v[:, 0], v[:, 1], color=edgecolor, linewidth=1)
            ax.add_line(patch)
        else:
            raise Exception(f'unsupported shape type: {type_}')
        self.patches[shape] = patch

    def _update_patch(self, shape):
        type_ = type(shape)
        patch = self.patches[shape]
        if type_ == pymunk.shapes.Circle:
            R = rotation_matrix(shape.body.angle)
            p = np.array(shape.body.position) + np.array(shape.offset) @ R.T
            patch.center = p
        elif type_ == pymunk.shapes.Poly:
            p = np.array(shape.body.position)
            R = rotation_matrix(shape.body.angle)
            v = p + np.array(shape.get_vertices()) @ R.T
            patch.set_xy(v)
        elif type_ == pymunk.shapes.Segment:
            p = np.array(shape.body.position)
            R = rotation_matrix(shape.body.angle)
            v = p + np.array([shape.a, shape.b]) @ R.T
            patch.set_data(v.T)
        else:
            raise Exception(f'unsupported shape type: {type_}')

    def render(self, ax):
        for shape in self.space.shapes:
            if shape not in self.patches:
                self._create_patch(shape, ax)
            else:
                self._update_patch(shape)

        for marker_path in self.marker_paths:
            marker_path.render(ax)

    # TODO not totally happy with this API
    def update_render(self):
        for shape in self.space.shapes:
            self._update_patch(shape)

        for marker_path in self.marker_paths:
            marker_path.update_render()


class PymunkSimulationTrayBalance:
    """Custom pymunk physics simulation for the tray balance project."""

    def __init__(self, dt, gravity=-9.81, iterations=10):
        """Initialize the pymunk simulation.

        Arguments:
            dt: simulation timestep (seconds)
            gravity: vertical acceleration due to gravity (m/s**2)
            iterations: number of iterations the solver should perform each
                step; the Pymunk default is 10
        """
        self.dt = dt
        self.space = pymunk.Space()
        self.space.gravity = (0, gravity)
        self.space.iterations = iterations

    def add_ground(self, y):
        ground = pymunk.Segment(self.space.static_body, (-10, y), (10, y),
                                0.01)
        self.space.add(ground)
        ground.elasticity = 0
        ground.friction = 0.5

    def add_robot(self, model, q0, w, mu, dynamic=False):
        self.q = np.copy(q0)
        self.dq = np.zeros_like(q0)
        self.a_cmd = np.zeros_like(q0)
        self.v_cmd = np.zeros_like(q0)

        # ground
        # self.add_ground(-model.bh)

        # base
        base_type = pymunk.Body.DYNAMIC if dynamic else pymunk.Body.KINEMATIC
        base_body = pymunk.Body(body_type=base_type)
        bx, by = model.base_corners(q0)
        by += 0.5*model.bh
        base = pymunk.Poly(
                base_body,
                [(x, y) for x, y in zip(bx, by)],
                pymunk.Transform(tx=0, ty=-0.5*model.bh))
        base.friction = 0
        # add the mass manually but let pymunk figure out the moment
        base.mass = model.mb
        self.space.add(base.body, base)

        # arm link 1
        ax, ay = model.arm_points(q0)
        dx1 = 0.5*model.l1*np.cos(q0[1])
        dy1 = 0.5*model.l1*np.sin(q0[1])
        link1_body = pymunk.Body(mass=model.m1, moment=model.I1)
        link1_body.position = (ax[0] + dx1, ay[0] + dy1)
        link1_body.angle = q0[1]
        link1 = pymunk.Segment(link1_body, (-0.5*model.l1, 0),
                               (0.5*model.l1, 0), radius=0.05)
        link1.friction = 0.25
        # links don't collide with anything for now
        link1.filter = pymunk.ShapeFilter(categories=0)
        self.space.add(link1.body, link1)

        # arm joint 1
        joint1 = pymunk.PinJoint(base.body, link1.body, (0, 0),
                                 (-0.5*model.l1, 0))
        joint1.collide_bodies = False
        self.space.add(joint1)

        # arm link 2
        dx2 = 0.5*model.l2*np.cos(q0[1]+q0[2])
        dy2 = 0.5*model.l2*np.sin(q0[1]+q0[2])
        link2_body = pymunk.Body(mass=model.m2, moment=model.I2)
        link2_body.position = (ax[1] + dx2, ay[1] + dy2)
        link2_body.angle = q0[1] + q0[2]
        link2 = pymunk.Segment(link2_body, (-0.5*model.l2, 0),
                               (0.5*model.l2, 0), radius=0.05)
        link2.friction = 0.25
        link2.filter = pymunk.ShapeFilter(categories=0)
        self.space.add(link2.body, link2)

        # arm joint 2
        joint2 = pymunk.PinJoint(link1.body, link2.body, (0.5*model.l1, 0),
                                 (-0.5*model.l2, 0))
        joint2.collide_bodies = False
        self.space.add(joint2)

        # end effector
        fr = 0.05
        ee_body = pymunk.Body()
        ee_body.position = (ax[2], ay[2])
        ee_body.angle = np.sum(q0[1:4])
        ee_palm = pymunk.Segment(ee_body, (-w, 0), (w, 0), radius=0)
        ee_palm.mass = 0.1
        ee_palm.friction = mu
        ee_palm.filter = pymunk.ShapeFilter(categories=0)
        ee_finger1 = pymunk.Circle(ee_body, fr, (-w, 0))
        ee_finger2 = pymunk.Circle(ee_body, fr, (w, 0))
        ee_finger1.friction = mu
        ee_finger2.friction = mu
        ee_finger1.mass = 0.1
        ee_finger2.mass = 0.1
        self.space.add(ee_body, ee_palm, ee_finger1, ee_finger2)

        ee_marker = Marker(ee_body)

        # arm joint 3: link 2 to EE
        joint3 = pymunk.PinJoint(link2.body, ee_body, (0.5*model.l2, 0), (0, 0))
        joint3.collide_bodies = False
        self.space.add(joint3)

        self.model = model
        self.links = [base.body, link1.body, link2.body, ee_body]
        self.markers = [ee_marker]

        motor1 = pymunk.constraints.SimpleMotor(self.links[0], self.links[1], 0)
        motor2 = pymunk.constraints.SimpleMotor(self.links[1], self.links[2], 0)
        motor3 = pymunk.constraints.SimpleMotor(self.links[2], self.links[3], 0)
        self.space.add(motor1, motor2, motor3)
        self.motors = [motor1, motor2, motor3]

    def _read_state(self):
        # subtract q1 from q2, since the angle of link 2 is relative to the
        # angle of link 1
        q = np.array([self.links[0].position[0], self.links[1].angle,
                      self.links[2].angle - self.links[1].angle,
                      self.links[3].angle - self.links[2].angle])
        dq = np.array([self.links[0].velocity[0],
                       self.links[1].angular_velocity,
                       self.links[2].angular_velocity - self.links[1].angular_velocity,
                       self.links[3].angular_velocity - self.links[2].angular_velocity])
        return q, dq

    def step(self):
        """Step the simulation forward in time."""
        # generate motor velocity from acceleration command
        self.v_cmd = self.v_cmd + self.dt * self.a_cmd
        self._set_motor_rates(self.v_cmd)

        self.space.step(self.dt)
        self.q, self.dq = self._read_state()
        return self.q, self.dq

    def _set_motor_rates(self, rate):
        self.links[0].velocity = (rate[0], 0)

        # Pymunk convention for motors is positive rate = clockwise rotation
        self.motors[0].rate = -rate[1]
        self.motors[1].rate = -rate[2]
        self.motors[2].rate = -rate[3]

    def command_acceleration(self, a_cmd):
        self.v_cmd = np.copy(self.dq)
        self.a_cmd = bound_array(a_cmd, -self.model.acc_lim, self.model.acc_lim)

    # def command_velocity(self, rate):
    #     # velocity limits
    #     rate = bound_array(rate, -self.model.vel_lim, self.model.vel_lim)
    #     self._set_motor_rates(rate)

