import numpy as np
import pymunk
from mm2d.util import bound_array


class PymunkSimulationTrayBalance:
    """Custom pymunk physics simulation for the tray balance project."""

    def __init__(self, dt, gravity=-9.8, iterations=10):
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

    def add_robot(self, model, q0, w, dynamic=False):
        self.q = np.copy(q0)
        self.dq = np.zeros_like(q0)
        self.a_cmd = np.zeros_like(q0)

        # ground
        self.add_ground(-model.bh)

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
        ee_finger1 = pymunk.Circle(ee_body, fr, (-w, 0))
        ee_finger2 = pymunk.Circle(ee_body, fr, (w, 0))
        ee_finger1.friction = 0.75
        ee_finger2.friction = 0.75
        ee_finger1.mass = 0.1
        ee_finger2.mass = 0.1
        self.space.add(ee_body, ee_finger1, ee_finger2)

        # arm joint 3: link 2 to EE
        joint3 = pymunk.PinJoint(link2.body, ee_body, (0.5*model.l2, 0), (0, 0))
        joint3.collide_bodies = False
        self.space.add(joint3)

        self.model = model
        self.links = [base.body, link1.body, link2.body, ee_body]

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
        v_cmd = self.dq + self.dt * self.a_cmd
        self._set_motor_rates(v_cmd)

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
        self.a_cmd = bound_array(a_cmd, -self.model.acc_lim, self.model.acc_lim)

    # def command_velocity(self, rate):
    #     # velocity limits
    #     rate = bound_array(rate, -self.model.vel_lim, self.model.vel_lim)
    #     self._set_motor_rates(rate)
