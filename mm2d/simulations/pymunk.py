import numpy as np
import pymunk
import IPython


class PymunkSimulation:
    def __init__(self, dt, gravity=-9.8):
        self.dt = dt
        self.space = pymunk.Space()
        self.space.gravity = (0, gravity)

    def add_robot(self, model, q0):
        self.q0 = q0
        self.dq_des = np.zeros(model.ni)
        self.q_des = np.copy(q0)
        self.q = np.copy(q0)
        self.dq = np.zeros(3)

        qz = np.zeros(model.ni)

        # base
        base_body = pymunk.Body(body_type=pymunk.Body.KINEMATIC)
        bx, by = model.base_corners(qz)
        by += 0.5*model.bh
        base = pymunk.Poly(
                base_body,
                [(x, y) for x, y in zip(bx, by)],
                pymunk.Transform(tx=0, ty=-0.5*model.bh))
        self.space.add(base.body, base)

        # arm link 1
        # TODO add mass and inertia directly for full control
        ax, ay = model.arm_points(q0)
        dx1 = 0.5*model.l1*np.cos(q0[1])
        dy1 = 0.5*model.l1*np.sin(q0[1])
        link1_body = pymunk.Body()
        link1_body.position = (ax[0] + dx1, ay[0] + dy1)
        link1 = pymunk.Segment(link1_body, (-dx1, -dy1), (dx1, dy1), radius=0.05)
        link1.mass = 1
        self.space.add(link1.body, link1)

        # arm joint 1
        joint1 = pymunk.PinJoint(base.body, link1.body, (0, 0), (-dx1, -dy1))
        joint1.collide_bodies = False
        motor1 = pymunk.constraints.SimpleMotor(base.body, link1.body, 0)
        self.space.add(joint1, motor1)

        # arm link 2
        dx2 = 0.5*model.l2*np.cos(q0[1]+q0[2])
        dy2 = 0.5*model.l2*np.sin(q0[1]+q0[2])
        link2_body = pymunk.Body()
        link2_body.position = (ax[1] + dx2, ay[1] + dy2)
        link2 = pymunk.Segment(
                link2_body,
                (-dx2, -dy2),
                (dx2, dy2),
                radius=0.05)
        link2.mass = 1
        self.space.add(link2.body, link2)

        # arm joint 2
        joint2 = pymunk.PinJoint(link1.body, link2.body, (dx1, dy1),
                                 (-dx2, -dy2))
        joint2.collide_bodies = False
        motor2 = pymunk.constraints.SimpleMotor(link1.body, link2.body, 0)
        self.space.add(joint2, motor2)

        self.model = model
        self.links = [base.body, link1.body, link2.body]
        self.motors = [motor1, motor2]

    def command_velocity(self, dq_des):
        self.dq_des = dq_des

    def command_torque(self, tau):
        # TODO calculate x = (q, dq) with u = tau, then limit motors to tau
        # force
        pass

    def _set_motor_rates(self, rate):
        self.links[0].velocity = (rate[0], 0)

        # Pymunk convention for motors is positive rate = clockwise rotation
        self.motors[0].rate = -rate[1]
        self.motors[1].rate = -rate[2]

    def _read_state(self):
        # we have to do some translation from the angle representation of
        # pymunk:
        # * subtract q1 from q2, since the angle of link2 is relative to the
        #   world in pymunk, but we want it relative to the angle of link1
        # * add q0, since whatever state we setup the sim with is set as 0
        q = np.array([self.links[0].position[0], self.links[1].angle,
                      self.links[2].angle - self.links[1].angle]) + self.q0

        dq = np.array([self.links[0].velocity[0],
                       self.links[1].angular_velocity,
                       self.links[2].angular_velocity])
        return q, dq

    def step(self):
        ''' Step the simulation forward in time. '''
        # internal control: we use a integral controller on the motor velocity
        # to reduce tracking error
        rate = 100 * (self.q_des - self.q) + self.dq_des
        self._set_motor_rates(rate)

        self.space.step(self.dt)

        # integrator for desired joint positions
        self.q_des += self.dt * self.dq_des

        self.q, self.dq = self._read_state()

        return self.q, self.dq
