import numpy as np
import pymunk


class PymunkSimulation:
    def __init__(self, gravity=9.8):
        self.space = pymunk.Space()
        self.space.gravity = (0, 9.8)

    def add_robot(self, model, q0):
        # self.model = model

        qz = np.zeros(model.ni)
        self.dq = np.zeros(model.ni)

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
        ax, ay = model.arm_points(qz)
        link1_body = pymunk.Body()
        link1_body.position = (ax[1], ay[1])
        # TODO can simplify below I think
        link1 = pymunk.Segment(link1_body, (-0.5*model.l1, 0),
                               (0.5*model.l1, 0), radius=0.1)
        link1.mass = 1
        self.space.add(link1.body, link1)

        # arm joint 1
        joint1 = pymunk.PinJoint(base.body, link1.body, (0, 0),
                                 (-0.5*model.l1, 0))
        joint1.collide_bodies = False
        self.space.add(joint1)

        # arm link 2
        link2_body = pymunk.Body()
        link2_body.position = (ax[2], ay[2])
        link2 = pymunk.Segment(link2_body, (-0.5*model.l2, 0),
                               (0.5*model.l2, 0), radius=0.1)
        link2.mass = 1
        self.space.add(link2.body, link2)

        # arm joint 2
        joint2 = pymunk.PinJoint(link1.body, link2.body, (0.5*model.l1, 0),
                                 (-0.5*model.l2, 0))
        joint2.collide_bodies = False
        self.space.add(joint2)

        # set initial joint positions
        base.body.position = (q0[0], 0)
        link1.body.angle = q0[1]
        link2.body.angle = q0[2]

        self.links = [base.body, link1.body, link2.body]

    def command_velocity(self, dq):
        self.dq = dq

    def command_torque(self, tau):
        # TODO need to get motors working for this to be possible
        pass

    def step(self, dt):
        self.links[0].velocity = (self.dq[0], 0)
        self.links[1].angular_velocity = self.dq[1]
        self.links[2].angular_velocity = self.dq[2]

        self.space.step(dt)
        q = np.array([self.links[0].position[0], self.links[1].angle,
                      self.links[2].angle])
        dq = np.array([self.links[0].velocity[0],
                       self.links[1].angular_velocity,
                       self.links[2].angular_velocity])
        return q, dq
