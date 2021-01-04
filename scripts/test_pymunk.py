import numpy as np
import matplotlib.pyplot as plt
import pymunk

from mm2d import models, control, plotter
from mm2d import trajectory as trajectories
from mm2d.util import rms

import IPython


# robot parameters
Lx = 0
Ly = 0
L1 = 1
L2 = 1
VEL_LIM = 1
ACC_LIM = 1

DT = 0.005         # simulation timestep (s)
PLOT_PERIOD = 20  # update plot every PLOT_PERIOD timesteps
CTRL_PERIOD = 20  # generate new control signal every CTRL_PERIOD timesteps

DURATION = 10.0  # duration of trajectory (s)


def main():
    N = int(DURATION / DT) + 1

    # TODO eventually pymunk acts as the sim, which is slightly different that
    # the model
    model = models.ThreeInputKinematicModel(VEL_LIM, ACC_LIM, l1=L1, l2=L2,
                                            output_idx=[0, 1])

    ts = DT * np.arange(N)
    q0 = np.array([0, np.pi/4.0, -np.pi/4.0])

    space = pymunk.Space()
    space.gravity = (0, -9.8)

    # base
    base_body = pymunk.Body(body_type=pymunk.Body.KINEMATIC)
    base_w = 1.0
    base_h = 0.25
    base_trans = pymunk.Transform(tx=0, ty=-base_h/2.0)
    base = pymunk.Poly(base_body, [(base_w/2.0, base_h/2.0), (-base_w/2.0, base_h/2.0), (-base_w/2.0, -base_h/2.0), (base_w/2.0, -base_h/2.0)], base_trans)
    space.add(base.body, base)

    # arm link 1
    x1_mid = q0[0] + Lx + 0.5*L1
    y1_mid = Ly
    l1_body = pymunk.Body()  # TODO add mass and inertia directly for full control
    l1_body.position = (x1_mid, y1_mid)
    l1 = pymunk.Segment(l1_body, (-0.5*L1, 0), (0.5*L1, 0), radius=0.1)
    l1.mass = 1
    space.add(l1.body, l1)

    # arm joint 1
    j1 = pymunk.PinJoint(base.body, l1.body, (0, 0), (-0.5*L1, 0))
    j1.collide_bodies = False
    space.add(j1)

    # arm link 2
    x2_mid = q0[0] + Lx + L1 + 0.5*L2
    y2_mid = Ly
    l2_body = pymunk.Body()  # TODO add mass and inertia directly for full control
    l2_body.position = (x2_mid, y2_mid)
    l2 = pymunk.Segment(l2_body, (-0.5*L2, 0), (0.5*L2, 0), radius=0.1)
    l2.mass = 1
    space.add(l2.body, l2)

    # arm joint 2
    j2 = pymunk.PinJoint(l1.body, l2.body, (0.5*L1, 0), (-0.5*L2, 0))
    j2.collide_bodies = False
    space.add(j2)

    # set initial joint positions
    base.body.position = (q0[0], 0)
    l1.body.angle = q0[1]
    l2.body.angle = q0[2]

    plt.ion()
    fig = plt.figure()
    ax = plt.gca()
    plt.grid()

    ax.set_xlabel('x (m)')
    ax.set_ylabel('y (m)')
    ax.set_xlim([-1, 6])
    ax.set_ylim([-1, 2])
    ax.set_aspect('equal')

    robot_renderer = plotter.ThreeInputRenderer(model, q0)
    robot_renderer.render(ax)

    q = q0

    for i in range(N - 1):
        t = ts[i]

        l1.body.angular_velocity = 0.4
        l2.body.angular_velocity = 0.4

        if i % PLOT_PERIOD == 0:
            q = np.array([base.body.position[0], l1.body.angle, l2.body.angle])
            print(q)

            robot_renderer.set_state(q)
            robot_renderer.update_render()

            fig.canvas.draw()
            fig.canvas.flush_events()

        space.step(DT)


if __name__ == '__main__':
    main()
