import numpy as np
import matplotlib.pyplot as plt
import pymunk

from mm2d import models, control, plotter
from mm2d import trajectory as trajectories
from mm2d.util import rms

import IPython


# robot parameters
L1 = 1
L2 = 1
VEL_LIM = 1
ACC_LIM = 1

DT = 0.1        # timestep (s)
DURATION = 10.0  # duration of trajectory (s)


def main():
    N = int(DURATION / DT) + 1

    model = models.ThreeInputModel(L1, L2, VEL_LIM, acc_lim=ACC_LIM, output_idx=[0, 1])

    space = pymunk.Space()
    space.gravity = (0, 9.81)

    # robot body
    body = pymunk.Body(0, 0, body_type=pymunk.Body.KINEMATIC)
    base_w = 1.0
    base_h = 0.25
    base_trans = pymunk.Transform(tx=0, ty=-base_h/2.0)
    base_shape = pymunk.Poly(body, [(base_w/2.0, base_h/2.0), (-base_w/2.0, base_h/2.0), (-base_w/2.0, -base_h/2.0), (base_w/2.0, -base_h/2.0)], base_trans)

    IPython.embed()

    W = 0.1 * np.eye(model.ni)
    K = np.eye(model.no)
    controller = control.DiffIKController(model, W, K, DT, VEL_LIM, ACC_LIM)

    ts = DT * np.arange(N)
    qs = np.zeros((N, model.ni))
    dqs = np.zeros((N, model.ni))
    us = np.zeros((N, model.ni))
    ps = np.zeros((N, model.no))
    vs = np.zeros((N, model.no))
    pds = np.zeros((N, model.no))

    q0 = np.array([0, np.pi/4.0, -np.pi/4.0])
    p0 = model.forward(q0)

    # reference trajectory
    timescaling = trajectories.QuinticTimeScaling(DURATION)
    trajectory = trajectories.PointToPoint(p0, p0 + [1, 0], timescaling, DURATION)

    q = q0
    p = p0
    dq = np.zeros(model.ni)
    qs[0, :] = q0
    ps[0, :] = p0
    pds[0, :] = p0

    robot_renderer = plotter.ThreeInputRenderer(model, q0)
    trajectory_renderer = plotter.TrajectoryRenderer(trajectory, ts)
    plot = plotter.RealtimePlotter([robot_renderer, trajectory_renderer])
    plot.start()

    for i in range(N - 1):
        t = ts[i]

        # controller
        pd, vd, ad = trajectory.sample(t, flatten=True)
        u = controller.solve(q, dq, pd, vd)

        # step the model
        q, dq = model.step(q, u, DT, dq_last=dq)
        p = model.forward(q)
        v = model.jacobian(q).dot(dq)

        # record
        us[i, :] = u
        dqs[i+1, :] = dq
        qs[i+1, :] = q
        ps[i+1, :] = p
        pds[i+1, :] = pd[:model.no]
        vs[i+1, :] = v

        # render
        robot_renderer.set_state(q)
        plot.update()
    plot.done()

    xe = pds[1:, 0] - ps[1:, 0]
    ye = pds[1:, 1] - ps[1:, 1]
    print('RMSE(x) = {}'.format(rms(xe)))
    print('RMSE(y) = {}'.format(rms(ye)))

    plt.figure()
    plt.plot(ts, pds[:, 0], label='$x_d$', color='b', linestyle='--')
    plt.plot(ts, pds[:, 1], label='$y_d$', color='r', linestyle='--')
    plt.plot(ts, ps[:, 0],  label='$x$', color='b')
    plt.plot(ts, ps[:, 1],  label='$y$', color='r')
    plt.grid()
    plt.legend()
    plt.xlabel('Time (s)')
    plt.ylabel('Position')
    plt.title('End effector position')

    plt.figure()
    plt.plot(ts, vs[:, 0],  label='$v_x$', color='b')
    plt.plot(ts, vs[:, 1],  label='$v_y$', color='r')
    plt.grid()
    plt.legend()
    plt.xlabel('Time (s)')
    plt.ylabel('Velocity (m/s)')
    plt.title('End effector velocity')

    plt.figure()
    plt.plot(ts, dqs[:, 0], label='$\\dot{q}_x$')
    plt.plot(ts, dqs[:, 1], label='$\\dot{q}_1$')
    plt.plot(ts, dqs[:, 2], label='$\\dot{q}_2$')
    plt.grid()
    plt.legend()
    plt.title('Actual joint velocity')
    plt.xlabel('Time (s)')
    plt.ylabel('Velocity')

    plt.figure()
    plt.plot(ts, us[:, 0], label='$u_x$')
    plt.plot(ts, us[:, 1], label='$u_1$')
    plt.plot(ts, us[:, 2], label='$u_2$')
    plt.grid()
    plt.legend()
    plt.title('Commanded joint velocity')
    plt.xlabel('Time (s)')
    plt.ylabel('Velocity')

    plt.figure()
    plt.plot(ts, qs[:, 0], label='$q_x$')
    plt.plot(ts, qs[:, 1], label='$q_1$')
    plt.plot(ts, qs[:, 2], label='$q_2$')
    plt.grid()
    plt.legend()
    plt.title('Joint positions')
    plt.xlabel('Time (s)')
    plt.ylabel('Joint positions')

    plt.show()


if __name__ == '__main__':
    main()
