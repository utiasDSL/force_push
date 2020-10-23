#!/usr/bin/env python
import numpy as np
import matplotlib.pyplot as plt

from mm2d.model import TopDownHolonomicModel
from mm2d.controller import DiffIKController
from mm2d.plotter import RealtimePlotter, TopDownHolonomicRenderer, TrajectoryRenderer, CircleRenderer
from mm2d.trajectory import Circle, Polygon, PointToPoint, CubicTimeScaling, QuinticTimeScaling, LinearTimeScaling, CubicBezier
from mm2d.obstacle import Circle as CircleObstacle
from mm2d.util import rms

import IPython


# robot parameters
L1 = 1
L2 = 1
VEL_LIM = 1
ACC_LIM = 1

DT = 0.1        # timestep (s)
DURATION = 10.0  # duration of trajectory (s)

NUM_WSR = 100     # number of working set recalculations


def main():
    N = int(DURATION / DT) + 1

    model = TopDownHolonomicModel(L1, L2, VEL_LIM, acc_lim=ACC_LIM, output_idx=[0, 1])

    W = 0.1 * np.eye(model.ni)
    K = np.eye(model.no)
    controller = DiffIKController(model, W, K, DT, VEL_LIM, ACC_LIM)

    ts = np.array([i * DT for i in range(N)])
    qs = np.zeros((N, model.ni))
    dqs = np.zeros((N, model.ni))
    us = np.zeros((N, model.ni))
    ps = np.zeros((N, model.no))
    vs = np.zeros((N, model.no))
    pds = np.zeros((N, model.no))

    # initial state
    q = np.array([0, 0, 0, 0])
    p = model.forward(q)
    dq = np.zeros(model.ni)

    # reference trajectory
    # trajectory = Line(p0, v0=np.zeros(2), a=np.array([0.01, 0]))
    timescaling = QuinticTimeScaling(DURATION)
    trajectory = PointToPoint(p, p + [1, 0], timescaling, DURATION)
    # trajectory2 = PointToPoint(p0 + [1, 0], p0 + [2, 0], timescaling, 0.5*DURATION)
    # trajectory = Chain([trajectory1, trajectory2])

    # points = np.array([p0, p0 + [1, 1], p0 + [2, -1], p0 + [3, 0]])
    # trajectory = CubicBezier(points, timescaling, DURATION)
    # trajectory = Circle(p, 0.5, timescaling, DURATION)
    # points = np.array([p0, p0 + [1, 0], p0 + [1, -1], p0 + [0, -1], p0])
    # trajectory = Polygon(points, v=0.4)

    obs = CircleObstacle(np.array([3., 0.1]), 0.5, 1000)

    qs[0, :] = q
    ps[0, :] = p
    pds[0, :] = p

    circle_renderer = CircleRenderer(obs)
    robot_renderer = TopDownHolonomicRenderer(model, q)
    trajectory_renderer = TrajectoryRenderer(trajectory, ts)
    plotter = RealtimePlotter([robot_renderer, trajectory_renderer, circle_renderer])
    plotter.start()

    for i in range(N - 1):
        t = ts[i]

        # controller
        pd, vd, ad = trajectory.sample(t, flatten=True)
        u = controller.solve(q, dq, pd, vd)

        # step the model
        q, dq = model.step(q, u, DT, dq_last=dq)
        p = model.forward(q)
        v = model.jacobian(q).dot(dq)

        f, movement = obs.force(p)
        obs.c += movement

        # record
        us[i, :] = u
        dqs[i+1, :] = dq
        qs[i+1, :] = q
        ps[i+1, :] = p
        pds[i+1, :] = pd[:model.no]
        vs[i+1, :] = v

        # render
        robot_renderer.set_state(q)
        plotter.update()
    plotter.done()

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
