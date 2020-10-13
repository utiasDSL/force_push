#!/usr/bin/env python
import numpy as np
import matplotlib.pyplot as plt

from mm2d.model import ThreeInputModel
from mm2d.controller import MPC
from mm2d.plotter import RealtimePlotter, ThreeInputRenderer, TrajectoryRenderer
# from mm2d.obstacle import Wall, Circle
from mm2d.trajectory import Line, Circle
from mm2d.util import rms, bound_array

import IPython


# robot parameters
L1 = 1
L2 = 1
VEL_LIM = 1
ACC_LIM = 0.3

DT = 0.1         # timestep (s)
DURATION = 10.0  # duration of trajectory (s)

# mpc parameters
NUM_HORIZON = 10  # number of time steps for prediction horizon
NUM_WSR = 100     # number of working set recalculations
NUM_ITER = 2      # number of linearizations/iterations


def main():
    N = int(DURATION / DT)

    model = ThreeInputModel(L1, L2, VEL_LIM, acc_lim=ACC_LIM, output_idx=[0, 1])

    Q = np.eye(model.no)
    R = np.eye(model.ni) * 0.01
    mpc = MPC(model, DT, Q, R, VEL_LIM, ACC_LIM)

    ts = np.array([i * DT for i in range(N)])
    qs = np.zeros((N, model.ni))
    dqs = np.zeros((N, model.ni))
    us = np.zeros((N, model.ni))
    ps = np.zeros((N, model.no))
    vs = np.zeros((N, model.no))
    pds = np.zeros((N, model.no))

    q0 = np.array([0, np.pi/4.0, -np.pi/4.0])
    p0 = model.forward(q0)

    # reference trajectory
    # trajectory = Line(p0, v=np.array([0.1, 0, 0]))
    trajectory = Circle(p0, r=0.5, duration=10)

    # obstacles
    # obs = Wall(x=2.5)
    # obs = Circle(c=np.array([3.0, 1.5]), r=1)

    q = q0
    p = p0
    dq = np.zeros(model.ni)
    qs[0, :] = q0
    ps[0, :] = p0
    pds[0, :] = p0

    robot_renderer = ThreeInputRenderer(model, q0)
    trajectory_renderer = TrajectoryRenderer(trajectory, ts)
    plotter = RealtimePlotter([robot_renderer, trajectory_renderer])
    plotter.start()

    for i in range(N - 1):
        # MPC
        # The +1 ts[i+1] is because we want to generate a u[i] such that
        # p[i+1] = FK(q[i+1]) = pd[i+1]
        n = min(NUM_HORIZON, N - 1 - i)
        pd, _ = trajectory.unroll(ts[i+1:i+1+n], flatten=True)
        u = mpc.solve(q, dq, pd, n)

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

        robot_renderer.set_state(q)
        plotter.update()
    plotter.done()

    xe = pds[1:, 0] - ps[1:, 0]
    ye = pds[1:, 1] - ps[1:, 1]
    print('RMSE(x) = {}'.format(rms(xe)))
    print('RMSE(y) = {}'.format(rms(ye)))

    # plt.plot(ts, pr, label='$\\theta_d$', color='k', linestyle='--')
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

    plt.show()


if __name__ == '__main__':
    main()
