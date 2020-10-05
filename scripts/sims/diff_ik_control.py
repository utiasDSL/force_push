#!/usr/bin/env python
import numpy as np
import matplotlib.pyplot as plt

from mm2d.model import ThreeInputModel
from mm2d.controller import BaselineController, BaselineController2
from mm2d.plotter import RobotPlotter
# from mm2d.obstacle import Wall, Circle
from mm2d.trajectory import Line, Circle
from mm2d.util import rms, bound_array

import IPython


# robot parameters
L1 = 1
L2 = 1

DT = 0.1        # timestep (s)
DURATION = 10.0  # duration of trajectory (s)

NUM_WSR = 100     # number of working set recalculations

LB = -1.0
UB = 1.0


def main():
    N = int(DURATION / DT)

    model = ThreeInputModel(L1, L2, LB, UB, output_idx=[0, 1])

    W = 0.1 * np.eye(model.ni)
    K = np.eye(model.no)
    # controller = BaselineController(model, W, K, LB, UB)
    controller = BaselineController2(model, W, K, LB, UB)

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

    q = q0
    p = p0
    dq = np.zeros(model.ni)
    qs[0, :] = q0
    ps[0, :] = p0
    pds[0, :] = p0

    plotter = RobotPlotter(model, trajectory)
    plotter.start(q0, ts)

    for i in range(N - 1):
        t = ts[i]

        # controller
        pd, vd = trajectory.sample(t)
        u = controller.solve(q, dq, pd, vd)

        # step the model
        q, dq = model.step(q, u, DT)
        p = model.forward(q)
        v = model.jacobian(q).dot(dq)

        # record
        us[i, :] = u

        dqs[i+1, :] = dq
        qs[i+1, :] = q
        ps[i+1, :] = p
        pds[i+1, :] = pd[:model.no]
        vs[i+1, :] = v

        plotter.update(q)

    plt.ioff()

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
