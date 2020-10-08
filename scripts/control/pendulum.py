#!/usr/bin/env python
import numpy as np
import matplotlib.pyplot as plt
import control

from mm2d.model import ThreeInputModel, InvertedPendulum
from mm2d.controller import BaselineController, BaselineController2
from mm2d.plotter import RealtimePlotter, ThreeInputRenderer, PendulumRenderer

import IPython


# pendulum parameters
G = 1
L_PEND = 1
M_PEND = 1

# robot parameters
L1 = 1
L2 = 1

LB = -1.0
UB = 1.0

# simulation parameters
DT = 0.1         # timestep (s)
DURATION = 10.0  # duration (s)

NUM_WSR = 100    # number of working set recalculations


def main():
    N = int(DURATION / DT)

    model = ThreeInputModel(L1, L2, LB, UB, output_idx=[0, 1])
    pendulum = InvertedPendulum(L_PEND, M_PEND, G)

    W = 0.1 * np.eye(model.ni)

    # we don't want position feedback on x, only y
    K = np.array([[0, 0], [0, 1]])
    controller = BaselineController2(model, W, K, LB, UB)

    Q = np.eye(4)
    R = 0.01*np.eye(1)

    # LQR controller for the pendulum
    A = pendulum.A
    B = pendulum.B.reshape((4, 1))
    Q = np.eye(4)
    R = 0.01*np.eye(1)
    K, _, _ = control.lqr(A, B, Q, R)
    K = K.flatten()

    ts = np.array([i * DT for i in range(N)])
    qs = np.zeros((N, model.ni))
    dqs = np.zeros((N, model.ni))
    us = np.zeros((N, model.ni))
    ps = np.zeros((N, model.no))
    vs = np.zeros((N, model.no))

    q0 = np.array([0, np.pi/4.0, -np.pi/4.0])
    p0 = model.forward(q0)

    q = q0
    p = p0
    dq = np.zeros(model.ni)
    v = np.zeros(model.no)
    qs[0, :] = q0
    ps[0, :] = p0

    # pendulum state
    X = np.zeros((N, 4))
    X[0, :] = np.array([0.3, 0, 0, 0])

    # real time plotting
    robot_renderer = ThreeInputRenderer(model, q0, render_path=False)
    pendulum_renderer = PendulumRenderer(pendulum, X[0, :], p0)
    plotter = RealtimePlotter([robot_renderer, pendulum_renderer])
    plotter.start()

    for i in range(N - 1):
        t = ts[i]

        u_pendulum = -K @ X[i, :]

        # controller
        pd = p0
        vd = np.zeros(2)
        vd[0] = v[0] + DT * u_pendulum
        u = controller.solve(q, dq, pd, vd)

        # step the model
        q, dq = model.step(q, u, DT)
        p = model.forward(q)
        v = model.jacobian(q).dot(dq)

        x_acc = (v[0] - vs[i, 0]) / DT
        X[i+1, :] = pendulum.step(X[i, :], x_acc, DT)

        # record
        us[i, :] = u

        dqs[i+1, :] = dq
        qs[i+1, :] = q
        ps[i+1, :] = p
        vs[i+1, :] = v

        robot_renderer.set_state(q)
        pendulum_renderer.set_state(X[i+1, :], p)
        plotter.update()

    plotter.done()

    plt.figure()
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

    plt.figure()
    plt.plot(ts, X[:, 0])
    plt.grid()
    plt.title('Pendulum Angle')
    plt.xlabel('Time (s)')
    plt.ylabel('Angle (rad)')

    plt.show()


if __name__ == '__main__':
    main()
