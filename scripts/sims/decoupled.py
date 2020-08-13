#!/usr/bin/env python
import numpy as np
import matplotlib.pyplot as plt

from mm2d.plotter import RobotPlotter
from mm2d.trajectory import Line
from mm2d.controller import BaselineController
from mm2d.model import ThreeInputModel


# model parameters
# link lengths
L1 = 1
L2 = 1

# input bounds
LB = -1
UB = 1

# trajectory parameters
DT = 0.1         # timestep (s)
DURATION = 20.0  # duration of trajectory (s)


def main():
    N = int(DURATION / DT)

    # robot model
    model = ThreeInputModel(L1, L2, LB, UB, output_idx=[0, 1])

    # robot controller
    W = np.eye(model.ni)
    K = np.eye(model.no)
    controller = BaselineController(model, W, K, LB, UB)

    ts = np.array([i * DT for i in range(N)])
    qs = np.zeros((N, model.ni))
    dqs = np.zeros((N, model.ni))
    us = np.zeros((N, model.ni))
    ps = np.zeros((N, model.no))
    vs = np.zeros((N, model.no))
    pds = np.zeros((N, model.no))

    # setup initial conditions
    q0 = np.array([0, np.pi/4.0, -np.pi/4.0])
    p0 = model.forward(q0)

    # reference trajectory
    trajectory = Line(p0, v=np.array([0.1, 0]))

    q = q0
    p = p0
    qs[0, :] = q0
    ps[0, :] = p0
    pds[0, :] = p0

    # real time plotter
    # plotter = RobotPlotter(model, trajectory)
    # plotter.start(q0, ts)

    # simulation loop
    for i in range(N - 1):
        t = ts[i+1]

        # step forward
        pd, vd = trajectory.sample(t)

        # extra constraints to decouple system
        J = model.jacobian(q)
        C = np.array([0, J[0, 1], J[0, 2]])

        u = controller.solve(q, pd, vd, C=None)

        q, dq = model.step(q, u, DT)
        p = model.forward(q)
        v = model.jacobian(q).dot(dq)

        # record
        us[i, :] = u
        dqs[i+1, :] = dq
        qs[i+1, :] = q
        ps[i+1, :] = p
        pds[i+1, :] = pd
        vs[i+1, :] = v

        # plot
        # plotter.update(q)

    plt.ioff()

    # plot cartesian trajectory
    plt.figure()
    plt.plot(pds[:, 0], pds[:, 1], label='Desired')
    plt.plot(ps[:, 0], ps[:, 1], label='Actual')
    plt.legend()
    plt.xlabel('x (m)')
    plt.ylabel('y (m)')

    # plot joints
    plt.figure()
    plt.subplot(211)
    plt.plot(ts, qs[:, 0], label='$x_b$ (m)')
    plt.plot(ts, qs[:, 1], label='$\\theta_1$ (rad)')
    plt.plot(ts, qs[:, 2], label='$\\theta_2$ (rad)')
    plt.legend()
    plt.grid()
    plt.ylabel('Joint positions')

    plt.subplot(212)
    plt.plot(ts[1:], us[1:, 0], label='$\dot{x}_b$ (m/s)')
    plt.plot(ts[1:], us[1:, 1], label='$\\theta_1$')
    plt.plot(ts[1:], us[1:, 2], label='$\\theta_2$')
    plt.legend()
    plt.grid()
    plt.ylabel('Joint velocities')
    plt.xlabel('Time (s)')

    # plot cost
    Js = np.array([u.T.dot(W).dot(u) for u in us])
    plt.figure()
    plt.plot(ts[:-1], Js[:-1])
    plt.xlabel('Time (s)')
    plt.ylabel('Cost')

    plt.show()


if __name__ == '__main__':
    main()
