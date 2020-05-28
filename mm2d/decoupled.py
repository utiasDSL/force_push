#!/usr/bin/env python2
from __future__ import print_function
import numpy as np
import matplotlib.pyplot as plt

from plotter import RobotPlotter
from trajectory import Line
from controller import OptimizingController
from model import ThreeInputModel

import IPython


# robot parameters
L1 = 1
L2 = 1

DT = 0.1         # timestep (s)
DURATION = 20.0  # duration of trajectory (s)

Q = np.diag([1.0, 1.0, 0.00001])
R = np.eye(3) * 0.01


def main():
    N = int(DURATION / DT)

    # robot model
    model = ThreeInputModel(L1, L2)

    # robot controller
    controller = OptimizingController(model, DT, Q, R)

    ts = np.array([i * DT for i in xrange(N+1)])
    ps = np.zeros((N+1, model.p))
    qs = np.zeros((N+1, model.n))
    dqs = np.zeros((N+1, model.n))
    us = np.zeros((N+1, model.p))
    vs = np.zeros((N+1, model.p))

    # setup initial conditions
    q0 = np.array([0, np.pi/4.0, -np.pi/4.0])
    p0 = model.forward(q0)

    q = q0
    p = p0
    qs[0, :] = q0
    ps[0, :] = p0

    # reference trajectory
    trajectory = Line(p0, v=[0.1, 0])

    # real time plotter
    plotter = RobotPlotter(model, trajectory)
    plotter.start(q0, ts)

    # simulation loop
    for i in xrange(N):
        t = ts[i+1]

        # step forward
        pd = trajectory.sample(t)
        u = controller.solve(q, pd)
        q, dq = model.step(q, u, DT)
        p = model.forward(q)
        v = model.jacobian(q).dot(dq)

        # record
        dqs[i+1, :] = dq
        qs[i+1, :] = q
        ps[i+1, :] = p
        us[i+1, :] = u
        vs[i+1, :] = v

        # plot
        plotter.update(q, t)

    plt.ioff()


if __name__ == '__main__':
    main()
