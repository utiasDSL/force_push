#!/usr/bin/env python2
from __future__ import print_function
import numpy as np
import matplotlib.pyplot as plt

from plotter import RobotPlotter
from trajectory import Line
from controller import BaselineController
from model import ThreeInputModel

import IPython


# model parameters
# link lengths
L1 = 1
L2 = 1

# input bounds
LB = -1
UB = 1

# controller parameters
Q = np.diag([1.0, 1.0, 0.00001])
R = np.eye(3) * 0.01

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

    ts = np.array([i * DT for i in xrange(N+1)])
    qs = np.zeros((N+1, model.ni))
    dqs = np.zeros((N+1, model.ni))
    us = np.zeros((N+1, model.ni))
    ps = np.zeros((N+1, model.no))
    vs = np.zeros((N+1, model.no))

    # setup initial conditions
    q0 = np.array([0, np.pi/4.0, -np.pi/4.0])
    p0 = model.forward(q0)

    q = q0
    p = p0
    qs[0, :] = q0
    ps[0, :] = p0

    # reference trajectory
    trajectory = Line(p0, v=np.array([0.1, 0]))

    # real time plotter
    plotter = RobotPlotter(model, trajectory)
    plotter.start(q0, ts)

    # simulation loop
    for i in xrange(N):
        t = ts[i+1]

        # step forward
        pd, vd = trajectory.sample(t)
        u = controller.solve(q, pd, vd)
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
        plotter.update(q)

    plt.ioff()


if __name__ == '__main__':
    main()
