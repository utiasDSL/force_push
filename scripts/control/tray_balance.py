#!/usr/bin/env python
import numpy as np
import matplotlib.pyplot as plt

from mm2d.model import ThreeInputModel
from mm2d.controller import DiffIKController
from mm2d.plotter import RealtimePlotter, ThreeInputRenderer, TrajectoryRenderer
from mm2d.trajectory import Circle, Polygon, PointToPoint, CubicTimeScaling, QuinticTimeScaling, LinearTimeScaling, CubicBezier
from mm2d.util import rms
import qpoases

import IPython


# robot parameters
L1 = 1
L2 = 1
VEL_LIM = 1
ACC_LIM = 1

DT = 0.01        # timestep (s)
DURATION = 1.0  # duration of trajectory (s)

NUM_WSR = 100     # number of working set recalculations


class TrayRenderer(object):
    def __init__(self, length, a, p0):
        self.length = length
        self.a = a
        self.p = p0

    def set_state(self, p):
        self.p = p

    def render(self, ax):
        # origin
        xo = self.p[0]
        yo = self.p[1]

        # sides
        xl = xo - 0.5*self.length * np.cos(self.p[2])
        yl = yo - 0.5*self.length * np.sin(self.p[2])
        xr = xo + 0.5*self.length * np.cos(self.p[2])
        yr = yo + 0.5*self.length * np.sin(self.p[2])

        # contact points
        x1 = xo - 0.5*self.a * np.cos(self.p[2])
        y1 = yo - 0.5*self.a * np.sin(self.p[2])
        x2 = xo + 0.5*self.a * np.cos(self.p[2])
        y2 = yo + 0.5*self.a * np.sin(self.p[2])

        self.plot, = ax.plot([xl, xr], [yl, yr], color='k')
        self.points, = ax.plot([x1, x2], [y1, y2], 'o', color='k')

    def update_render(self):
        # origin
        xo = self.p[0]
        yo = self.p[1]

        # sides
        xl = xo - 0.5*self.length * np.cos(self.p[2])
        yl = yo - 0.5*self.length * np.sin(self.p[2])
        xr = xo + 0.5*self.length * np.cos(self.p[2])
        yr = yo + 0.5*self.length * np.sin(self.p[2])

        # contact points
        x1 = xo - 0.5*self.a * np.cos(self.p[2])
        y1 = yo - 0.5*self.a * np.sin(self.p[2])
        x2 = xo + 0.5*self.a * np.cos(self.p[2])
        y2 = yo + 0.5*self.a * np.sin(self.p[2])

        self.plot.set_xdata([xl, xr])
        self.plot.set_ydata([yl, yr])

        self.points.set_xdata([x1, x2])
        self.points.set_ydata([y1, y2])


def rot2d(a):
    c = np.cos(a)
    s = np.sin(a)
    R = np.array([[c, -s], [s, c]])
    return R


def main():
    N = int(DURATION / DT) + 1

    g = -9.81

    # tray params
    l = 1
    a = 0.4
    b = 0
    m = 5
    I = m*l**2/12.0
    M = np.diag([m, m, I])
    mu = 0.4

    # control params
    kp = 1
    kv = 0.1

    # cost parameters
    Q = np.zeros((7, 7))
    Q[0, 0] = 1
    R = np.diag([0.1, 0.1, 0.1, 0.0001, 0.0001, 0.0001, 0.0001])

    # constant optimization matrices
    E = np.array([[0, 0, 0, 0, -1, 0, 0],
                  [0, 0, 0, 0, 0, 0, -1],
                  [0, 0, 0, 1, -mu, 0, 0],
                  [0, 0, 0, -1, -mu, 0, 0],
                  [0, 0, 0, 0, 0, 1, -mu],
                  [0, 0, 0, 0, 0, -1, -mu]])

    lbA = np.array([0, m*g, 0, -np.infty, -np.infty, -np.infty, -np.infty, -np.infty, -np.infty])
    ubA = np.array([0, m*g, 0, 0, 0, 0, 0, 0, 0])

    ts = np.array([i * DT for i in range(N)])
    us = np.zeros((N, 3))
    pes = np.zeros((N, 3))
    ves = np.zeros((N, 3))
    pts = np.zeros((N, 3))
    fs = np.zeros((N, 4))

    pe = np.array([0, 0, 0])
    ve = np.array([0, 0, 0])
    pes[0, :] = pe

    timescaling = CubicTimeScaling(DURATION)
    trajectory = PointToPoint(pe, pe + [1, 0, 0], timescaling, DURATION)

    pds, *other = trajectory.sample(ts)

    tray_renderer = TrayRenderer(l, a, pe)
    trajectory_renderer = TrajectoryRenderer(trajectory, ts)
    plotter = RealtimePlotter([tray_renderer, trajectory_renderer])
    plotter.start()

    for i in range(N - 1):
        t = ts[i]

        # solve opt problem
        pd, vd, ad = trajectory.sample(t, flatten=True)

        # cost
        ddx_ref = kp*(pd[0] - pe[0]) + kv*(vd[0] - ve[0]) + ad[0]
        Xref = np.zeros(7)
        Xref[0] = ddx_ref

        H = Q + R
        g = -Q.dot(Xref)

        # constraints
        theta = pe[2]
        B = np.array([[1, 0, b*np.cos(theta)],
                      [0, 1, b*np.sin(theta)],
                      [0, 0, 1]])
        rot = rot2d(theta)
        D = np.block([[rot, rot], [0, -(0.5*a+b), 0, 0.5*a-b]])
        A = np.block([[M.dot(B), -D], [E]])

        qp = qpoases.PyQProblem(7, 9)
        options = qpoases.PyOptions()
        options.printLevel = qpoases.PyPrintLevel.NONE
        qp.setOptions(options)
        ret = qp.init(H, g, A, None, None, lbA, ubA, NUM_WSR)

        X = np.zeros(7)
        qp.getPrimalSolution(X)
        f = X[3:]

        u = X[:3]  # EE accel input is first three values
        ve = ve + DT * u
        pe = pe + DT * ve
        print(pe[2])

        # tray position is a constant offset from EE frame
        theta = pe[2]
        B = np.array([[1, 0, b*np.cos(theta)],
                      [0, 1, b*np.sin(theta)],
                      [0, 0, 1]])
        pt = B.dot(pe)

        # record
        us[i, :] = u
        pes[i+1, :] = pe
        ves[i+1, :] = ve
        pts[i+1, :] = pt
        pds[i, :] = pd
        fs[i, :] = f

        tray_renderer.set_state(pt)
        plotter.update()
    plotter.done()

    # IPython.embed()

    # xe = pds[1:, 0] - ps[1:, 0]
    # ye = pds[1:, 1] - ps[1:, 1]
    # print('RMSE(x) = {}'.format(rms(xe)))
    # print('RMSE(y) = {}'.format(rms(ye)))

    plt.figure()
    plt.plot(ts, pds[:, 0], label='$x_d$', color='b', linestyle='--')
    plt.plot(ts, pds[:, 1], label='$y_d$', color='r', linestyle='--')
    plt.plot(ts, pes[:, 0],  label='$x$', color='b')
    plt.plot(ts, pes[:, 1],  label='$y$', color='r')
    plt.grid()
    plt.legend()
    plt.xlabel('Time (s)')
    plt.ylabel('Position')
    plt.title('End effector position')
    #
    # plt.figure()
    # plt.plot(ts, dqs[:, 0], label='$\\dot{q}_x$')
    # plt.plot(ts, dqs[:, 1], label='$\\dot{q}_1$')
    # plt.plot(ts, dqs[:, 2], label='$\\dot{q}_2$')
    # plt.grid()
    # plt.legend()
    # plt.title('Actual joint velocity')
    # plt.xlabel('Time (s)')
    # plt.ylabel('Velocity')
    #
    # plt.figure()
    # plt.plot(ts, us[:, 0], label='$u_x$')
    # plt.plot(ts, us[:, 1], label='$u_1$')
    # plt.plot(ts, us[:, 2], label='$u_2$')
    # plt.grid()
    # plt.legend()
    # plt.title('Commanded joint velocity')
    # plt.xlabel('Time (s)')
    # plt.ylabel('Velocity')
    #
    # plt.figure()
    # plt.plot(ts, qs[:, 0], label='$q_x$')
    # plt.plot(ts, qs[:, 1], label='$q_1$')
    # plt.plot(ts, qs[:, 2], label='$q_2$')
    # plt.grid()
    # plt.legend()
    # plt.title('Joint positions')
    # plt.xlabel('Time (s)')
    # plt.ylabel('Joint positions')
    #
    plt.show()


if __name__ == '__main__':
    main()
