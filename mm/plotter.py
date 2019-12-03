import numpy as np
import matplotlib.pyplot as plt


class RobotPlotter(object):
    def __init__(self, l1, l2):
        self.l1 = l1
        self.l2 = l2
        self.xs = []
        self.ys = []

    def start(self, q0, xr, yr, obs):
        ''' Launch the plot. '''
        plt.ion()

        self.fig = plt.figure()
        self.ax = plt.gca()

        self.ax.set_xlabel('x (m)')
        self.ax.set_ylabel('y (m)')
        self.ax.set_xlim([-1, 6])
        self.ax.set_ylim([-1, 2])

        xa, ya = self._calc_arm_pts(q0)
        xb, yb = self._calc_body_pts(q0)

        self.xs.append(xa[-1])
        self.ys.append(ya[-1])

        self.arm, = self.ax.plot(xa, ya, color='k')
        self.body, = self.ax.plot(xb, yb, color='k')
        self.ref, = self.ax.plot(xr, yr, linestyle='--')
        self.act, = self.ax.plot(self.xs, self.ys, color='r')

        # self.ax.plot([3.0, 3.0], [0, 2], color='k')
        obs.draw(self.ax)

    def _calc_arm_pts(self, q):
        x0 = q[0]
        y0 = 0
        x = [x0, x0 + self.l1*np.cos(q[1]), x0 + self.l1*np.cos(q[1]) + self.l2*np.cos(q[1]+q[2])]
        y = [y0, y0 + self.l1*np.sin(q[1]), y0 + self.l1*np.sin(q[1]) + self.l2*np.sin(q[1]+q[2])]
        return x, y

    def _calc_body_pts(self, q):
        x0 = q[0]
        y0 = 0
        r = 0.5
        h = 0.25

        x = [x0, x0 - r, x0 - r, x0 + r, x0 + r, x0]
        y = [y0, y0, y0 - h, y0 - h, y0, y0]

        return x, y

    def update(self, q):
        ''' Update plot based on current transforms. '''
        xa, ya = self._calc_arm_pts(q)
        xb, yb = self._calc_body_pts(q)

        self.xs.append(xa[-1])
        self.ys.append(ya[-1])

        self.arm.set_xdata(xa)
        self.arm.set_ydata(ya)

        self.body.set_xdata(xb)
        self.body.set_ydata(yb)

        self.act.set_xdata(self.xs)
        self.act.set_ydata(self.ys)

        self.fig.canvas.draw()
        self.fig.canvas.flush_events()
