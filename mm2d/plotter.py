import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FFMpegWriter


class PointRenderer:
    def __init__(self, p, color='k'):
        self.p = p
        self.color = color

    def render(self, ax):
        self.plot, = ax.plot(self.p[0], self.p[1], 'o', color=self.color)

    def set_state(self, p):
        self.p = p

    def update_render(self):
        self.plot.set_xdata(self.p[0])
        self.plot.set_ydata(self.p[1])


class CircleRenderer:
    def __init__(self, r, pc):
        self.r = r
        self.pc = pc

    def render(self, ax):
        self.patch = plt.Circle(self.pc, self.r, color='k', fill=False)
        ax.add_patch(self.patch)

    def set_state(self, pc):
        self.pc = pc

    def update_render(self):
        self.patch.center = self.pc


class TrajectoryRenderer(object):
    def __init__(self, trajectory, ts):
        self.trajectory = trajectory
        self.ts = ts

    def render(self, ax):
        # reference trajectory doesn't change, so we can just unroll and plot
        # the whole thing now
        pr, *other = self.trajectory.sample(self.ts)
        xr = pr[:, 0]
        yr = pr[:, 1]
        self.ref, = ax.plot(xr, yr, linestyle='--')

    def update_render(self):
        # only renders once, no need to update
        pass


class TopDownHolonomicRenderer:
    def __init__(self, model, q0, length=1, width=0.5, render_path=True, render_collision=False):
        self.model = model
        self.q = q0
        self.render_path = render_path
        self.render_collision = render_collision
        self.xs = []
        self.ys = []
        self.length = length
        self.width = width
        self.base_r = np.sqrt(0.25*width**2 + 0.25*length**2)

    def calc_base_points(self, q):
        ''' Generate an array of points representing the base of the robot. '''
        pb = q[:2]
        θb = q[2]
        rx = self.length * 0.5
        ry = self.width * 0.5

        R = np.array([[np.cos(θb), -np.sin(θb)],
                      [np.sin(θb),  np.cos(θb)]])
        p = pb + R.dot([[-rx, rx, rx, -rx, -rx],
                        [-ry, -ry, ry, ry, -ry]]).T

        return p[:, 0], p[:, 1]

    def calc_arm_points(self, q):
        ''' Generate an array of points representing the arm of the robot. '''
        xb, yb, θb, θ1, θ2 = q
        x1 = xb + self.model.l1*np.cos(θb+θ1)
        x2 = x1 + self.model.l2*np.cos(θb+θ1+θ2)

        y1 = yb + self.model.l1*np.sin(θb+θ1)
        y2 = y1 + self.model.l2*np.sin(θb+θ1+θ2)

        x = np.array([xb, x1, x2])
        y = np.array([yb, y1, y2])

        return x, y

    def set_state(self, q):
        self.q = q

    def render(self, ax):
        xa, ya = self.calc_arm_points(self.q)
        xb, yb = self.calc_base_points(self.q)

        self.xs.append(xa[-1])
        self.ys.append(ya[-1])

        self.arm_plot, = ax.plot(xa, ya, color='k')
        self.body_plot, = ax.plot(xb, yb, color='k')

        if self.render_path:
            self.path_plot, = ax.plot(self.xs, self.ys, color='r')

        self.patch = plt.Circle(self.q[:2], self.base_r, linestyle='--', color='b', fill=False)
        if self.render_collision:
            ax.add_patch(self.patch)

    def update_render(self):
        xa, ya = self.calc_arm_points(self.q)
        xb, yb = self.calc_base_points(self.q)

        self.xs.append(xa[-1])
        self.ys.append(ya[-1])

        self.arm_plot.set_xdata(xa)
        self.arm_plot.set_ydata(ya)

        self.body_plot.set_xdata(xb)
        self.body_plot.set_ydata(yb)

        if self.render_path:
            self.path_plot.set_xdata(self.xs)
            self.path_plot.set_ydata(self.ys)

        self.patch.center = self.q[:2]


class ThreeInputRenderer(object):
    def __init__(self, model, q0, width=1, height=0.25, render_path=True):
        self.model = model
        self.q = q0
        self.render_path = render_path
        self.xs = []
        self.ys = []
        self.width = width
        self.height = height

    def calc_base_points(self, q):
        ''' Generate an array of points representing the base of the robot. '''
        x0 = q[0]
        y0 = 0
        r = self.width * 0.5
        h = self.height

        x = np.array([x0, x0 - r, x0 - r, x0 + r, x0 + r, x0])
        y = np.array([y0, y0, y0 - h, y0 - h, y0, y0])

        return x, y

    def calc_arm_points(self, q):
        ''' Generate an array of points representing the arm of the robot. '''
        x0 = q[0]
        x1 = x0 + self.model.l1*np.cos(q[1])
        x2 = x1 + self.model.l2*np.cos(q[1]+q[2])

        y0 = 0
        y1 = y0 + self.model.l1*np.sin(q[1])
        y2 = y1 + self.model.l2*np.sin(q[1]+q[2])

        x = np.array([x0, x1, x2])
        y = np.array([y0, y1, y2])

        return x, y

    def set_state(self, q):
        self.q = q

    def render(self, ax):
        xa, ya = self.calc_arm_points(self.q)
        xb, yb = self.calc_base_points(self.q)

        self.xs.append(xa[-1])
        self.ys.append(ya[-1])

        self.arm_plot, = ax.plot(xa, ya, color='k')
        self.body_plot, = ax.plot(xb, yb, color='k')

        if self.render_path:
            self.path_plot, = ax.plot(self.xs, self.ys, color='r')

    def update_render(self):
        xa, ya = self.calc_arm_points(self.q)
        xb, yb = self.calc_base_points(self.q)

        self.xs.append(xa[-1])
        self.ys.append(ya[-1])

        self.arm_plot.set_xdata(xa)
        self.arm_plot.set_ydata(ya)

        self.body_plot.set_xdata(xb)
        self.body_plot.set_ydata(yb)

        if self.render_path:
            self.path_plot.set_xdata(self.xs)
            self.path_plot.set_ydata(self.ys)


class PendulumRenderer(object):
    def __init__(self, model, X0, p0):
        self.model = model
        self.X = X0
        self.p = p0

    def set_state(self, X, p):
        self.X = X
        self.p = p

    def render(self, ax):
        # pendulum origin
        xo = self.p[0]
        yo = self.p[1]

        # pendulum top
        x = xo - self.model.length * np.sin(self.X[0])
        y = yo + self.model.length * np.cos(self.X[0])

        self.plot, = ax.plot([xo, x], [yo, y], color='k')

    def update_render(self):
        # pendulum origin
        xo = self.p[0]
        yo = self.p[1]

        # pendulum top
        x = xo - self.model.length * np.sin(self.X[0])
        y = yo + self.model.length * np.cos(self.X[0])

        self.plot.set_xdata([xo, x])
        self.plot.set_ydata([yo, y])


class Video:
    ''' Writer for a video recording of the real time plot. '''
    def __init__(self, name='recording.mp4', fps=10):
        self.name = name
        self.writer = FFMpegWriter(fps=fps)

    def setup(self, fig):
        self.writer.setup(fig, self.name)


class RealtimePlotter(object):
    ''' Real time plotter for the robot and associated trajectories. '''
    def __init__(self, renderers, video=None):
        self.renderers = renderers
        self.video = video

    def start(self, limits=[-1, 6, -1, 2], grid=False):
        ''' Launch the plot. '''
        plt.ion()

        self.fig = plt.figure()
        self.ax = plt.gca()

        if grid:
            plt.grid()

        self.ax.set_xlabel('x (m)')
        self.ax.set_ylabel('y (m)')
        self.ax.set_xlim(limits[:2])
        self.ax.set_ylim(limits[2:])

        for renderer in self.renderers:
            renderer.render(self.ax)

        if self.video:
            self.video.setup(self.fig)
            self.video.writer.grab_frame()

    def update(self):
        ''' Update plot based on current transforms. '''

        for renderer in self.renderers:
            renderer.update_render()

        self.fig.canvas.draw()
        self.fig.canvas.flush_events()

        if self.video:
            self.video.writer.grab_frame()

    def done(self):
        plt.ioff()
        if self.video:
            self.video.writer.finish()
