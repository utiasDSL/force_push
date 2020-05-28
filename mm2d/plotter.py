import matplotlib.pyplot as plt

from trajectory import unroll


class RobotPlotter(object):
    ''' Real time plotter for the robot and associated trajectories. '''
    def __init__(self, model, trajectory):
        self.model = model
        self.trajectory = trajectory
        self.xs = []
        self.ys = []

    def start(self, q0, ts, obstacles=None):
        ''' Launch the plot. '''
        plt.ion()

        self.fig = plt.figure()
        self.ax = plt.gca()

        self.ax.set_xlabel('x (m)')
        self.ax.set_ylabel('y (m)')
        self.ax.set_xlim([-1, 6])
        self.ax.set_ylim([-1, 2])

        xa, ya = self.model.arm(q0)
        xb, yb = self.model.base(q0)

        self.xs.append(xa[-1])
        self.ys.append(ya[-1])

        # reference trajectory doesn't change
        pr = unroll(ts, self.trajectory)
        xr = pr[:, 0]
        yr = pr[:, 1]
        self.ref, = self.ax.plot(xr, yr, linestyle='--')

        self.arm, = self.ax.plot(xa, ya, color='k')
        self.body, = self.ax.plot(xb, yb, color='k')
        self.act, = self.ax.plot(self.xs, self.ys, color='r')

        # obs.draw(self.ax)

    def update(self, q):
        ''' Update plot based on current transforms. '''
        xa, ya = self.model.arm(q0)
        xb, yb = self.model.base(q0)

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
