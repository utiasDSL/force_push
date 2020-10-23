import numpy as np
import matplotlib.pyplot as plt


class Wall(object):
    def __init__(self, x):
        self.x = x  # location

    def apply_force(self, p, dp):
        ''' Apply force based on end effector position p. '''
        if p[0] > self.x:
            dx = p[0] - self.x
            f = np.array([SPRING_CONST * dx + DAMPER_CONST * dp[0], 0])
        else:
            f = np.zeros(2)
        return f + F_SIGMA * np.random.randn(f.shape[0]) + F_MEAN

    def draw(self, ax):
        ax.plot([self.x, self.x], [0, 2], color='k')


def unit(a):
    return a / np.linalg.norm(a)


class Circle(object):
    def __init__(self, c, r, k, f_fric=10):
        self.c = c
        self.r = r
        self.k = k
        self.f_fric = f_fric

    def force(self, p):
        a = p[:2] - self.c
        d = np.linalg.norm(a)
        b = self.r * a / d

        # force is normal to surface -- we do not account for friction
        if d < self.r:
            dx = a - b
            f = self.k * dx  #+ self.b * v[:2]
        else:
            f = np.zeros(2)

        # if we have overcome friction, then reduce the force to friction and
        # calculate quasi-static movement of obstacle
        if f @ f > self.f_fric**2:
            f = self.f_fric * unit(f)
            dx_new = self.f_fric * unit(dx) / self.k
            movement = dx - dx_new
        else:
            movement = np.zeros(2)

        return f, movement
