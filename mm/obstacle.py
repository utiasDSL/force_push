import numpy as np
import matplotlib.pyplot as plt


# force model params
SPRING_CONST = 1e3
DAMPER_CONST = 0.0
F_MEAN = 0.0
F_SIGMA = 0.0


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


class Circle(object):
    def __init__(self, c, r):
        self.c = c
        self.r = r

    def apply_force(self, p, dp):
        a = p[:2] - self.c
        d = np.linalg.norm(a)
        b = self.r * a / d
        if d < self.r:
            dx = a - b
            f = SPRING_CONST * dx + DAMPER_CONST * dp[:2]
        else:
            f = np.zeros(2)
        return f + F_SIGMA * np.random.randn(f.shape[0]) + F_MEAN

    def draw(self, ax):
        ax.add_patch(plt.Circle(self.c, self.r, color='k', fill=False))
