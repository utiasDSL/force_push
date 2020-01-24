import numpy as np


def spiral(p0, ts):
    a = 0.1
    b = 0.08
    x = p0[0] + (a + b*ts) * np.cos(ts)
    y = p0[1] + (a + b*ts) * np.sin(ts)
    return x, y


def point(p0, ts):
    x = p0[0] * np.ones(ts.shape[0])
    y = p0[1] * np.ones(ts.shape[0])
    return x, y


def line(p0, ts):
    v = 0.125
    x = p0[0] + np.array([v*t for t in ts])
    y = p0[1] * np.ones(ts.shape[0])
    theta = np.ones(ts.shape[0]) * np.pi * 0.25
    return x, y, theta


def circle(p0, ts):
    R = 0.5
    t_max = ts[-1]
    angles = np.array([2.0 * np.pi * t / t_max for t in ts]) - np.pi
    x = p0[0] + R * np.cos(angles) + R
    y = p0[1] + R * np.sin(angles)
    # theta = np.ones(ts.shape[0]) * 0
    theta = angles
    return x, y, theta


class Line(object):
    def __init__(self, p0, v):
        self.p0 = p0
        self.v = v

    def sample(self, t):
        x = self.p0[0] + self.v[0] * t
        y = self.p0[1] + self.v[1] * t
        return x, y
