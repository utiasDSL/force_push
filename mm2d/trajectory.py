import numpy as np


# TODO rewrite
def spiral(p0, ts):
    a = 0.1
    b = 0.08
    x = p0[0] + (a + b*ts) * np.cos(ts)
    y = p0[1] + (a + b*ts) * np.sin(ts)
    return x, y


class Point(object):
    def __init__(self, x0):
        self.x0 = x0

    def sample(self, t):
        return self.x0, np.zeros(self.x.shape)

    def unroll(self, ts, flatten=False):
        xs = np.tile(self.x0, (ts.shape[0], 1))
        vs = np.zeros(xs.shape)
        if flatten:
            return xs.flatten(), vs.flatten()
        return xs, vs


class Line(object):
    def __init__(self, p0, v):
        self.p0 = p0
        self.v = v

    def sample(self, t):
        pd = self.p0 + self.v * t
        vd = self.v
        return pd, vd

    def unroll(self, ts, flatten=False):
        ''' Unroll the trajectory over the given times ts.
            Returns the desired position and velocity arrays.
            If flatten=True, then the arrays are flattened before returning. '''
        pds = self.p0 + np.array([self.v * t for t in ts])
        vds = np.array([self.v for _ in ts])
        if flatten:
            return pds.flatten(), vds.flatten()
        return pds, vds


class Circle(object):
    def __init__(self, p0, r, duration):
        self.p0 = p0
        self.r = r
        self.duration = duration

    def sample(self, t):
        a = 2.0 * np.pi * t / self.duration - np.pi

        # position
        x = self.p0[0] + self.r * np.cos(a) + self.r
        y = self.p0[1] + self.r * np.sin(a)
        theta = a
        p = np.array([x, y, theta])

        # velocity
        da = 2.0 * np.pi * np.ones(t.shape) / self.duration
        vx = -self.r * np.sin(a) * da
        vy = self.r * np.cos(a) * da
        vtheta = da
        v = np.array([vx, vy, vtheta])

        # truncate in the case that theta is not included
        n = self.p0.shape[0]
        return p[:n], v[:n]

    def unroll(self, ts, flatten=False):
        # Because of the way sample is written for Circle, it neatly extended
        # to vector-valued time inputs
        ps, vs = self.sample(ts)
        ps = ps.T
        vs = vs.T
        if flatten:
            return ps.flatten(), vs.flatten()
        return ps, vs
