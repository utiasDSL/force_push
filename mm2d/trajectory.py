import numpy as np

import IPython


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
    ''' Constant-velocity linear trajectory.
        Parameters:
          p0: starting position
          v:  desired velocity
    '''
    def __init__(self, p0, v):
        self.p0 = p0

        # TODO v should be a scalar and handled like the polygon class below
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


class Polygon(object):
    def __init__(self, points, v):
        self.points = points

        # calculate time it takes to traverse each line segment
        delta = points[1:, :] - points[:-1, :]
        dists = np.sqrt(np.sum(delta**2, axis=1))
        durations = dists / v
        self.times = np.zeros(durations.shape[0] + 1)
        self.times[1:] = np.cumsum(durations)

        # velocity is projected onto the unit vector for each line segment
        self.velocities = v * delta / dists[:, None]

    def sample(self, t):

        # if the sample time is at or past the end of the trajectory, return
        # the last point with zero velocity
        if t >= self.times[-1]:
            p = self.points[-1, :]
            v = np.zeros_like(p)
            return p, v

        idx = np.searchsorted(self.times, t, side='right') - 1
        t1 = self.times[idx]
        t2 = self.times[idx+1]
        p1 = self.points[idx]
        p2 = self.points[idx+1]

        # linearly interpolate the current position
        t_interp = (t - t1) / (t2 - t1)
        p = p1 + (p2 - p1) * t_interp

        v = self.velocities[idx, :]

        return p, v

    def unroll(self, ts, flatten=False):
        # partition in times during the active trajectory, and those when it is
        # done
        t_active = ts[ts < self.times[-1]]
        t_done = ts[ts >= self.times[-1]]

        idx = np.searchsorted(self.times, t_active, side='right') - 1
        t1 = self.times[idx]
        t2 = self.times[idx+1]
        p1 = self.points[idx]
        p2 = self.points[idx+1]

        # linearly interpolate the current position
        t_interp = (t_active - t1) / (t2 - t1)
        p_active = p1 + (p2 - p1) * t_interp[:, None]

        v_active = self.velocities[idx, :]

        # times after the trajectory is done just specify the last point and
        # zero velocity
        p_done = np.tile(self.points[-1, :], (t_done.shape[0], 1))
        v_done = np.zeros_like(p_done)

        p = np.concatenate((p_active, p_done), axis=0)
        v = np.concatenate((v_active, v_done), axis=0)

        if flatten:
            return p.flatten(), v.flatten()
        return p, v


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
        # Because of the way sample is written for Circle, it neatly extends
        # to vector-valued time inputs
        ps, vs = self.sample(ts)
        ps = ps.T
        vs = vs.T
        if flatten:
            return ps.flatten(), vs.flatten()
        return ps, vs
