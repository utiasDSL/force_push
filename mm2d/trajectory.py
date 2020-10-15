import numpy as np


# TODO rewrite
def spiral(p0, ts):
    a = 0.1
    b = 0.08
    x = p0[0] + (a + b*ts) * np.cos(ts)
    y = p0[1] + (a + b*ts) * np.sin(ts)
    return x, y


class LinearTimeScaling:
    ''' Linear time-scaling: constant velocity. '''
    def __init__(self, duration):
        self.duration = duration

    def eval(self, t):
        s = t / self.duration
        ds = np.ones_like(t) / self.duration
        dds = np.zeros_like(t)
        return s, ds, dds


class CubicTimeScaling:
    ''' Cubic time-scaling: zero velocity at end points. '''
    def __init__(self, duration):
        self.coeffs = np.array([0, 0, 3 / duration**2, -2 / duration**3])

    def eval(self, t):
        s = self.coeffs.dot([np.ones_like(t), t, t**2, t**3])
        ds = self.coeffs[1:].dot([np.ones_like(t), 2*t, 3*t**2])
        dds = self.coeffs[2:].dot([2*np.ones_like(t), 6*t])
        # s, ds, dds = np.atleast_1d(s, ds, dds)
        return s, ds, dds


class QuinticTimeScaling:
    ''' Quintic time-scaling: zero velocity and acceleration at end points. '''
    def __init__(self, T):
        A = np.array([[1, 0, 0, 0, 0, 0],
                      [1, T, T**2, T**3, T**4, T**5],
                      [0, 1, 0, 0, 0, 0],
                      [0, 1, 2*T, 3*T**2, 4*T**3, 5*T**4],
                      [0, 0, 2, 0, 0, 0],
                      [0, 0, 2, 6*T, 12*T**2, 20*T**3]])
        b = np.array([0, 1, 0, 0, 0, 0])
        self.coeffs = np.linalg.solve(A, b)

    def eval(self, t):
        s = self.coeffs.dot([np.ones_like(t), t, t**2, t**3, t**4, t**5])
        ds = self.coeffs[1:].dot([np.ones_like(t), 2*t, 3*t**2, 4*t**3, 5*t**4])
        dds = self.coeffs[2:].dot([2*np.ones_like(t), 6*t, 12*t**2, 20*t**3])
        return s, ds, dds


class TrapezoidalTimeScaling:
    def __init__(self, duration):
        pass


class CubicBezier:
    def __init__(self, points, timescaling, duration):
        ''' Points should be a (4*2) array of control points, with p[0, :]
            being the initial position and p[-1, :] the final. '''
        self.points = points
        self.timescaling = timescaling
        self.duration = duration

    def sample(self, t):
        s, ds, dds = self.timescaling.eval(t)

        cp = np.array([(1-s)**3, 3*(1-s)**2*s, 3*(1-s)*s**2, s**3])
        p = cp.T.dot(self.points)

        cv = np.array([3*(1-s)**2, 6*(1-s)*s, 3*s**2])
        dpds = cv.T.dot(self.points[1:, :] - self.points[:-1, :])
        v = (dpds.T * ds).T

        ca = np.array([6*(1-s), 6*s])
        dpds2 = ca.T.dot(self.points[2:, :] - 2*self.points[1:-1, :] + self.points[:-2, :])
        a = (dpds.T * dds + dpds2.T * ds**2).T

        return p, v, a

    # def unroll(self, ts, flatten=False):
    #     s, ds, dds = self.timescaling.eval(ts)
    #
    #     cp = np.array([(1-s)**3, 3*(1-s)**2*s, 3*(1-s)*s**2, s**3])
    #     p = cp.T.dot(self.points)
    #
    #     cv = np.array([3*(1-s)**2, 6*(1-s)*s, 3*s**2])
    #     dpds = cv.T.dot(self.points[1:, :] - self.points[:-1, :])
    #     v = dpds * ds[:, None]
    #
    #     ca = np.array([6*(1-s), 6*s])
    #     dpds2 = ca.T.dot(self.points[2:, :] - 2*self.points[1:-1, :] + self.points[:-2, :])
    #     a = dpds * dds[:, None] + dpds2 * ds[:, None]**2
    #
    #     if flatten:
    #         return p.flatten(), v.flatten(), a.flatten()
    #     return p, v, a


class PointToPoint:
    def __init__(self, p0, p1, timescaling, duration):
        self.p0 = p0
        self.p1 = p1
        self.timescaling = timescaling
        self.duration = duration

    def sample(self, t):
        s, ds, dds = self.timescaling.eval(t)
        p = self.p0 + s * (self.p1 - self.p0)
        v = ds * (self.p1 - self.p0)
        a = dds * (self.p1 - self.p0)
        return p, v, a

    def unroll(self, ts, flatten=False):
        s, ds, dds = self.timescaling.eval(ts)
        p = self.p0 + s[:, None] * (self.p1 - self.p0)
        v = ds[:, None] * (self.p1 - self.p0)
        a = dds[:, None] * (self.p1 - self.p0)
        if flatten:
            return p.flatten(), v.flatten(), a.flatten()
        return p, v, a


class CircleS:
    def __init__(self, p0, r, timescaling, duration):
        self.r = r
        self.pc = p0 + [r, 0]  # start midway up left side of circle
        self.timescaling = timescaling
        self.duration = duration

    def sample(self, t):
        s, ds, dds = self.timescaling.eval(t)

        cs = np.cos(2*np.pi*s - np.pi)
        ss = np.sin(2*np.pi*s - np.pi)
        p = self.pc + self.r * np.array([cs, ss])

        dpds = 2*np.pi*self.r * np.array([-ss, cs])
        v = dpds * ds

        dpds2 = 4*np.pi**2*self.r * np.array([-cs, -ss])
        a = dpds * dds + dpds2 * ds**2

        return p, v, a

    def unroll(self, ts, flatten=False):
        s, ds, dds = self.timescaling.eval(ts)

        cs = np.cos(2*np.pi*s - np.pi)
        ss = np.sin(2*np.pi*s - np.pi)
        p = self.pc + self.r * np.array([cs, ss]).T

        dpds = 2*np.pi*self.r * np.array([-ss, cs]).T
        v = dpds * ds[:, None]

        dpds2 = 4*np.pi**2*self.r * np.array([-cs, -ss]).T
        a = dpds * dds[:, None] + dpds2 * ds[:, None]**2

        if flatten:
            return p.flatten(), v.flatten(), a.flatten()
        return p, v, a


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
    ''' Constant-acceleration linear trajectory.
        Parameters:
          p0: starting position
          v:  desired velocity
    '''
    def __init__(self, p0, v0, a):
        self.p0 = p0
        self.v0 = v0
        self.a = a

    def sample(self, t):
        ad = self.a
        vd = self.v0 + ad * t
        pd = self.p0 + vd * t
        return pd, vd, ad

    def unroll(self, ts, flatten=False):
        ''' Unroll the trajectory over the given times ts.
            Returns the desired position and velocity arrays.
            If flatten=True, then the arrays are flattened before returning. '''
        ads = np.tile(self.a, (len(ts), 1))
        vds = self.v0 + ads * ts[:, None]
        pds = self.p0 + vds * ts[:, None]
        if flatten:
            return pds.flatten(), vds.flatten(), ads.flatten()
        return pds, vds, ads


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
