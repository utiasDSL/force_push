import time
import numpy as np
from scipy.optimize import minimize_scalar
from scipy.integrate import quad

from force_push import util

import IPython


class CirclePath:
    """Circle path to track."""

    def __init__(self, radius):
        # TODO generalize to centers not at the origin
        self.radius = radius

    def compute_closest_point(self, p):
        return self.radius * self.compute_normal(p)

    def compute_normal(self, p):
        """Outward-facing normal."""
        return util.unit(p)

    def compute_lateral_offset(self, p):
        c = self.compute_closest_point(p)
        n = self.compute_normal(p)
        return -n @ (p - c)

    def compute_travel_direction(self, p, d=0):
        c = self.compute_point_ahead(p, d)
        return util.rot2d(np.pi / 2) @ self.compute_normal(c)
        # return util.rot2d(np.pi / 2) @ self.compute_normal(p)

    def compute_point_ahead(self, p, d):
        """Compute point distance d ahead of closest point on path to p."""
        c1 = self.compute_closest_point(p)
        φ = d / self.radius
        c2 = util.rot2d(φ) @ c1
        return c2


def translate_segments(segments, offset):
    """Translate all segments by a specified offset."""
    return [segment.offset(offset) for segment in segments]


class LineSegment:
    """Path segment that is a straight line."""

    def __init__(self, v1, v2, infinite=False):
        self.v1 = np.array(v1)
        self.v2 = np.array(v2)
        self.direction = util.unit(self.v2 - self.v1)
        self.infinite = infinite
        self.length = np.inf if infinite else np.linalg.norm(self.v2 - self.v1)

    def offset(self, Δ):
        return LineSegment(v1=self.v1 + Δ, v2=self.v2 + Δ, infinite=self.infinite)

    def closest_point_and_direction(self, p):
        return self.closest_point_and_distance(p)[0], self.direction

    def closest_point_and_distance(self, p, tol=1e-8):
        r = p - self.v1
        proj = self.direction @ r  # project onto the line

        if self.infinite:
            # closest point is beyond the end of the segment
            if proj <= 0:
                return self.v1, np.linalg.norm(r)
            c = self.v1 + proj * self.direction
            return c, np.linalg.norm(p - c)
        else:
            v = self.v2 - self.v1

            # degenerate case when the line segment has zero length (i.e., is a point)
            if self.length < tol:
                return self.v1, np.linalg.norm(r)

            # closest point is beyond the end of the segment
            if proj <= 0:
                return self.v1, np.linalg.norm(r)
            if proj >= self.length:
                return self.v2, np.linalg.norm(p - self.v2)

            c = self.v1 + proj * self.direction
            return c, np.linalg.norm(p - c)

    def point_at_distance(self, d):
        """Compute the point that is distance ``d`` from the start of the segment."""
        assert d >= 0
        if d > self.length:
            raise ValueError(f"Segment is shorter than distance {d}")
        return self.v1 + d * self.direction


class QuadBezierSegment:
    """Path segment that is a quadratic Bezier curve."""

    def __init__(self, v1, v2, v3):
        self.v1 = np.array(v1)
        self.v2 = np.array(v2)
        self.v3 = np.array(v3)

        self.length = quad(lambda t: np.linalg.norm(self._time_derivative(t)), 0, 1)[0]

    def offset(self, Δ):
        return QuadBezierSegment(v1=self.v1 + Δ, v2=self.v2 + Δ, v3=self.v3 + Δ)

    def _evaluate(self, t):
        assert 0 <= t <= 1
        return (1 - t) ** 2 * self.v1 + 2 * (1 - t) * t * self.v2 + t**2 * self.v3

    def _time_derivative(self, t):
        assert 0 <= t <= 1
        return 2 * (1 - t) * (self.v2 - self.v1) + 2 * t * (self.v3 - self.v2)

    def closest_point_and_direction(self, p):

        # minimize distance between curve and point p
        def fun(t):
            b = self._evaluate(t)
            return (p - b) @ (p - b)

        res = minimize_scalar(fun, bounds=(0, 1), method="bounded")
        t = res.x

        closest = self._evaluate(t)
        direction = util.unit(self._time_derivative(t))
        return closest, direction

    def point_at_distance(self, d):
        """Compute the point that is distance ``d`` from the start of the segment."""
        assert d >= 0
        if d > self.length:
            raise ValueError(f"Segment is shorter than distance {d}")

        def fun(s):
            L = quad(lambda t: np.linalg.norm(self._time_derivative(t)), 0, s)[0]
            return (d - L)**2

        res = minimize_scalar(fun, bounds=(0, 1), method="bounded")
        return self._evaluate(res.x)


class SegmentPath:
    def __init__(self, segments, origin=None):
        """Construct the path from a list of segments.

        It is the user's responsibility to ensure suitable continuity between
        the segments.

        If `origin` is not `None`, then the vertices of each segment will be
        offset by `origin`.
        """
        if origin is None:
            self.segments = segments
        else:
            self.segments = translate_segments(segments, origin)

    @classmethod
    def line(cls, direction, origin=None):
        """Construct an infinite line path."""
        if origin is None:
            origin = np.zeros(2)
        v1 = origin
        v2 = origin + direction
        return cls([LineSegment(v1, v2, infinite=True)])

    def compute_closest_point(self, p):
        minimum = (np.inf, None, None)
        for segment in self.segments:
            closest, direction = segment.closest_point_and_direction(p)
            dist = np.linalg.norm(p - closest)
            if dist < minimum[0]:
                minimum = (dist, closest, direction)
        return minimum[1]

    def compute_direction_and_offset(self, p):
        """Compute travel direction and lateral offset for a point p."""
        minimum = (np.inf, None, None)
        for segment in self.segments:
            closest, direction = segment.closest_point_and_direction(p)
            dist = np.linalg.norm(p - closest)
            if dist < minimum[0]:
                minimum = (dist, closest, direction)

        _, closest, direction = minimum
        R = util.rot2d(np.pi / 2)
        perp = R @ direction
        offset = perp @ (p - closest)
        return direction, offset

    def get_plotting_coords(self, n_bez=25, dist=0):
        """Get (x, y) coordinates of the path for plotting.

        Parameters:
            n_bez: number of points to discretize each Bezier curve

        Returns a shape (n, 2) array of (x, y) coordinates.
        """
        vertices = []
        ts = np.linspace(0, 1, n_bez)
        for segment in self.segments:
            if type(segment) is LineSegment:
                vertices.extend([segment.v1, segment.v2])
            elif type(segment) is QuadBezierSegment:
                vertices.extend([segment._evaluate(t) for t in ts])

        # if the last segment is an infinite line, we append a final vertex a
        # distance `dist` along the line from the last vertex
        last_seg = self.segments[-1]
        if type(last_seg) is LineSegment and last_seg.infinite:
            vertices.append(last_seg.v2 + dist * last_seg.direction)

        return np.array(vertices)

    def point_at_distance(self, d):
        """Compute the point that is distance ``d`` from the start of the path."""
        assert d >= 0

        L = 0
        for segment in self.segments:
            if L + segment.length > d:
                return segment.point_at_distance(d - L)
            L += segment.length
        raise ValueError(f"Path is shorter than distance {d}")
