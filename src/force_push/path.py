from collections import namedtuple

import numpy as np
from scipy.optimize import minimize_scalar
from scipy.integrate import quad

from force_push import util

import IPython


def translate_segments(segments, offset):
    """Translate all segments by a specified offset."""
    return [segment.offset(offset) for segment in segments]


ClosestPointInfo = namedtuple(
    "ClosestPointInfo", ["point", "deviation", "distance_from_start", "direction"]
)


class LineSegment:
    """Path segment that is a straight line."""

    def __init__(self, v1, v2, infinite=False):
        self.v1 = np.array(v1)
        self.v2 = np.array(v2)
        self.direction = util.unit(self.v2 - self.v1)
        self.infinite = infinite

    @property
    def length(self):
        return np.inf if self.infinite else np.linalg.norm(self.v2 - self.v1)

    def offset(self, Δ):
        return LineSegment(v1=self.v1 + Δ, v2=self.v2 + Δ, infinite=self.infinite)

    def closest_point_info(self, p, tol=1e-8):
        r = p - self.v1
        proj = self.direction @ r  # project onto the line

        # compute closest point (and its distance from the start of the path)
        if self.length < tol or proj <= 0:
            closest = self.v1
            dist_from_start = 0
        elif not self.infinite and proj >= self.length:
            closest = self.v2
            dist_from_start = self.length
        else:
            closest = self.v1 + proj * self.direction
            dist_from_start = proj

        # deviation from the path
        deviation = np.linalg.norm(p - closest)

        return ClosestPointInfo(
            point=closest,
            deviation=deviation,
            distance_from_start=dist_from_start,
            direction=self.direction,
        )

    def point_at_distance(self, d):
        """Compute the point that is distance ``d`` from the start of the segment."""
        assert d >= 0
        if d > self.length:
            raise ValueError(f"Segment is shorter than distance {d}")
        return self.v1 + d * self.direction


# class QuadBezierSegment:
#     """Path segment defined by a quadratic Bezier curve."""
#
#     def __init__(self, v1, v2, v3):
#         self.v1 = np.array(v1)
#         self.v2 = np.array(v2)
#         self.v3 = np.array(v3)
#
#         self.length = quad(lambda t: np.linalg.norm(self._time_derivative(t)), 0, 1)[0]
#
#     def offset(self, Δ):
#         return QuadBezierSegment(v1=self.v1 + Δ, v2=self.v2 + Δ, v3=self.v3 + Δ)
#
#     def _evaluate(self, t):
#         assert 0 <= t <= 1
#         return (1 - t) ** 2 * self.v1 + 2 * (1 - t) * t * self.v2 + t**2 * self.v3
#
#     def _time_derivative(self, t):
#         assert 0 <= t <= 1
#         return 2 * (1 - t) * (self.v2 - self.v1) + 2 * t * (self.v3 - self.v2)
#
#     def closest_point_and_direction(self, p):
#
#         # minimize distance between curve and point p
#         def fun(t):
#             b = self._evaluate(t)
#             return (p - b) @ (p - b)
#
#         res = minimize_scalar(fun, bounds=(0, 1), method="bounded")
#         t = res.x
#
#         closest = self._evaluate(t)
#         direction = util.unit(self._time_derivative(t))
#         return closest, direction
#
#     def point_at_distance(self, d):
#         """Compute the point that is distance ``d`` from the start of the segment."""
#         assert d >= 0
#         if d > self.length:
#             raise ValueError(f"Segment is shorter than distance {d}")
#
#         def fun(s):
#             L = quad(lambda t: np.linalg.norm(self._time_derivative(t)), 0, s)[0]
#             return (d - L) ** 2
#
#         res = minimize_scalar(fun, bounds=(0, 1), method="bounded")
#         return self._evaluate(res.x)


class CircularArcSegment:
    """Path segment defined by a circular arc."""

    def __init__(self, center, point, angle):
        assert 0 <= angle <= 2 * np.pi

        self.center = np.array(center)
        self.v1 = np.array(point)
        self.angle = angle
        self.radius = np.linalg.norm(self._arm)

        self.v2 = self.center + util.rot2d(angle) @ self._arm

    @property
    def length(self):
        return self.radius * self.angle

    @property
    def _arm(self):
        """Vector from the center to the start of the arc."""
        return self.v1 - self.center

    def offset(self, Δ):
        return CircularArcSegment(
            center=self.center + Δ, point=self.v1 + Δ, angle=self.angle
        )

    def closest_point_info(self, p):
        if np.allclose(p, self.center):
            # at the center point all points on the arc are equally close
            # TODO should I do more intelligent tie-breaking?
            closest = self.v2
            dist_from_start = self.length
        else:
            # compute closest point on the full circle
            c0 = self.radius * util.unit(p - self.center)
            angle = util.signed_angle(self._arm, c0)
            if angle < 0:
                angle += 2 * np.pi

            # check if that point is inside the arc
            # if not, then one of the end points is the closest point
            if angle <= self.angle:
                closest = self.center + c0
                dist_from_start = self.radius * angle
            else:
                endpoints = np.vstack((self.v1, self.v2))
                endpoint_dists = np.array([0, self.length])
                idx = np.argmin(np.linalg.norm(endpoints - p, axis=1))
                closest = endpoints[idx, :]
                dist_from_start = endpoint_dists[idx]

        direction = util.rot2d(np.pi / 2) @ util.unit(closest - self.center)
        deviation = np.linalg.norm(p - closest)
        return ClosestPointInfo(
            point=closest,
            deviation=deviation,
            distance_from_start=dist_from_start,
            direction=direction,
        )

    def point_at_distance(self, d):
        """Compute the point that is distance ``d`` from the start of the segment."""
        assert d >= 0
        if d > self.length:
            raise ValueError(f"Segment is shorter than distance {d}")

        angle = d / self.radius
        return self.center + util.rot2d(angle) @ self._arm


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

    def compute_direction_and_offset(self, p):
        """Compute travel direction and lateral offset for a point p."""
        minimum = ClosestPointInfo(
            point=None, deviation=np.inf, distance_from_start=None, direction=None
        )
        for segment in self.segments:
            info = segment.closest_point_info(p)
            if info.deviation < minimum.deviation:
                minimum = info

        R = util.rot2d(np.pi / 2)
        perp = R @ minimum.direction
        offset = perp @ (p - minimum.point)
        return minimum.direction, offset

    def compute_closest_point_info(self, p):
        infos = [segment.closest_point_info(p) for segment in self.segments]
        idx = np.argmin([info.deviation for info in infos])

        # total dist from start is the dist from the start of the segment with
        # the closest point plus the length of all preceeding segments
        dist_from_start = infos[idx].distance_from_start
        for i in range(idx):
            dist_from_start += self.segments[i].length

        return ClosestPointInfo(
            point=infos[idx].point,
            deviation=infos[idx].deviation,
            distance_from_start=dist_from_start,
            direction=infos[idx].direction,
        )

    def get_plotting_coords(self, n_bez=25, dist=0):
        """Get (x, y) coordinates of the path for plotting.

        Parameters:
            n_bez: number of points to discretize each Bezier curve

        Returns a shape (n, 2) array of (x, y) coordinates.
        """
        vertices = []
        for segment in self.segments:
            if type(segment) is LineSegment:
                vertices.extend([segment.v1, segment.v2])
            # elif type(segment) is QuadBezierSegment:
            #     ts = np.linspace(0, 1, n_bez)
            #     vertices.extend([segment._evaluate(t) for t in ts])
            elif type(segment) is CircularArcSegment:
                ds = np.linspace(0, segment.length, n_bez)
                vertices.extend([segment.point_at_distance(d) for d in ds])
            else:
                raise TypeError(f"Don't know how to plot {segment}")

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
