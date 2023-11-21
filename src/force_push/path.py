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
    "ClosestPointInfo",
    ["point", "deviation", "distance_from_start", "direction", "offset"],
)


def compute_offset(point, direction, p):
    R = util.rot2d(np.pi / 2)
    perp = R @ direction
    return perp @ (p - point)


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

    def closest_point_info(self, p, min_dist_from_start=0, tol=1e-8):
        assert 0 <= min_dist_from_start <= self.length

        # project onto the line
        proj = self.direction @ (p - self.v1)

        # compute closest point (and its distance from the start of the segment)
        if self.length < tol:
            closest = self.v1
            dist_from_start = min_dist_from_start
        elif not self.infinite and proj >= self.length:
            closest = self.v2
            dist_from_start = self.length
        else:
            dist = max(proj, min_dist_from_start)
            closest = self.v1 + dist * self.direction
            dist_from_start = dist

        # deviation from the path
        deviation = np.linalg.norm(p - closest)

        offset = compute_offset(point=self.v1, direction=self.direction, p=p)

        # if deviation > 0.1:
        #     IPython.embed()

        return ClosestPointInfo(
            point=closest,
            deviation=deviation,
            distance_from_start=dist_from_start,
            direction=self.direction,
            offset=offset,
        )

    def point_at_distance(self, d):
        """Compute the point that is distance ``d`` from the start of the segment."""
        assert d >= 0
        if d > self.length:
            raise ValueError(f"Segment is shorter than distance {d}")
        return self.v1 + d * self.direction


class CircularArcSegment:
    """Path segment defined by a circular arc."""

    def __init__(self, center, point, angle):
        assert 0 <= angle <= 2 * np.pi

        self.center = np.array(center)
        self.v1 = np.array(point)
        self.angle = angle
        self.radius = np.linalg.norm(self._arm)

        assert self.radius > 0

        self.v2 = self._rotate_from_start(angle)

    @property
    def length(self):
        return self.radius * self.angle

    @property
    def _arm(self):
        """Vector from the center to the start of the arc."""
        return self.v1 - self.center

    def _rotate_from_start(self, angle):
        """Compute the point on the circle that is a rotation ``angle`` from the start."""
        return self.center + util.rot2d(angle) @ self._arm

    def offset(self, Δ):
        """Create a new ``CircularArcSegment`` that is translated by ``Δ``."""
        return CircularArcSegment(
            center=self.center + Δ, point=self.v1 + Δ, angle=self.angle
        )

    def closest_point_info(self, p, min_dist_from_start=0):
        assert 0 <= min_dist_from_start <= self.length

        min_angle_from_start = min_dist_from_start / self.radius

        if np.allclose(p, self.center):
            # at the center point all points on the arc are equally close; take
            # the one closest to the allowable start of the arc
            closest = self._rotate_from_start(min_angle_from_start)
            dist_from_start = min_dist_from_start
            # TODO for consistency with previous work
            # closest = self.v2
            # dist_from_start = self.length
        else:
            # compute closest point on the full circle
            c0 = self.radius * util.unit(p - self.center)
            angle = util.signed_angle(self._arm, c0)
            if angle < 0:
                angle += 2 * np.pi

            # check if that point is inside the arc
            # if not, then one of the end points is the closest point
            if min_angle_from_start <= angle <= self.angle:
                closest = self.center + c0
                dist_from_start = self.radius * angle
            else:
                start = self._rotate_from_start(min_angle_from_start)
                end = self.v2
                endpoints = np.vstack((start, end))
                endpoint_dists = np.array([min_dist_from_start, self.length])

                idx = np.argmin(np.linalg.norm(endpoints - p, axis=1))
                closest = endpoints[idx, :]
                dist_from_start = endpoint_dists[idx]

        direction = util.rot2d(np.pi / 2) @ util.unit(closest - self.center)
        deviation = np.linalg.norm(p - closest)
        offset = compute_offset(point=closest, direction=direction, p=p)
        return ClosestPointInfo(
            point=closest,
            deviation=deviation,
            distance_from_start=dist_from_start,
            direction=direction,
            offset=offset,
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

        # check that the segments actually form a connected path
        for i in range(1, len(self.segments)):
            if not np.allclose(self.segments[i - 1].v2, self.segments[i].v1):
                d = np.linalg.norm(self.segments[i - 1].v2 - self.segments[i].v1)
                raise ValueError(f"Segment {i - 1} and {i} are {d} distance apart!")

        self._seg_lengths = np.array([seg.length for seg in segments])
        self._cum_seg_lengths = np.cumsum(self._seg_lengths)

    @classmethod
    def line(cls, direction, origin=None):
        """Construct an infinite line path."""
        if origin is None:
            origin = np.zeros(2)
        v1 = origin
        v2 = origin + direction
        return cls([LineSegment(v1, v2, infinite=True)])

    def compute_closest_point_info(self, p, min_dist_from_start=0):
        assert min_dist_from_start >= 0

        # only start looking starting at this segment; the others are two close
        # to the start
        min_idx = np.argmax(self._cum_seg_lengths >= min_dist_from_start)

        # the minimum distance for each segment is 0, except possibly for the
        # first one
        min_dist_from_seg_starts = np.zeros(len(self.segments) - min_idx)
        if min_idx == 0:
            min_dist_from_seg_starts[0] = min_dist_from_start
        else:
            min_dist_from_seg_starts[0] = (
                min_dist_from_start - self._cum_seg_lengths[min_idx - 1]
            )

        # find the segment with minimum deviation
        infos = []
        for seg, min_dist in zip(self.segments[min_idx:], min_dist_from_seg_starts):
            infos.append(seg.closest_point_info(p, min_dist_from_start=min_dist))
        idx = np.argmin([info.deviation for info in infos])
        info = infos[idx]
        seg_idx = min_idx + idx

        # total dist from start is the dist from the start of the segment with
        # the closest point plus the length of all preceeding segments
        dist_from_start = info.distance_from_start
        if seg_idx > 0:
            dist_from_start += self._cum_seg_lengths[seg_idx - 1]

        return ClosestPointInfo(
            point=info.point,
            deviation=info.deviation,
            distance_from_start=dist_from_start,
            direction=info.direction,
            offset=info.offset,
        )

    def get_plotting_coords(self, n_bez=25, dist=0):
        """Get (x, y) coordinates of the path for plotting.

        Parameters
        ----------
        n_bez :
            Number of points to discretize each curved segments.
        dist :
            If the final segment is an infinite line, add a final point
            ``dist`` along this line from the previous vertex.

        Returns
        -------
        :
            A shape (n, 2) array of (x, y) coordinates.
        """
        vertices = []
        for segment in self.segments:
            if type(segment) is LineSegment:
                vertices.extend([segment.v1, segment.v2])
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
