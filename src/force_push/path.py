import numpy as np

from force_push import util


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


# TODO deprecate in favour of SegmentPath.line
class StraightPath:
    """Straight path to track in given direction and passing through origin."""

    def __init__(self, direction, origin=None):
        if origin is None:
            origin = np.zeros(2)
        self.origin = np.array(origin)
        self.direction = np.array(direction)
        self.perp = util.rot2d(np.pi / 2) @ direction

    def compute_closest_point(self, p):
        dist = (p - self.origin) @ self.direction
        return dist * self.direction + self.origin

    def compute_travel_direction(self, p):
        return self.direction

    def compute_lateral_offset(self, p):
        return self.perp @ (p - self.origin)


def half_line_segment_to_point_dist(v1, direction, p, tol=1e-8):
    """Get distance between "half" line segment which starts at vertex `v1` and
    extends infinitely in `direction` and the point `p`.

    Also returns the closest point on the line segment to `p`.
    """
    r = p - v1
    proj = direction @ r  # project onto the line

    # closest point is beyond the end of the segment
    if proj <= 0:
        return np.linalg.norm(r), v1
    c = v1 + proj * direction
    return np.linalg.norm(p - c), c


def line_segment_to_point_dist(v1, v2, p, tol=1e-8):
    """Get distance between line segment defined by vertices v1 and v2 and a point p.

    Also returns the closest point on the line segment to p.
    """
    v = v2 - v1
    r = p - v1
    L = np.linalg.norm(v)

    # degenerate case when the line segment has zero length (i.e., is a point)
    if L < tol:
        return np.linalg.norm(r), v1

    n = v / L  # unit vector along the line
    proj = n @ r  # project onto the line

    # closest point is beyond the end of the segment
    if proj <= 0:
        return np.linalg.norm(r), v1
    if proj >= L:
        return np.linalg.norm(p - v2), v2

    c = v1 + proj * n
    return np.linalg.norm(p - c), c


class SegmentPath:
    def __init__(self, vertices, final_direction=None):
        """If final_direction is None, the path is assumed to be closed.
        Otherwise it extends infinitely toward `final_direction` after the
        last vertex.
        """
        self.vertices = np.array(vertices)
        self.closed = final_direction is None

        n = self.vertices.shape[0]
        assert n > 0
        if n == 1:
            assert (
                not self.closed
            ), "final_direction must be specified when only one vertex given"

        self.directions = np.zeros((n, 2))
        for i in range(n - 1):
            self.directions[i, :] = util.unit(self.vertices[i + 1, :] - self.vertices[i, :])
        if self.closed:
            self.directions[-1, :] = util.unit(
                self.vertices[0, :] - self.vertices[-1, :]
            )
        else:
            self.directions[-1, :] = util.unit(final_direction)

        R = util.rot2d(np.pi / 2)
        self.perps = (R @ self.directions.T).T

    @classmethod
    def square(cls, half_length, center=None):
        if center is None:
            center = np.zeros(2)
        h = half_length
        vertices = np.array([[h, h], [-h, h], [-h, -h], [h, -h]]) + center
        return cls(vertices)

    @classmethod
    def line(cls, direction, origin=None):
        if origin is None:
            origin = np.zeros(2)
        return cls(np.atleast_2d(origin), direction)

    def _compute_closest_segment_and_point(self, p):
        """Compute the closest segment index as well as the closest point on
        that segment and distance to it."""
        n = self.vertices.shape[0]
        closest = (0, np.infty, None)
        for i in range(n - 1):
            v1 = self.vertices[i, :]
            v2 = self.vertices[i + 1, :]
            d, c = line_segment_to_point_dist(v1, v2, p)
            if d <= closest[1]:
                closest = (i, d, c)
        if self.closed:
            v1 = self.vertices[-1, :]
            v2 = self.vertices[0, :]
            d, c = line_segment_to_point_dist(v1, v2, p)
        else:
            v1 = self.vertices[-1, :]
            d, c = half_line_segment_to_point_dist(v1, self.directions[-1, :], p)
        if d <= closest[1]:
            closest = (n - 1, d, c)
        return closest

    def compute_closest_point(self, p):
        return self._compute_closest_segment_and_point(p)[2]

    def compute_travel_direction(self, p, d=0.5):
        # idx, _, _ = self._compute_closest_segment_and_point(p)
        # return self.directions[idx, :]
        idx, _, c1 = self._compute_closest_segment_and_point(p)
        n = self.vertices.shape[0]
        if idx == n - 1 or d < np.linalg.norm(self.vertices[idx + 1, :] - c1):
            return self.directions[idx, :]
        return self.directions[idx + 1, :]

    def compute_shortest_distance(self, p):
        return self._compute_closest_segment_and_point(p)[1]

    def compute_lateral_offset(self, p, d=0.5):
        # idx, _, _ = self._compute_closest_segment_and_point(p)
        # return self.perps[idx, :] @ (p - self.vertices[idx, :])
        idx, _, c1 = self._compute_closest_segment_and_point(p)
        n = self.vertices.shape[0]
        if idx == n - 1 or d < np.linalg.norm(self.vertices[idx + 1, :] - c1):
            return self.perps[idx, :] @ (p - self.vertices[idx, :])
        return self.perps[idx + 1, :] @ (p - self.vertices[idx + 1, :])

    def compute_point_ahead(self, p, d):
        """Compute point distance d ahead of closest point on path to p."""
        idx, _, c1 = self._compute_closest_segment_and_point(p)
        n = self.vertices.shape[0]
        if idx == n - 1:
            return c1 + d * self.directions[idx, :]
        Δ = np.linalg.norm(self.vertices[idx + 1, :] - c1)
        if d < Δ:
            return c1 + d * self.directions[idx, :]
        # NOTE: right now we are assuming the lookahead is not so high as to
        # cover more than one vertex
        return self.vertices[idx + 1, :] + (Δ - d) * self.directions[idx + 1, :]

