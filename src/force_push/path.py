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

    def compute_shortest_distance(self, p):
        return self._compute_closest_segment_and_point(p)[1]

    def compute_direction_and_offset(self, p, lookahead=0):
        """Compute travel direction and lateral offset for a point p and given
        lookahead distance."""
        assert lookahead >= 0
        if lookahead < 1e-8:
            idx, _, c = self._compute_closest_segment_and_point(p)
            direction = self.directions[idx, :]
            offset = self.perps[idx, :] @ (p - c)
        else:
            c1, c2 = self._compute_lookahead_points(p, lookahead)
            direction = util.unit(c2 - c1)
            perp = util.rot2d(np.pi / 2) @ direction
            offset = perp @ (p - c1)
        return direction, offset

    def _compute_lookahead_points(self, p, lookahead):
        """Compute the closest point on the path to `p` as well as the point
        `lookahead` distance ahead of the closest point."""
        idx, _, c = self._compute_closest_segment_and_point(p)

        # when the path is closed we can wrap around to vertices at the start
        # of the path, soo we need to iterate beyond the end of the vertex
        # array (and module to wrap back around)
        n = self.vertices.shape[0]
        if self.closed:
            N = 2 * n
        else:
            N = n

        # iterate through vertices until we find the point `lookahead` distance
        # beyond the current point
        v0 = c
        for j in range(idx + 1, N):
            i = j % n
            v = self.vertices[i, :]
            dist = np.linalg.norm(v - v0)
            if lookahead > dist:
                lookahead -= dist
                v0 = v
            else:
                return c, v0 + lookahead * self.directions[i - 1, :]

        # at this point we must have run out of vertices, which means we must
        # have an open path
        assert not self.closed
        return c, v0 + lookahead * self.directions[-1, :]

    def get_coords(self, dist=5):
        """Get coordinates of the path (for plotting).

        For an open path, the `dist` parameter defines how far the final
        coordinate should be from the final vertex along the final direction.

        Returns a shape (n, 2) array of (x, y) coordinates.
        """
        if self.closed:
            last_vertex = self.vertices[0, :]
        else:
            last_vertex = self.vertices[-1, :] + dist * self.directions[-1, :]
        return np.vstack((self.vertices, last_vertex))
