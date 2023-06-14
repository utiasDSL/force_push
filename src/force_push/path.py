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


class StraightPath:
    """Straight path to track in given direction and passing through origin."""

    def __init__(self, direction, origin=None):
        if origin is None:
            origin = np.zeros(2)
        self.origin = origin
        self.direction = direction
        self.perp = util.rot2d(np.pi / 2) @ direction

    def compute_closest_point(self, p):
        dist = (p - self.origin) @ self.direction
        return dist * self.direction + self.origin

    def compute_travel_direction(self, p):
        return self.direction

    def compute_lateral_offset(self, p):
        return self.perp @ (p - self.origin)
