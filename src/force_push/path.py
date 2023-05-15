import numpy as np

from force_push import util


class CirclePath:
    """Circle path to track."""

    def __init__(self, radius):
        self.radius = radius

    def compute_closest_point(self, p):
        return self.radius * self.compute_normal(p)

    def compute_normal(self, p):
        return util.unit(p)

    def compute_lateral_offset(self, p):
        c = self.compute_closest_point(p)
        n = self.compute_normal(p)
        return -n @ (p - c)

    def compute_travel_direction(self, p):
        return util.rot2d(np.pi / 2) @ self.compute_normal(p)


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
