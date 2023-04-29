import numpy as np

from mmpush import util


class QuadSlider:
    """Quadrilateral slider with half lengths hx and hy."""

    def __init__(self, hx, hy, cof=None):
        self.hx = hx
        self.hy = hy

        # center of friction (relative to the centroid of the shape)
        if cof is None:
            cof = np.zeros(2)
        self.cof = np.array(cof)

    def contact_point(self, s, check=True):
        """Contact point given parameter s.

        s is relative to the centroid of the shape
        """
        # for a quad, s is the y position along the face
        if check and np.abs(s) > self.hy:
            raise ValueError("Pusher lost contact with slider.")
        return np.array([-self.hx, s])  # - self.cof

    def contact_normal(self, s):
        """Inward-facing contact normal."""
        # constant for a quad
        return np.array([1, 0])

    def s_dot(self, α):
        """Time derivative of s."""
        return α


class CircleSlider:
    """Circular slider with radius r."""

    def __init__(self, r, cof=None):
        self.r = r

        if cof is None:
            cof = np.zeros(2)
        self.cof = np.array(cof)

    def _angle(self, s):
        # negative because s increases clockwise, due to orientation of perp2d
        return -s / self.r

    def contact_point(self, s, check=True):
        angle = self._angle(s)
        return util.rot2d(angle) @ [-self.r, 0]

    def contact_normal(self, s):
        angle = self._angle(s)
        return np.array([np.cos(angle), np.sin(angle)])

    def s_dot(self, α):
        """Return derivative of angle s representing displacement of contact
        point along slider's edge."""
        return α
