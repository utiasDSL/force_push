import math
import numpy as np


def wrap_to_pi(x):
    """Wrap a value to [-π, π]"""
    return math.remainder(x, 2 * np.pi)


def rot2d(θ):
    """2D rotation matrix."""
    return np.array([[np.cos(θ), -np.sin(θ)], [np.sin(θ), np.cos(θ)]])


def signed_angle(a, b):
    """Angle to rotate a to b.

    See <https://stackoverflow.com/a/2150111/5145874>"""
    θa = np.arctan2(a[1], a[0])
    θb = np.arctan2(b[1], b[0])
    return wrap_to_pi(θb - θa)


def unit(x):
    """Normalize a vector.

    If the norm of the vector is zero, just return the vector.
    """
    norm = np.linalg.norm(x)
    if norm > 0:
        return x / norm
    return x


def perp2d(x):
    """Return a vector perpendicular to 2D vector x."""
    return np.array([-x[1], x[0]])
