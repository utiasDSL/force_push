import math
import numpy as np


def wrap_to_pi(x):
    """Wrap a value to [-π, π]"""
    return math.remainder(x, 2 * np.pi)


def rot2d(θ):
    """2D rotation matrix."""
    return np.array([[np.cos(θ), -np.sin(θ)], [np.sin(θ), np.cos(θ)]])


def skew2d(x):
    """Form a skew-symmetric matrix out of scalar x."""
    return np.array([[0, -x], [x, 0]])


def skew3d(v):
    """Form a skew-symmetric matrix out of 3-dimensional vector v."""
    x, y, z = v
    return np.array([[0, -z, y], [z, 0, -x], [-y, x, 0]])


def signed_angle(a, b):
    """Angle to rotate a to b (in radians).

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
