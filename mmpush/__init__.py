import numpy as np

def rot2d(θ):
    """2D rotation matrix."""
    return np.array([[np.cos(θ), -np.sin(θ)], [np.sin(θ), np.cos(θ)]])


def signed_angle(a, b):
    """Angle to rotate a to b.

    See <https://stackoverflow.com/a/2150111/5145874>"""
    return np.arctan2(b[1], b[0]) - np.arctan2(a[1], a[0])


def unit(x):
    """Normalize a vector."""
    norm = np.linalg.norm(x)
    if norm > 0:
        return x / norm
    return x


def pursuit(p, lookahead):
    """Pure pursuit along the x-axis."""
    if np.abs(p[1]) >= lookahead:
        return np.array([0, -np.sign(p[1]) * lookahead])
    x = lookahead**2 - p[1] ** 2
    return np.array([x, -p[1]])


def perp2d(x):
    return np.array([-x[1], x[0]])
