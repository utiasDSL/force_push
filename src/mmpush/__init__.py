import numpy as np
from scipy import sparse
import osqp

from mmpush.util import *
from mmpush.path import *
from mmpush.slider import *
from mmpush.motion import *


def pursuit(p, lookahead):
    """Pure pursuit along the x-axis."""
    if np.abs(p[1]) >= lookahead:
        return np.array([0, -np.sign(p[1]) * lookahead])
    x = lookahead**2 - p[1] ** 2
    return np.array([x, -p[1]])


def circle_r_tau(radius):
    """r_tau for a circular support area with uniform friction."""
    return 2.0 * radius / 3


def _alpha_rect(w, h):
    # alpha_rect for half of the rectangle
    d = np.sqrt(h * h + w * w)
    return (w * h * d + w * w * w * (np.log(h + d) - np.log(w))) / 12.0


def rectangle_r_tau(width, height):
    """r_tau for a rectangular support area with uniform friction."""
    # see pushing notes
    area = width * height
    return (_alpha_rect(width, height) + _alpha_rect(height, width)) / area
