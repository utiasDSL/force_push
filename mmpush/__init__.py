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


