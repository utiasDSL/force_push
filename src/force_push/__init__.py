from pathlib import Path

import numpy as np
import rospkg

from force_push.util import *
from force_push.path import *
from force_push.slider import *
from force_push.motion import *
from force_push.quasi_simulation import *
from force_push.pyb_simulation import *
from force_push.plotting import *
from force_push.control import *
from force_push.inertia import *


rospack = rospkg.RosPack()
HOME_CONFIG_FILE = Path(rospack.get_path("force_push")) / "config/home.yaml"


def pursuit(p, lookahead):
    """Pure pursuit along the x-axis."""
    if np.abs(p[1]) >= lookahead:
        return np.array([0, -np.sign(p[1]) * lookahead])
    x = lookahead**2 - p[1] ** 2
    return np.array([x, -p[1]])
