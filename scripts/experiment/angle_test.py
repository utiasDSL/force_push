import numpy as np
import force_push as fp
import IPython

x = np.array([1, 0])
y = np.array([0, 1])
direction = fp.rot2d(np.deg2rad(125 - 180)) @ x
angle = fp.signed_angle(direction, x)

IPython.embed()
