import pytest
import numpy as np

import force_push as fp


def test_admittance_control():
    controller = fp.AdmittanceController(kf=1, force_max=10, vel_max=1)

    # force within force_max, velocity not changed
    v1 = np.array([1, 0])
    force = np.array([10, 0])
    v2 = controller.update(force=force, v_cmd=v1)
    assert np.allclose(v1, v2)

    # force causes velocity to go back in the opposite direction
    v1 = np.array([1, 0])
    force = np.array([20, 0])
    v2 = controller.update(force=force, v_cmd=v1)
    assert np.allclose(v1, -v2)
