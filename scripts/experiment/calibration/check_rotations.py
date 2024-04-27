#!/usr/bin/env python3
"""Check rotation of force-torque sensor."""
import numpy as np
from spatialmath.base import rotx, roty, rotz, r2x

import mobile_manipulation_central as mm
import force_push as fp

import IPython


def main():
    np.set_printoptions(precision=6, suppress=True)

    # home = mm.load_home_position(name="pushing_diag", path=fp.HOME_CONFIG_FILE)
    model = mm.MobileManipulatorKinematics(tool_link_name="gripper")
    ft_idx = model.get_link_index("ft_sensor")
    # q_arm = home[3:]

    # C_test = rotz(np.pi / 2) @ rotx(np.pi) @ roty(-np.pi / 2) @ rotz(np.deg2rad(-15))
    # q_test = np.concatenate((np.zeros(3), q_arm))
    q_test = np.concatenate(
        (
            np.zeros(3),
            [
                np.pi / 2,
                np.deg2rad(-10),
                np.pi / 2,
                np.deg2rad(-80),
                np.pi / 2,
                np.deg2rad(75),
            ],
        )
    )

    C_arm_ft = rotz(q_test[3]) @ rotx(np.pi) @ roty(np.pi / 2) @ rotz(np.deg2rad(-15))
    C_base_ft = rotz(-np.pi / 2) @ C_arm_ft
    C_world_ft = rotz(q_test[2]) @ C_base_ft

    model.forward(q_test)
    # C_wa = model.link_pose(link_idx=model.get_link_index("ur10_arm_base_link"), rotation_matrix=True)[1]
    C_wf = model.link_pose(link_idx=ft_idx, rotation_matrix=True)[1]
    # C_af = C_wa.T @ C_wf
    # C_we = model.link_pose(link_idx=model.get_link_index("ur10_arm_tool0"), rotation_matrix=True)[1]

    # check that C_wf and C_world_ft are close

    IPython.embed()


if __name__ == "__main__":
    main()
