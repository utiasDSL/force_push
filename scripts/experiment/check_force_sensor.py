#!/usr/bin/env python3
"""Print out filtered force values."""
import argparse
import datetime
import pickle
import time
import yaml

import rospy
import numpy as np
from spatialmath.base import rotz

import mobile_manipulation_central as mm
import force_push as fp

import IPython


# Datasheet claims the F/T sensor output rate is 100Hz, though rostopic says
# more like ~62Hz
RATE = 100  # Hz
TIMESTEP = 1.0 / RATE

# Origin is taken as the EE's starting position
STRAIGHT_DIRECTION = fp.rot2d(np.deg2rad(125)) @ np.array([1, 0])
STRAIGHT_REV_DIRECTION = fp.rot2d(np.deg2rad(125 - 180)) @ np.array([1, 0])

# pushing speed
PUSH_SPEED = 0.1

# control gains
Kθ = 0.3
KY = 0.3
Kω = 1
KF = 0.003
CON_INC = 0.1
DIV_INC = 0.3

# base velocity bounds
VEL_UB = np.array([0.5, 0.5, 0.25])
VEL_LB = -VEL_UB

VEL_WEIGHT = 1.0
ACC_WEIGHT = 0.0

# only control based on force when it is high enough (i.e. in contact with
# something)
FORCE_MIN_THRESHOLD = 5
FORCE_MAX_THRESHOLD = 50

# time constant for force filter
# FILTER_TIME_CONSTANT = 0.1
FILTER_TIME_CONSTANT = 0.05

# minimum obstacle distance
BASE_OBS_MIN_DIST = 0.65  # 0.55 radius circle + 0.1 obstacle dist
EE_OBS_MIN_DIST = 0.1


def main():
    np.set_printoptions(precision=6, suppress=True)

    rospy.init_node("check_force_sensor_node")

    home = mm.load_home_position(name="pushing_diag", path=fp.HOME_CONFIG_FILE)
    model = mm.MobileManipulatorKinematics(tool_link_name="gripper")
    ft_idx = model.get_link_index("ft_sensor")
    q_arm = home[3:]

    # load calibrated offset between contact point and base frame origin
    with open(fp.CONTACT_POINT_CALIBRATION_FILE) as f:
        r_bc_b = np.array(yaml.safe_load(f)["r_bc_b"])

    # wait until robot feedback has been received
    robot = mm.RidgebackROSInterface()
    rate = rospy.Rate(RATE)
    while not rospy.is_shutdown() and not robot.ready():
        rate.sleep()

    # zero the F-T sensor
    print("Estimating F-T sensor bias...")
    bias_estimator = fp.WrenchBiasEstimator()
    bias = bias_estimator.estimate(RATE)
    print(f"Done. Bias = {bias}")

    wrench_estimator = fp.WrenchEstimator(bias=bias, τ=FILTER_TIME_CONSTANT)

    while not rospy.is_shutdown():
        q = np.concatenate((robot.q, q_arm))
        r_bw_w = q[:2]
        C_wb = fp.rot2d(q[2])
        r_cw_w = r_bw_w - C_wb @ r_bc_b

        model.forward(q)
        C_wf = model.link_pose(link_idx=ft_idx, rotation_matrix=True)[1]
        f_f = wrench_estimator.wrench_filtered[:3]
        f_w = C_wf @ f_f

        f_b = rotz(q[2]).T @ f_w

        f = -f_b[:2]
        print(f)

        rate.sleep()


if __name__ == "__main__":
    main()
