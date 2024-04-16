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
from spatialmath import UnitQuaternion as UQ
import matplotlib.pyplot as plt

import mobile_manipulation_central as mm
import force_push as fp

import IPython


# Datasheet claims the F/T sensor output rate is 100Hz, though rostopic says
# more like ~62Hz
RATE = 100  # Hz

DURATION = 30

# time constant for force filter
# FILTER_TIME_CONSTANT = 0.1
FILTER_TIME_CONSTANT = 0.05

# set to `True` to use the current calibrated orientation to rotate force into
# the base frame
# this **does not** affect the data saved for subsequent calibration
USE_CALIBRATED_FORCE_ORN = False


def main():
    np.set_printoptions(precision=6, suppress=True)

    parser = argparse.ArgumentParser()
    parser.add_argument("--save", help="Save data to this file.")
    args = parser.parse_args()

    timestamp = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

    rospy.init_node("check_force_sensor_node")

    home = mm.load_home_position(name="pushing_diag", path=fp.HOME_CONFIG_FILE)
    model = mm.MobileManipulatorKinematics(tool_link_name="gripper")
    ft_idx = model.get_link_index("ft_sensor")
    q_arm = home[3:]

    if USE_CALIBRATED_FORCE_ORN:
        with open(fp.FORCE_ORN_CALIBRATION_FILE) as f:
            data = yaml.safe_load(f)
            ΔC = UQ(s=data["w"], v=[data["x"], data["y"], data["z"]]).R
    else:
        ΔC = np.eye(3)

    # zero the F-T sensor
    print("Estimating F-T sensor bias...")
    bias_estimator = fp.WrenchBiasEstimator()
    bias = bias_estimator.estimate(RATE)
    print(f"Done. Bias = {bias}")

    input("Hang the weight and press enter to continue...")

    rate = rospy.Rate(RATE)
    wrench_estimator = fp.WrenchEstimator(bias=bias, τ=FILTER_TIME_CONSTANT)
    while not rospy.is_shutdown() and not wrench_estimator.ready():
        rate.sleep()
    if rospy.is_shutdown():
        return

    ts = []
    f_bs = []
    f_fs = []
    C_bfs = []

    t0 = rospy.Time.now().to_sec()
    t = t0

    while not rospy.is_shutdown() and t - t0 < DURATION:
        # q = np.concatenate((robot.q, q_arm))
        q = np.concatenate((np.zeros(3), q_arm))

        model.forward(q)
        C_wf = model.link_pose(link_idx=ft_idx, rotation_matrix=True)[1]
        f_f = wrench_estimator.wrench_filtered[:3]
        f_w = C_wf @ f_f

        C_bf = rotz(q[2]).T @ C_wf
        f_b = ΔC @ C_bf @ f_f

        f_fs.append(f_f)
        f_bs.append(f_b)
        ts.append(t - t0)
        C_bfs.append(C_bf)
        print(f_b)

        rate.sleep()
        t = rospy.Time.now().to_sec()

    ts = np.array(ts)
    f_fs = np.array(f_fs)
    f_bs = np.array(f_bs)
    C_bfs = np.array(C_bfs)

    if args.save is not None:
        filename = f"{args.save}_{timestamp}.npz"
        np.savez_compressed(filename, ts=ts, f_fs=f_fs, C_bfs=C_bfs)
        print(f"Force data saved to {filename}.")

    print(f"mean f_b = {np.mean(f_bs, axis=0)}")

    plt.plot(ts, f_bs[:, 0], label="fx")
    plt.plot(ts, f_bs[:, 1], label="fy")
    plt.plot(ts, f_bs[:, 2], label="fz")
    plt.legend()
    plt.grid()
    plt.show()


if __name__ == "__main__":
    main()
