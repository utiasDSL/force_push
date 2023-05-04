#!/usr/bin/env python3
"""Plot slider position from a ROS bag."""
import argparse

import numpy as np
import rosbag
import matplotlib.pyplot as plt
from spatialmath.base import q2r

import mobile_manipulation_central as mm
from mobile_manipulation_central import ros_utils
from mmpush import *

import IPython


VICON_OBJECT_NAME = "ThingBarrel"
VICON_OBJECT_TOPIC = ros_utils.vicon_topic_name(VICON_OBJECT_NAME)

FORCE_THRESHOLD = 5

DIRECTION = rot2d(np.deg2rad(125)) @ np.array([1, 0])
DIRECTION_PERP = rot2d(np.pi / 2) @ DIRECTION

BARREL_OFFSET = np.array([-0.00273432, -0.01013547, -0.00000609])


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("bagfile", help="Bag file to plot.")
    args = parser.parse_args()
    bag = rosbag.Bag(args.bagfile)

    home = mm.load_home_position(name="pushing")
    model = mm.MobileManipulatorKinematics()
    ft_idx = model.get_link_index("ft_sensor")
    q_arm = home[3:]

    rb_msgs = [msg for _, msg, _ in bag.read_messages("/ridgeback/joint_states")]
    rb_times, qs_rb, _ = ros_utils.parse_ridgeback_joint_state_msgs(rb_msgs)
    q0 = np.concatenate((qs_rb[0, :], q_arm))
    model.forward(q0)
    r_fw_w = model.link_pose(link_idx=ft_idx)[0]
    c0 = r_fw_w[:2]

    # parse wrenches to find the first time when contact force is above
    # FORCE_THRESHOLD, indicating contact has started
    wrench_msgs = [msg for _, msg, _ in bag.read_messages("/wrench/filtered")]
    wrench_times, wrenches = ros_utils.parse_wrench_stamped_msgs(
        wrench_msgs, normalize_time=False
    )
    wrench_idx = np.argmax(np.linalg.norm(wrenches[:, :2], axis=1) > FORCE_THRESHOLD)
    t0 = wrench_times[wrench_idx]

    # parse position of slider since contact begins
    vicon_msgs = [msg for _, msg, _ in bag.read_messages(VICON_OBJECT_TOPIC)]
    vicon_msgs = ros_utils.trim_msgs(vicon_msgs, t0=t0)
    vicon_times, poses = ros_utils.parse_transform_stamped_msgs(
        vicon_msgs, normalize_time=True
    )
    positions = poses[:, :2]

    # normalize x-position of slider to always start at zero, but for
    # y-position of contact point as the reference
    xs = (positions - c0) @ DIRECTION
    ys = (positions - c0) @ DIRECTION_PERP
    xs -= xs[0]

    plt.figure()
    plt.plot(vicon_times, xs, label="x")
    plt.plot(vicon_times, ys, label="y")
    plt.xlabel("Time [s]")
    plt.ylabel("Position [m]")
    plt.title("Slider position vs. time")
    plt.legend()
    plt.grid()

    plt.figure()
    ax = plt.gca()
    ax.set_aspect("equal")
    plt.plot(xs, ys)
    plt.xlabel("x [m]")
    plt.ylabel("y [m]")
    plt.title("Slider position")
    plt.grid()

    plt.show()


if __name__ == "__main__":
    main()
