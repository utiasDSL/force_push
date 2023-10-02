#!/usr/bin/env python3
"""Plot slider position from a ROS bag."""
import argparse
import glob
from pathlib import Path
import pickle

import numpy as np
import rosbag
import matplotlib.pyplot as plt
from spatialmath.base import q2r

import mobile_manipulation_central as mm
from mobile_manipulation_central import ros_utils
import force_push as fp

import IPython

# TODO
BARREL_OFFSET = np.array([-0.00273432, -0.01013547, -0.00000609])


def parse_bag_dir(directory):
    """Parse params pickle path and bag path from a data directory.

    Returns (param_path, bag_path), as strings."""
    dir_path = Path(directory)

    param_files = glob.glob(dir_path.as_posix() + "/*.pkl")
    if len(param_files) == 0:
        raise FileNotFoundError(
            "Error: could not find a pickle in the specified directory."
        )
    if len(param_files) > 1:
        raise FileNotFoundError(
            "Error: multiple pickles in the specified directory. Please specify the name using the `--config_name` option."
        )
    param_path = param_files[0]

    bag_files = glob.glob(dir_path.as_posix() + "/*.bag")
    if len(bag_files) == 0:
        raise FileNotFoundError(
            "Error: could not find a bag file in the specified directory."
        )
    if len(bag_files) > 1:
        raise FileNotFoundError(
            "Error: multiple bag files in the specified directory. Please specify the name using the `--bag_name` option."
        )
    bag_path = bag_files[0]
    return param_path, bag_path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "bagdir", help="Directory containing bag file and pickled parameters."
    )
    parser.add_argument(
        "--slider", help="Name of slider Vicon object.", default="ThingBox"
    )
    args = parser.parse_args()

    param_path, bag_path = parse_bag_dir(args.bagdir)

    bag = rosbag.Bag(bag_path)
    with open(param_path, "rb") as f:
        params = pickle.load(f)

    home = mm.load_home_position(name="pushing_diag", path=fp.HOME_CONFIG_FILE)
    model = mm.MobileManipulatorKinematics()
    ft_idx = model.get_link_index("ft_sensor")
    q_arm = home[3:]

    r_bc_b = params["r_bc_b"]

    # Ridgeback
    rb_msgs = [msg for _, msg, _ in bag.read_messages("/ridgeback/joint_states")]
    rb_times, qs_rb, _ = ros_utils.parse_ridgeback_joint_state_msgs(
        rb_msgs, normalize_time=False
    )
    t0 = rb_times[0]
    rb_times -= t0
    q = np.concatenate((qs_rb[0, :], q_arm))
    model.forward(q)
    r_bw_w = q[:2]
    C_wb = fp.rot2d(q[2])
    r_cw_w = r_bw_w - C_wb @ r_bc_b
    r_bw_ws = qs_rb[:, :2] - r_cw_w

    # print(f"r_bw_w act = {r_bw_w}")
    # print(f"r_bw_w des = {home[:2]}")
    # return

    # base velocity commands
    cmd_msgs = [msg for _, msg, _ in bag.read_messages("/ridgeback/cmd_vel")]
    cmd_times = [t.to_sec() for _, _, t in bag.read_messages("/ridgeback/cmd_vel")]
    cmd_vels = []
    for msg in cmd_msgs:
        cmd_vel = [msg.linear.x, msg.linear.y, msg.angular.z]
        cmd_vels.append(cmd_vel)
    cmd_vels = np.array(cmd_vels)
    cmd_times -= t0

    # contact point
    r_cw_ws = []
    for i in range(len(rb_msgs)):
        r_bw_w = qs_rb[i, :2]
        C_wb = fp.rot2d(qs_rb[i, 2])
        r_cw_ws.append(r_bw_w - C_wb @ r_bc_b)
    r_cw_ws = np.array(r_cw_ws) - r_cw_w

    # parse wrenches to find the first time when contact force is above
    # minimum force threshold, indicating contact has started
    wrench_msgs = [msg for _, msg, _ in bag.read_messages("/wrench/filtered")]
    wrench_times, wrenches = ros_utils.parse_wrench_stamped_msgs(
        wrench_msgs, normalize_time=False
    )
    wrench_idx = np.argmax(
        np.linalg.norm(wrenches[:, :2], axis=1) >= params["force_min"]
    )
    wrench_times -= t0

    # print(wrench_idx)
    # return

    # slider
    slider_topic = ros_utils.vicon_topic_name(args.slider)
    slider_msgs = [msg for _, msg, _ in bag.read_messages(slider_topic)]
    slider_times, slider_poses = ros_utils.parse_transform_stamped_msgs(
        slider_msgs, normalize_time=False
    )
    slider_times -= t0
    r_sw_ws = slider_poses[:, :2] - r_cw_w

    # path
    # need to normalize to r_cw_w origin like everything else here
    path = fp.SegmentPath(params["path"].segments, origin=-r_cw_w)
    d = path.segments[-1].direction
    v = path.segments[-1].v2
    points = np.vstack((r_sw_ws, r_cw_ws, r_bw_ws))
    dist = np.max((points[:, :2] - v) @ d)
    dist = max(0, dist)
    r_dw_ws = path.get_plotting_coords(dist=dist)

    # obstacles
    obstacles = params["obstacles"]
    if obstacles is None:
        obstacles = []

    plt.figure()
    plt.plot(slider_times, r_sw_ws[:, 0], label="x")
    plt.plot(slider_times, r_sw_ws[:, 1], label="y")
    plt.xlabel("Time [s]")
    plt.ylabel("Position [m]")
    plt.title("Slider position vs. time")
    plt.legend()
    plt.grid()

    plt.figure()
    ax = plt.gca()
    ax.set_aspect("equal")
    plt.plot(r_sw_ws[:, 0], r_sw_ws[:, 1], label="Slider")
    plt.plot(r_dw_ws[:, 0], r_dw_ws[:, 1], "--", color="k", label="Desired")
    plt.plot(r_cw_ws[:, 0], r_cw_ws[:, 1], label="EE")
    plt.plot(r_bw_ws[:, 0], r_bw_ws[:, 1], label="Base")
    for obstacle in obstacles:
        plt.plot(
            [obstacle.v1[0], obstacle.v2[0]],
            [obstacle.v1[1], obstacle.v2[1]],
            color="r",
            label="Obstacle",
        )
    plt.legend()
    plt.xlabel("x [m]")
    plt.ylabel("y [m]")
    plt.title("Slider position")
    plt.grid()

    plt.figure()
    plt.plot(rb_times, r_bw_ws[:, 0], label="x")
    plt.plot(rb_times, r_bw_ws[:, 1], label="y")
    plt.xlabel("Time [s]")
    plt.ylabel("Position [m]")
    plt.title("Base position vs. time")
    plt.legend()
    plt.grid()

    # plt.figure()
    # ax = plt.gca()
    # ax.set_aspect("equal")
    # plt.plot(r_bw_ws[:, 0], r_bw_ws[:, 1])
    # plt.xlabel("x [m]")
    # plt.ylabel("y [m]")
    # plt.title("Base position")
    # plt.grid()

    plt.figure()
    plt.plot(rb_times, r_cw_ws[:, 0], label="x")
    plt.plot(rb_times, r_cw_ws[:, 1], label="y")
    plt.xlabel("Time [s]")
    plt.ylabel("Position [m]")
    plt.title("EE position vs. time")
    plt.legend()
    plt.grid()

    plt.figure()
    plt.plot(cmd_times, cmd_vels[:, 0], label="vx")
    plt.plot(cmd_times, cmd_vels[:, 1], label="vy")
    plt.plot(cmd_times, cmd_vels[:, 2], label="ω")
    plt.xlabel("Time [s]")
    plt.ylabel("Command")
    plt.title("Base commands vs. time")
    plt.legend()
    plt.grid()

    plt.figure()
    plt.plot(wrench_times, wrenches[:, 0], label="fx")
    plt.plot(wrench_times, wrenches[:, 1], label="fy")
    plt.plot(wrench_times, wrenches[:, 2], label="fz")
    plt.plot(wrench_times, np.linalg.norm(wrenches[:, :2], axis=1), label="fxy")
    plt.xlabel("Time [s]")
    plt.ylabel("Contact force [N]")
    plt.title("Contact forces vs. time")
    plt.legend()
    plt.grid()

    # plt.figure()
    # ax = plt.gca()
    # ax.set_aspect("equal")
    # plt.plot(r_cw_ws[:, 0], r_cw_ws[:, 1])
    # plt.xlabel("x [m]")
    # plt.ylabel("y [m]")
    # plt.title("EE position")
    # plt.grid()

    plt.show()


if __name__ == "__main__":
    main()
