#!/usr/bin/env python3
"""Plot slider position from a ROS bag."""
import argparse
import glob
from pathlib import Path
import pickle
import yaml

import numpy as np
import rosbag
import matplotlib.pyplot as plt
from spatialmath.base import q2r, rotz
from spatialmath import UnitQuaternion

import mobile_manipulation_central as mm
from mobile_manipulation_central import ros_utils
import force_push as fp

import IPython


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "bagdir", help="Directory containing bag file and pickled parameters."
    )
    parser.add_argument(
        "--slider",
        help="Name of slider Vicon object.",
        choices=["ThingBox", "ThingBarrel"],
        required=True,
    )
    args = parser.parse_args()

    param_path, bag_path = fp.parse_bag_dir(args.bagdir)

    bag = rosbag.Bag(bag_path)
    with open(param_path, "rb") as f:
        params = pickle.load(f)

    if params["environment"] == "straight":
        home = mm.load_home_position(name="pushing_diag", path=fp.HOME_CONFIG_FILE)
    elif params["environment"] == "straight_rev":
        home = mm.load_home_position(name="pushing_diag_rev", path=fp.HOME_CONFIG_FILE)
    else:
        home = mm.load_home_position(name="pushing_corner", path=fp.HOME_CONFIG_FILE)

    model = mm.MobileManipulatorKinematics()
    ft_idx = model.get_link_index("ft_sensor")
    q_arm = home[3:]

    # compute rotate from force sensor to base
    q_nom = np.concatenate((np.zeros(3), q_arm))
    model.forward(q_nom)
    C_bf = model.link_pose(link_idx=ft_idx, rotation_matrix=True)[1]
    ΔC = params["ΔC"]

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
    r_cw_w0 = r_bw_w - C_wb @ r_bc_b
    r_bw_ws = qs_rb[:, :2] - r_cw_w0

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
    r_cw_ws = np.array(r_cw_ws) - r_cw_w0

    # parse wrenches to find the first time when contact force is above
    # minimum force threshold, indicating contact has started
    wrench_msgs = [msg for _, msg, _ in bag.read_messages("/wrench/filtered")]
    wrench_times, wrenches = ros_utils.parse_wrench_stamped_msgs(
        wrench_msgs, normalize_time=False
    )
    wrench_times -= t0

    # align the messages so we can put them align them with the world frame
    wrenches_aligned = np.array(
        ros_utils.interpolate_list(rb_times, wrench_times, wrenches)
    )
    f_ws = np.zeros((wrenches_aligned.shape[0], 3))
    for i in range(f_ws.shape[0]):
        C_wb = rotz(qs_rb[i, 2])
        # negative to switch to applied force
        f_ws[i, :] = -C_wb @ ΔC @ C_bf @ wrenches_aligned[i, :3]
    force_idx = np.argmax(np.linalg.norm(f_ws[:, :2], axis=1) >= params["force_min"])
    print(f"First contact wrench index = {force_idx}")

    # slider
    slider_topic = ros_utils.vicon_topic_name(args.slider)
    slider_msgs = [msg for _, msg, _ in bag.read_messages(slider_topic)]
    slider_times, slider_poses = ros_utils.parse_transform_stamped_msgs(
        slider_msgs, normalize_time=False
    )
    slider_times -= t0
    slider_positions = slider_poses[:, :3]
    r_sw_ws = slider_positions[:, :2] - r_cw_w0

    # check initial yaw angle
    # Q_sw0 = slider_poses[0, 3:]
    # yaw = UnitQuaternion(s=Q_sw0[3], v=Q_sw0[:3]).rpy()[2]
    # print(f"yaw = {np.rad2deg(yaw)}")

    # path
    # need to normalize to r_cw_w0 origin like everything else here
    path = fp.SegmentPath(params["path"].segments, origin=-r_cw_w0)
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
    for i in range(len(obstacles)):
        obstacles[i] = obstacles[i].offset(-r_cw_w0)

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
    plt.plot(rb_times, f_ws[:, 0], label="fx")
    plt.plot(rb_times, f_ws[:, 1], label="fy")
    plt.plot(rb_times, f_ws[:, 2], label="fz")
    plt.plot(rb_times, np.linalg.norm(f_ws[:, :2], axis=1), label="fxy")
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
