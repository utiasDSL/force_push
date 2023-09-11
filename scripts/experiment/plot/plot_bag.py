#!/usr/bin/env python3
"""Plot slider position from a ROS bag."""
import argparse
import yaml

import numpy as np
import rosbag
import matplotlib.pyplot as plt
from spatialmath.base import q2r

import mobile_manipulation_central as mm
from mobile_manipulation_central import ros_utils
import force_push as fp

import IPython


FORCE_THRESHOLD = 5

# TODO
BARREL_OFFSET = np.array([-0.00273432, -0.01013547, -0.00000609])


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("bagfile", help="Bag file to plot.")
    parser.add_argument(
        "--slider", help="Name of slider Vicon object.", default="ThingBox"
    )
    args = parser.parse_args()
    bag = rosbag.Bag(args.bagfile)

    home = mm.load_home_position(name="pushing_diag", path=fp.HOME_CONFIG_FILE)
    model = mm.MobileManipulatorKinematics()
    ft_idx = model.get_link_index("ft_sensor")
    q_arm = home[3:]

    # load calibrated offset between contact point and base frame origin
    with open(fp.CONTACT_POINT_CALIBRATION_FILE) as f:
        r_bc_b = np.array(yaml.safe_load(f)["r_bc_b"])

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

    # base velocity commands
    cmd_msgs = [msg for _, msg, _ in bag.read_messages("/ridgeback/cmd_vel")]
    cmd_times = [t.to_sec() for _, _, t in bag.read_messages("/ridgeback/cmd_vel")]
    cmd_vels = []
    for msg in cmd_msgs:
        cmd_vel = [msg.linear.x, msg.linear.y, msg.angular.z]
        cmd_vels.append(cmd_vel)
    cmd_vels = np.array(cmd_vels)
    cmd_times -= t0

    # desired path
    path = fp.SegmentPath(
        [
            fp.LineSegment([0.0, 0.0], [0.0, 2.0]),
            fp.QuadBezierSegment([0.0, 2.0], [0.0, 4.0], [-2.0, 4.0]),
            fp.LineSegment([-2.0, 4.0], [-4.0, 4.0], infinite=True),
        ]
    )
    r_dw_ws = path.get_plotting_coords()

    # obstacles = [fp.LineSegment([-3.0, 4.35], [3.0, 4.35])]
    obstacles = []

    # contact point
    r_cw_ws = []
    for i in range(len(rb_msgs)):
        r_bw_w = qs_rb[i, :2]
        C_wb = fp.rot2d(qs_rb[i, 2])
        r_cw_ws.append(r_bw_w - C_wb @ r_bc_b)
    r_cw_ws = np.array(r_cw_ws) - r_cw_w

    # parse wrenches to find the first time when contact force is above
    # FORCE_THRESHOLD, indicating contact has started
    wrench_msgs = [msg for _, msg, _ in bag.read_messages("/wrench/filtered")]
    wrench_times, wrenches = ros_utils.parse_wrench_stamped_msgs(
        wrench_msgs, normalize_time=False
    )
    wrench_idx = np.argmax(np.linalg.norm(wrenches[:, :2], axis=1) > FORCE_THRESHOLD)

    # slider
    slider_topic = ros_utils.vicon_topic_name(args.slider)
    slider_msgs = [msg for _, msg, _ in bag.read_messages(slider_topic)]
    slider_times, slider_poses = ros_utils.parse_transform_stamped_msgs(
        slider_msgs, normalize_time=False
    )
    slider_times -= t0
    r_sw_ws = slider_poses[:, :2] - r_cw_w

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
