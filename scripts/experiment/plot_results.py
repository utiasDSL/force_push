#!/usr/bin/env python3
"""Plot slider position from a ROS bag."""
from pathlib import Path

import numpy as np
import rosbag
import matplotlib as mpl
import matplotlib.pyplot as plt
from spatialmath.base import q2r
import seaborn

import mobile_manipulation_central as mm
from mobile_manipulation_central import ros_utils
from mmpush import *

import IPython


FIGURE_PATH = "experimental_results.pdf"

FORCE_THRESHOLD = 5
MAX_DISTANCE = 5

DIRECTION = np.array([0, 1])
DIRECTION_PERP = rot2d(np.pi / 2) @ DIRECTION

BARREL_OFFSET = np.array([-0.00273432, -0.01013547, -0.00000609])

BAG_DIR = Path(mm.BAG_DIR)
ROOT_DIR = BAG_DIR / "2023-05-01"

CLOSED_LOOP_BOX_BAGS = [
    ROOT_DIR / "closed-loop/box" / name
    for name in [
        "box1_2023-05-01-13-32-03.bag",
        "box2_2023-05-01-13-34-32.bag",
        "box3_2023-05-01-13-37-11.bag",
        "box4_2023-05-01-13-39-41.bag",
        "box5_2023-05-01-13-42-13.bag",
    ]
]
OPEN_LOOP_BOX_BAGS = [
    ROOT_DIR / "open-loop/box" / name
    for name in [
        "box1_2023-05-01-13-59-26.bag",
        "box2_2023-05-01-14-00-24.bag",
        "box3_2023-05-01-14-01-23.bag",
        "box4_2023-05-01-14-02-31.bag",
        "box5_2023-05-01-14-03-38.bag",
        "box6_2023-05-01-14-04-37.bag",
    ]
]

CLOSED_LOOP_BARREL_BAGS = [
    ROOT_DIR / "closed-loop/barrel" / name
    for name in [
        "barrel1_2023-05-01-13-14-34.bag",
        "barrel2_2023-05-01-13-16-59.bag",
        "barrel3_2023-05-01-13-19-31.bag",
        "barrel4_2023-05-01-13-22-06.bag",
        "barrel5_2023-05-01-13-24-57.bag",
    ]
]
OPEN_LOOP_BARREL_BAGS = [
    ROOT_DIR / "open-loop/barrel" / name
    for name in [
        "barrel1_2023-05-01-13-48-50.bag",
        "barrel2_2023-05-01-13-54-07.bag",
        "barrel3_2023-05-01-13-55-37.bag",
        "barrel4_2023-05-01-13-56-51.bag",
        "barrel5_2023-05-01-13-57-54.bag",
    ]
]


def parse_bag_data(vicon_object_name, path, c0):
    bag = rosbag.Bag(path)

    # parse wrenches to find the first time when contact force is above
    # FORCE_THRESHOLD, indicating contact has started
    wrench_msgs = [msg for _, msg, _ in bag.read_messages("/wrench/filtered")]
    wrench_times, wrenches = ros_utils.parse_wrench_stamped_msgs(
        wrench_msgs, normalize_time=False
    )
    wrench_idx = np.argmax(np.linalg.norm(wrenches[:, :2], axis=1) > FORCE_THRESHOLD)
    t0 = wrench_times[wrench_idx]

    # parse position of slider since contact begins
    vicon_object_topic = ros_utils.vicon_topic_name(vicon_object_name)
    vicon_msgs = [msg for _, msg, _ in bag.read_messages(vicon_object_topic)]
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

    # trim paths to 5 m
    if xs[-1] > 5:
        last_idx = np.argmax(xs > MAX_DISTANCE)
        xs = xs[:last_idx]
        ys = ys[:last_idx]

    return xs, ys
    # return positions[:, 0], positions[:, 1]


def hide_x_ticks(ax):
    ax.set_xticklabels([])
    ax.tick_params(axis="x", colors=(0, 0, 0, 0))


def main():
    home = mm.load_home_position(name="pushing")
    model = mm.MobileManipulatorKinematics()
    ft_idx = model.get_link_index("ft_sensor")


    # initial contact point position
    model.forward(home)
    r_fw_w = model.link_pose(link_idx=ft_idx)[0]
    c0 = r_fw_w[:2]

    box_cl_data = []
    for path in CLOSED_LOOP_BOX_BAGS:
        box_cl_data.append(parse_bag_data("ThingBox", path, c0))

    box_ol_data = []
    for path in OPEN_LOOP_BOX_BAGS:
        box_ol_data.append(parse_bag_data("ThingBox", path, c0))

    barrel_cl_data = []
    for path in CLOSED_LOOP_BARREL_BAGS:
        barrel_cl_data.append(parse_bag_data("ThingBarrel", path, c0))

    barrel_ol_data = []
    for path in OPEN_LOOP_BARREL_BAGS:
        barrel_ol_data.append(parse_bag_data("ThingBarrel", path, c0))

    mpl.use("pgf")
    plt.rcParams.update(
        {
            "pgf.texsystem": "pdflatex",
            "font.size": 6,
            "font.family": "serif",
            "font.sans-serif": "DejaVu Sans",
            "font.weight": "normal",
            "text.usetex": True,
            "legend.fontsize": 6,
            "axes.titlesize": 6,
            "axes.labelsize": 6,
            "xtick.labelsize": 6,
            "pgf.preamble": "\n".join(
                [
                    r"\usepackage[utf8]{inputenc}",
                    r"\usepackage[T1]{fontenc}",
                    r"\usepackage{siunitx}",
                    r"\usepackage{bm}",
                ]
            ),
        }
    )

    palette = seaborn.color_palette("deep")

    fig = plt.figure(figsize=(3.25, 1.5))
    ax = plt.subplot(2, 1, 1)
    ax.set_aspect("equal")
    for data in box_ol_data:
        plt.plot(data[0], data[1], color="k", alpha=0.75, solid_capstyle="round")
    for data in box_cl_data:
        plt.plot(data[0], data[1], color=palette[0], alpha=0.75, solid_capstyle="round")
    plt.ylabel("$y$ [m]")
    hide_x_ticks(ax)
    plt.grid()
    plt.legend([], [], title="Square slider", labelspacing=0, loc=(0.5, 0.65))

    ax = plt.subplot(2, 1, 2)
    ax.set_aspect("equal")
    for data in barrel_ol_data:
        plt.plot(data[0], data[1], color="k", alpha=0.75, solid_capstyle="round")
    for data in barrel_cl_data:
        plt.plot(data[0], data[1], color=palette[3], alpha=0.75, solid_capstyle="round")
    plt.xlabel("$x$ [m]")
    plt.ylabel("$y$ [m]")
    plt.grid()
    plt.legend([], [], title="Circle slider", labelspacing=0, loc=(0.5, 0.65))

    fig.tight_layout(pad=0.1)
    fig.savefig(FIGURE_PATH)
    print(f"Saved figure to {FIGURE_PATH}")


if __name__ == "__main__":
    main()
