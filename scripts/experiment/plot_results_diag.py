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


# FIGURE_PATH = "experimental_results_diag.pdf"
FIGURE_PATH = "/home/adam/phd/papers/pushing/heins-icra23/tex/figures/experimental_results.pdf"

FORCE_THRESHOLD = 5
MAX_DISTANCE = 6

DIRECTION = rot2d(np.deg2rad(125)) @ np.array([1, 0])
DIRECTION_PERP = rot2d(np.pi / 2) @ DIRECTION

BARREL_OFFSET = np.array([-0.00273432, -0.01013547, -0.00000609])

BAG_DIR = Path(mm.BAG_DIR)
ROOT_DIR = BAG_DIR / "../icra23/diag"

CLOSED_LOOP_BOX_BAGS = [
    ROOT_DIR / "closed-loop/box" / name
    for name in [
        "box_diag1_2023-05-03-16-25-14.bag",
        "box_diag2_2023-05-03-16-28-09.bag",
        "box_diag3_2023-05-03-16-30-45.bag",
        "box_diag4_2023-05-03-16-33-19.bag",
        "box_diag5_2023-05-03-16-36-18.bag",
        "box_diag6_2023-05-03-17-35-12.bag",
        "box_diag7_2023-05-03-17-37-48.bag",
        "box_diag8_2023-05-03-17-40-26.bag",
        "box_diag9_2023-05-03-17-43-04.bag",
        "box_diag10_2023-05-03-17-45-36.bag",
    ]
]
OPEN_LOOP_BOX_BAGS = [
    ROOT_DIR / "open-loop/box" / name
    for name in [
        "box_diag1_2023-05-03-17-52-01.bag",
        "box_diag2_2023-05-03-17-53-06.bag",
        "box_diag3_2023-05-03-17-53-58.bag",
        "box_diag4_2023-05-03-17-54-48.bag",
        "box_diag5_2023-05-03-17-55-51.bag",
    ]
]

# dbox is the "double box"; i.e., containing *two* 5-lb weights
CLOSED_LOOP_DBOX_BAGS = [
    ROOT_DIR / "closed-loop/dbox" / name
    for name in [
        "dbox_diag1_2023-05-03-16-42-19.bag",
        "dbox_diag2_2023-05-03-16-44-56.bag",
        "dbox_diag3_2023-05-03-16-47-44.bag",
        "dbox_diag4_2023-05-03-16-50-23.bag",
        "dbox_diag5_2023-05-03-16-53-26.bag",
        "dbox_diag6_2023-05-03-17-08-30.bag",
        "dbox_diag7_2023-05-03-17-11-05.bag",
        "dbox_diag8_2023-05-03-17-14-12.bag",
        "dbox_diag9_2023-05-03-17-16-46.bag",
        "dbox_diag10_2023-05-03-17-19-20.bag",
    ]
]
OPEN_LOOP_DBOX_BAGS = [
    ROOT_DIR / "open-loop/dbox" / name
    for name in [
        # "dbox_diag1_2023-05-03-17-25-28.bag",  # too similar to runs 2 and 3
        "dbox_diag2_2023-05-03-17-26-34.bag",
        "dbox_diag3_2023-05-03-17-27-29.bag",
        "dbox_diag4_2023-05-03-17-28-23.bag",
        "dbox_diag5_2023-05-03-17-29-13.bag",
        "dbox_diag6_2023-05-03-17-31-49.bag",
    ]
]

CLOSED_LOOP_BARREL_BAGS = [
    ROOT_DIR / "closed-loop/barrel" / name
    for name in [
        "barrel_diag1_2023-05-03-15-52-22.bag",
        "barrel_diag2_2023-05-03-16-03-38.bag",
        "barrel_diag3_2023-05-03-16-06-30.bag",
        "barrel_diag4_2023-05-03-16-09-05.bag",
        "barrel_diag5_2023-05-03-16-11-38.bag",
        "barrel_diag6_2023-05-03-18-03-13.bag",
        "barrel_diag7_2023-05-03-18-05-47.bag",
        "barrel_diag8_2023-05-03-18-08-19.bag",
        "barrel_diag9_2023-05-03-18-10-56.bag",
        "barrel_diag10_2023-05-03-18-13-28.bag",
    ]
]
OPEN_LOOP_BARREL_BAGS = [
    ROOT_DIR / "open-loop/barrel" / name
    for name in [
        "barrel_diag1_2023-05-03-18-20-57.bag",
        "barrel_diag2_2023-05-03-18-22-12.bag",
        "barrel_diag3_2023-05-03-18-23-22.bag",
        "barrel_diag4_2023-05-03-18-24-31.bag",
        "barrel_diag5_2023-05-03-18-25-20.bag",
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
    # account for offset of Barrel's Vicon object origin from actual centroid
    if vicon_object_topic == "ThingBarrel":
        r_ow_ws = poses[:, :3]
        Q_wos = poses[:, 3:]
        r_cw_ws = np.zeros_like(r_ow_ws)
        for i in range(r_cw_ws.shape[0]):
            C_wo = q2r(Q_wos[i, :], order="xyzs")
            r_cw_ws[i, :] = r_ow_ws[i, :] + C_wo @ BARREL_OFFSET
        positions = r_cw_ws[:, :2]
    else:
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


def hide_x_ticks(ax):
    ax.set_xticklabels([])
    ax.tick_params(axis="x", colors=(0, 0, 0, 0))


def main():
    home = mm.load_home_position(name="pushing2")
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

    dbox_cl_data = []
    for path in CLOSED_LOOP_DBOX_BAGS:
        dbox_cl_data.append(parse_bag_data("ThingBox", path, c0))

    dbox_ol_data = []
    for path in OPEN_LOOP_DBOX_BAGS:
        dbox_ol_data.append(parse_bag_data("ThingBox", path, c0))

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
            "figure.labelsize": 6,
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

    XLIM = [-0.2, 6.2]
    YLIM = [-0.6, 0.6]
    YTICKS = [-0.5, 0, 0.5]

    fig = plt.figure(figsize=(3.25, 2))
    ax = plt.subplot(3, 1, 1)
    ax.set_aspect("equal")
    plt.axhline(0, color="k", linestyle="--", linewidth=0.75, zorder=-1)
    for data in box_cl_data:
        plt.plot(data[0], data[1], color=palette[0], alpha=0.75, solid_capstyle="round")
    for data in box_ol_data:
        plt.plot(data[0], data[1], color="k", alpha=0.75, solid_capstyle="round")
    hide_x_ticks(ax)
    plt.grid(color=(0.75, 0.75, 0.75), alpha=0.5, linewidth=0.5)
    plt.xlim(XLIM)
    plt.ylim(YLIM)
    ax.set_yticks(YTICKS)
    plt.legend([], [], title="Box (5 lb)", labelspacing=0, loc="upper right") #loc=(0.5, 0.7))

    ax = plt.subplot(3, 1, 2)
    ax.set_aspect("equal")
    plt.axhline(0, color="k", linestyle="--", linewidth=0.75, zorder=-1)
    for data in dbox_cl_data:
        plt.plot(data[0], data[1], color=palette[2], alpha=0.75, solid_capstyle="round")
    for data in dbox_ol_data:
        plt.plot(data[0], data[1], color="k", alpha=0.75, solid_capstyle="round")
    hide_x_ticks(ax)
    plt.grid(color=(0.75, 0.75, 0.75), alpha=0.5, linewidth=0.5)
    plt.xlim(XLIM)
    plt.ylim(YLIM)
    ax.set_yticks(YTICKS)
    plt.legend([], [], title="Box (10 lb)", labelspacing=0, loc="upper right") #loc=(0.5, 0.7))

    ax = plt.subplot(3, 1, 3)
    ax.set_aspect("equal")
    plt.axhline(0, color="k", linestyle="--", linewidth=0.75, zorder=-1)
    for data in barrel_cl_data:
        plt.plot(data[0], data[1], color=palette[3], alpha=0.75, solid_capstyle="round")
    for data in barrel_ol_data:
        plt.plot(data[0], data[1], color="k", alpha=0.75, solid_capstyle="round")
    plt.grid(color=(0.75, 0.75, 0.75), alpha=0.5, linewidth=0.5)
    plt.xlim(XLIM)
    plt.ylim(YLIM)
    ax.set_yticks(YTICKS)
    plt.xlabel("$x$ [m]")
    plt.legend([], [], title="Barrel", labelspacing=0, loc="upper right") #loc=(0.5, 0.7))

    # fake hidden label to get correct spacing
    plt.ylabel("$y$ [m]", alpha=0)

    # fig.supxlabel("$x$ [m]")
    fig.supylabel("$y$ [m]")

    fig.tight_layout(pad=0.1)
    fig.savefig(FIGURE_PATH)
    print(f"Saved figure to {FIGURE_PATH}")


if __name__ == "__main__":
    main()
