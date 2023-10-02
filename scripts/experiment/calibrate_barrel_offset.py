#!/usr/bin/env python3
"""Fit offset from barrel reference to centroid using Vicon measurements."""
import argparse
from pathlib import Path
import yaml

import numpy as np
import rosbag
import matplotlib.pyplot as plt
from spatialmath.base import q2r

import mobile_manipulation_central as mm
from mobile_manipulation_central import ros_utils


VICON_OBJECT_NAME = "ThingBarrel"
VICON_OBJECT_TOPIC = ros_utils.vicon_topic_name(VICON_OBJECT_NAME)

CALIBRATION_FILE = "barrel_offset_calibration.yaml"


def average_quaternion(Qs):
    """Compute the average of a 2D array of quaternions, one per row."""
    Qs = np.array(Qs)
    e, V = np.linalg.eig(Qs.T @ Qs)
    i = np.argmax(e)
    return V[:, i]


def parse_marker_positions(marker_msgs, N=25):
    points = []
    for msg in marker_msgs:
        for marker in msg.markers:
            if marker.occluded:
                continue
            if marker.segment_name == "ThingBarrel":
                x = marker.translation.x
                y = marker.translation.y
                points.append([x, y])
                if len(points) >= N:
                    return np.array(points) / 1000  # convert to meters
    raise ValueError(f"Less than {N} points found to fit the circle.")


def fit_barrel_circle(bag):
    vicon_msgs = [msg for _, msg, _ in bag.read_messages(VICON_OBJECT_TOPIC)]
    poses = ros_utils.parse_transform_stamped_msgs(vicon_msgs)[1]

    r_ow_w = np.mean(poses[:10, :3], axis=0)
    C_wo = q2r(average_quaternion(poses[:10, 3:]), order="xyzs")

    marker_msgs = [msg for _, msg, _ in bag.read_messages("/vicon/markers")]
    positions = parse_marker_positions(marker_msgs)

    A = np.hstack((positions, np.ones((positions.shape[0], 1))))
    b = np.sum(positions**2, axis=1)
    x = np.linalg.lstsq(A, b, rcond=None)[0]

    r_cw_w = np.append(0.5 * x[:2], r_ow_w[2])
    radius = 0.5 * np.sqrt(4 * x[2] + x[0] ** 2 + x[1] ** 2)
    r_co_o = C_wo.T @ (r_cw_w - r_ow_w)
    return r_co_o


def main():
    np.set_printoptions(suppress=True)

    parser = argparse.ArgumentParser()
    parser.add_argument("bagfile", help="Bag file to plot.")
    args = parser.parse_args()
    bag = rosbag.Bag(args.bagfile)

    r_co_o = fit_barrel_circle(bag)
    print(f"r_co_o = {r_co_o}")

    bag_path = Path(args.bagfile).resolve().as_posix()

    # save the results
    with open(CALIBRATION_FILE, "w") as f:
        yaml.dump({"bag_path": bag_path, "r_co_o": r_co_o.tolist()}, stream=f)
    print(f"Saved to {CALIBRATION_FILE}")


if __name__ == "__main__":
    main()
