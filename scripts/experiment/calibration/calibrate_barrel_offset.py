#!/usr/bin/env python3
"""Fit offset from barrel reference to centroid using Vicon measurements."""
import argparse
from pathlib import Path
import yaml

import numpy as np
import rosbag
import matplotlib.pyplot as plt
from spatialmath.base import q2r
from scipy.spatial.transform import Rotation

import mobile_manipulation_central as mm
from mobile_manipulation_central import ros_utils


VICON_OBJECT_NAME = "ThingBarrel"
VICON_OBJECT_TOPIC = ros_utils.vicon_topic_name(VICON_OBJECT_NAME)


def average_quaternion(Qs, as_rotation_matrix=False):
    """Compute the average of a 2D array of quaternions, one per row."""
    R = Rotation.from_quat(Qs).mean()
    if as_rotation_matrix:
        return R.as_matrix()
    return R.as_quat()


def parse_marker_positions(marker_msgs, N=25):
    points = []
    for msg in marker_msgs:
        for marker in msg.markers:
            if marker.occluded:
                continue
            if marker.segment_name == VICON_OBJECT_NAME:
                x = marker.translation.x
                y = marker.translation.y
                points.append([x, y])
                if len(points) >= N:
                    return np.array(points) / 1000  # convert to meters
    raise ValueError(f"Less than {N} points found to fit the circle.")


def fit_barrel_circle(bag):
    """Fit the barrel center to Vicon data from a ROS bag."""
    vicon_msgs = [msg for _, msg, _ in bag.read_messages(VICON_OBJECT_TOPIC)]
    poses = ros_utils.parse_transform_stamped_msgs(vicon_msgs)[1]

    # pose of the barrel
    r_vw_w = np.mean(poses[:10, :3], axis=0)
    C_wv = average_quaternion(poses[:10, 3:], as_rotation_matrix=True)

    marker_msgs = [msg for _, msg, _ in bag.read_messages("/vicon/markers")]
    r_mw_ws = parse_marker_positions(marker_msgs)

    # construct and solve least squares problem
    A = np.hstack((r_mw_ws, np.ones((r_mw_ws.shape[0], 1))))
    b = np.sum(r_mw_ws**2, axis=1)
    x = np.linalg.lstsq(A, b, rcond=None)[0]

    # translate back to quantities we care about
    r_cw_w = np.append(0.5 * x[:2], r_vw_w[2])
    radius = 0.5 * np.sqrt(4 * x[2] + x[0] ** 2 + x[1] ** 2)

    # offset pointing from Vicon frame origin to the actual centroid
    r_cv_v = C_wv.T @ (r_cw_w - r_vw_w)
    return r_cv_v


def zero_pose_xml(model_name, value_name, value):
    return f'<param name="{model_name}/zero_pose/{value_name}" value="{value:.6f}" type="double" />'


def zero_pose_position(model_name, position):
    s = []
    for name, value in zip(["x", "y", "z"], position):
        s.append(zero_pose_xml(model_name, f"position/{name}", value))
    return s


def main():
    np.set_printoptions(suppress=True)

    parser = argparse.ArgumentParser()
    parser.add_argument("bagfile", help="Bag file containing barrel measurements.")
    args = parser.parse_args()
    bag = rosbag.Bag(args.bagfile)

    r_cv_v = fit_barrel_circle(bag)
    print(f"r_cv_v = {r_cv_v}")

    # assume the {c} and {v} frames have the same orientation
    print("\n".join(zero_pose_position("$(arg barrel_vicon_model)", -r_cv_v)))


if __name__ == "__main__":
    main()
