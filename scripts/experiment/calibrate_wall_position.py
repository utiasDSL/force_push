#!/usr/bin/env python3
"""Print positions of markers indicating location of wall obstacle."""
import yaml

import rospy
import numpy as np

from vicon_bridge.msg import Markers

import mobile_manipulation_central as mm
import force_push as fp

import IPython


RATE = 100  # Hz


def is_inside_polygon(vertices, point):
    n = vertices.shape[0]
    R = fp.rot2d(np.pi / 2)
    for i in range(n - 1):
        v = vertices[i]
        normal = R @ fp.unit(vertices[i + 1] - v)
        if normal @ (point - v) < 0:
            return False
    v = vertices[-1]
    normal = R @ fp.unit(vertices[0] - v)
    if normal @ (point - v) < 0:
        return False
    return True


class WallPositionCalibrator:
    def __init__(self, vertices, r_cw_w):
        """Set up the calibrator to look for a marker with radius distance of point c0."""
        self.vertices = vertices
        self.r_cw_w = r_cw_w

        self.marker_sub = rospy.Subscriber("/vicon/markers", Markers, self._marker_cb)

    def _marker_cb(self, msg):
        positions = []
        for marker in msg.markers:
            p = marker.translation
            p = np.array([p.x, p.y]) / 1000  # convert to meters
            if is_inside_polygon(self.vertices, p):
                positions.append(p - self.r_cw_w)

        # sort by x-position
        if len(positions) > 0:
            positions = np.array(positions)
            idx = np.argsort(positions[:, 0])
            print(positions[idx, :])
        else:
            print("no markers found")


def main():
    np.set_printoptions(precision=6, suppress=True)

    rospy.init_node("calibrate_wall_position_node", disable_signals=True)
    rate = rospy.Rate(RATE)

    box_vertices = np.array([[2.2, 2.0], [2.2, 2.5], [-2.2, 2.5], [-2.2, 2.0]])

    # load calibrated offset between contact point and base frame origin
    with open(fp.CONTACT_POINT_CALIBRATION_FILE) as f:
        r_bc_b = np.array(yaml.safe_load(f)["r_bc_b"])

    robot = mm.RidgebackROSInterface()
    rate = rospy.Rate(RATE)
    while not rospy.is_shutdown() and not robot.ready():
        rate.sleep()

    q = robot.q
    r_bw_w = q[:2]
    C_wb = fp.rot2d(q[2])
    r_cw_w = r_bw_w - C_wb @ r_bc_b

    calibrator = WallPositionCalibrator(box_vertices, r_cw_w)
    while not rospy.is_shutdown():
        rate.sleep()


if __name__ == "__main__":
    main()
