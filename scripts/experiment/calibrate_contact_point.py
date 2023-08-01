#!/usr/bin/env python3
"""Calibrate the 2D offset from the contact point to the base frame origin.

The results are saved to a yaml file for consumption by other scripts (such as
controllers).
"""
import yaml

import rospy
import numpy as np

from vicon_bridge.msg import Markers

import mobile_manipulation_central as mm
import force_push as fp

import IPython


RATE = 100  # Hz
SEARCH_RADIUS = 0.2

CALIBRATION_FILE = "contact_point_calibration.yaml"


class ContactPointCalibrator:
    def __init__(self, c0, radius=0.4):
        """Set up the calibrator to look for a marker with radius distance of point c0."""
        self.c0 = c0
        self.radius = radius

        # track the markers seen
        self.count = 0
        self.average = np.zeros(3)

        self.marker_sub = rospy.Subscriber("/vicon/markers", Markers, self._marker_cb)

    def _marker_cb(self, msg):
        positions = []
        for marker in msg.markers:
            p = marker.translation
            p = np.array([p.x, p.y, p.z]) / 1000  # convert to meters
            if np.linalg.norm(self.c0 - p) < self.radius:
                positions.append(p)

        if len(positions) == 0:
            print("no markers found")
        elif len(positions) > 1:
            print("multiple markers found")
        else:
            print(f"marker found at {positions[0]}")
            self.average = (self.count * self.average + positions[0]) / (1 + self.count)
            self.count += 1


def compute_base_to_contact_vector(C_wb, r_cw_w, r_bw_w):
    r_bc_b = -C_wb.T @ (r_cw_w - r_bw_w)
    return r_bc_b


def main():
    np.set_printoptions(precision=6, suppress=True)

    rospy.init_node("calibrate_contact_point_node", disable_signals=True)

    home = mm.load_home_position(name="pushing_diag", path=fp.HOME_CONFIG_FILE)
    model = mm.MobileManipulatorKinematics(tool_link_name="gripper")
    q_arm = home[3:]

    # wait until robot feedback has been received
    robot = mm.RidgebackROSInterface()
    rate = rospy.Rate(RATE)
    print("Waiting for feedback from robot...")
    while not rospy.is_shutdown() and not robot.ready():
        rate.sleep()
    print("...feedback received.")

    # custom signal handler to brake the robot
    signal_handler = mm.RobotSignalHandler(robot)

    # get the nominal contact point based on the model
    q = np.concatenate((robot.q, q_arm))
    C_wb = fp.rot2d(q[2])
    model.forward(q)
    r_ew_w = model.link_pose()[0]

    # look for a marker close to this point
    # add an offset because we know it is slightly in front
    calibrator = ContactPointCalibrator(
        r_ew_w + np.append(C_wb @ [0.2, 0], 0), radius=SEARCH_RADIUS
    )
    while not rospy.is_shutdown() and calibrator.count < 20:
        rate.sleep()
    calibrator.marker_sub.unregister()

    # compute offset from base
    r_cw_w = calibrator.average
    r_be_b = compute_base_to_contact_vector(C_wb, r_ew_w[:2], q[:2])
    r_bc_b = compute_base_to_contact_vector(C_wb, r_cw_w[:2], q[:2])

    print("r_bc_b")
    print(f"nominal = {r_be_b}")
    print(f"actual  = {r_bc_b}")

    # save the results
    with open(CALIBRATION_FILE, "w") as f:
        yaml.dump({"r_bc_b": r_bc_b.tolist()}, stream=f)
    print(f"Saved to {CALIBRATION_FILE}")


if __name__ == "__main__":
    main()
