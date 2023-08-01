#!/usr/bin/env python3
"""Simulation demonstrating QP-based robot controller."""
import time
import yaml

import rospkg
import numpy as np
import matplotlib.pyplot as plt
import pyb_utils

import mobile_manipulation_central as mm
import force_push as fp

import IPython


TIMESTEP = 0.01
TOOL_JOINT_NAME = "tool_gripper_joint"  # corresponds to the gripper link
DURATION = 100

PUSH_SPEED = 0.1

# base velocity bounds
VEL_UB = np.array([0.5, 0.5, 0.25])
VEL_LB = -VEL_UB

# minimum obstacle distance
OBS_MIN_DIST = 0.75

# obstacle starts to influence at this distance
OBS_INFL_DIST = 1.5


def main():
    np.set_printoptions(precision=6, suppress=True)

    # find the URDF (this has to be compiled first using the script
    # mobile_manipulation_central/urdf/compile_xacro.sh)
    rospack = rospkg.RosPack()
    mm_path = rospack.get_path("mobile_manipulation_central")
    urdf_path = mm_path + "/urdf/compiled/thing_pyb.urdf"

    # load initial joint configuration
    home = mm.load_home_position(name="pushing_corner", path=fp.HOME_CONFIG_FILE)

    # create the simulation
    sim = mm.BulletSimulation(TIMESTEP)
    robot = mm.BulletSimulatedRobot(urdf_path, TOOL_JOINT_NAME)
    robot.reset_joint_configuration(home)

    # load calibrated offset between contact point and base frame origin
    with open(fp.CONTACT_POINT_CALIBRATION_FILE) as f:
        r_bc_b = np.array(yaml.safe_load(f)["r_bc_b"])

    # initial position
    r_ew_w = robot.link_pose()[0]
    r0 = r_ew_w[:2]

    # desired EE path
    path = fp.SegmentPath(
        [
            fp.LineSegment([0.0, 0], [0.0, 1]),
            fp.QuadBezierSegment([0.0, 1], [0.0, 3], [-2.0, 3]),
            fp.LineSegment([-2.0, 3], [-3.0, 3], infinite=True),
        ],
        origin=r0,
    )
    obstacles = fp.translate_segments([fp.LineSegment([-3, 3], [3, 3])], r0)

    robot_controller = fp.RobotController(
        -r_bc_b, lb=VEL_LB, ub=VEL_UB, obstacles=obstacles, min_dist=OBS_MIN_DIST
    )

    for obstacle in obstacles:
        pyb_utils.debug_frame_world(0.2, list(obstacle.v1) + [0.1], line_width=3)
        pyb_utils.debug_frame_world(0.2, list(obstacle.v2) + [0.1], line_width=3)

    # xy = path.get_plotting_coords()
    # plt.plot(xy[:, 0], xy[:, 1])
    # plt.grid()
    # plt.show()

    # yaw gain
    kω = 1

    t = 0
    while t <= DURATION:
        q, _ = robot.joint_states()
        p_ee = robot.link_pose()[0]
        v_ee = robot.link_velocity()[0]
        θ = q[2]  # yaw

        # desired linear velocity is along the path direction, with angular
        # velocity computed to make the robot oriented in the direction of the
        # path
        pathdir, _ = path.compute_direction_and_offset(p_ee[:2])
        θd = np.arctan2(pathdir[1], pathdir[0])
        ωd = kω * fp.wrap_to_pi(θd - θ)
        V_ee_d = np.append(PUSH_SPEED * pathdir, ωd)

        # move the base so that the desired EE velocity is achieved
        C_wb = fp.rot2d(q[2])
        u_base = robot_controller.update(q[:2], C_wb, V_ee_d)
        if u_base is None:
            print("Failed to solve QP!")
            break
        u = np.concatenate((u_base, np.zeros(6)))

        # note that in simulation the mobile base takes commands in the world
        # frame, but the real mobile base takes commands in the body frame
        # (this is just an easy 2D rotation away)
        robot.command_velocity(u)

        # step the sim forward in time
        t = sim.step(t)
        # time.sleep(TIMESTEP)


main()
