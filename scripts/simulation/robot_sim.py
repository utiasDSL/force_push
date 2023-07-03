#!/usr/bin/env python3
"""Simulation demonstrating how to inverse kinematics on the base to achieve
desired EE linear and angular velocity (in the plane)."""
import rospkg
import numpy as np
import time

import mobile_manipulation_central as mm
import force_push as fp


TIMESTEP = 0.001
TOOL_JOINT_NAME = "tool_gripper_joint"
DURATION = 10


def main():
    # find the URDF (this has to be compiled first using the script
    # mobile_manipulation_central/urdf/compile_xacro.sh)
    rospack = rospkg.RosPack()
    mm_path = rospack.get_path("mobile_manipulation_central")
    urdf_path = mm_path + "/urdf/compiled/thing_pyb.urdf"

    # load initial joint configuration
    home = mm.load_home_position()

    # create the simulation
    sim = mm.BulletSimulation(TIMESTEP)
    robot = mm.BulletSimulatedRobot(urdf_path, TOOL_JOINT_NAME)
    robot.reset_joint_configuration(home)

    r_ew_w = robot.link_pose()[0]  # initial position
    q, _ = robot.joint_states()
    r_be_b = -(r_ew_w[:2] - q[:2])

    # desired EE velocity
    # x, y, θ
    V_ee_d = np.array([0.1, 0, 0.1])

    t = 0
    while t <= DURATION:
        q, _ = robot.joint_states()

        v_ee = robot.link_velocity()[0]
        # print(v_ee[:2])

        # Move the base so that the desired EE velocity is achieved
        C_wb = fp.rot2d(q[2])
        δ = np.append(fp.skew2d(V_ee_d[2]) @ C_wb @ r_be_b, 0)
        u_base = V_ee_d + δ
        u = np.concatenate((u_base, np.zeros(6)))

        # note that in simulation the mobile base takes commands in the world
        # frame, but the real mobile base takes commands in the body frame
        # (this is just an easy 2D rotation away)
        robot.command_velocity(u)

        # step the sim forward in time
        t = sim.step(t)
        time.sleep(TIMESTEP)


main()
