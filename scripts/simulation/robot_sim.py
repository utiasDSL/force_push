#!/usr/bin/env python3
"""Simulation demonstrating how to inverse kinematics on the base to achieve
desired EE linear and angular velocity (in the plane)."""
import rospkg
import numpy as np
import time
import matplotlib.pyplot as plt
import pyb_utils
from qpsolvers import solve_qp

import mobile_manipulation_central as mm
import force_push as fp

import IPython


TIMESTEP = 0.01
TOOL_JOINT_NAME = "tool_gripper_joint"
DURATION = 100

LOOKAHEAD = 2
PUSH_SPEED = 0.1


class RobotController:
    def __init__(self, r_cb_b):
        self.r_cb_b = r_cb_b

    def update(self, C_wb, V_ee_d, normal=None):
        S = np.array([[0, -1], [1, 0]])
        J = np.hstack((np.eye(2), (S @ C_wb @ self.r_cb_b)[:, None]))

        P = np.diag([0.1, 0.1, 1])
        q = np.array([0, 0, -V_ee_d[2]])

        lb = np.array([-1, -1, -0.5])
        ub = np.array([1, 1, 0.5])

        if normal is None:
            G = None
            h = None
        else:
            G = np.append(normal, 0).reshape((1, 3))
            h = np.zeros(1)

        A = J
        b = V_ee_d[:2]

        u = solve_qp(P=P, q=q, A=A, b=b, G=G, h=h, lb=lb, ub=ub, solver="proxqp")
        return u


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

    r_ew_w = robot.link_pose()[0]  # initial position
    q, _ = robot.joint_states()
    C_wb = fp.rot2d(q[2])
    c = r_ew_w[:2]
    r_be_b = -C_wb.T @ (c - q[:2])

    robot_controller = RobotController(-r_be_b)

    # desired EE path
    path = fp.SegmentPath(
        [
            fp.LineSegment([0., 0], [0., 1]),
            fp.QuadBezierSegment([0., 1], [0., 3], [-2., 3]),
            fp.LineSegment([-2., 3], [-3., 3], infinite=True),
        ],
        origin=c,
    )
    obstacle = fp.LineSegment(c + [-3, 3], c + [3, 3])
    pyb_utils.debug_frame_world(0.2, list(obstacle.v1) + [0.1], line_width=3)
    pyb_utils.debug_frame_world(0.2, list(obstacle.v2) + [0.1], line_width=3)

    # xy = path.get_plotting_coords()
    # plt.plot(xy[:, 0], xy[:, 1])
    # plt.grid()
    # plt.show()

    # yaw gain
    kθ = 1

    ω_max = 0

    ωu = 0.1
    ωl = -ωu

    t = 0
    while t <= DURATION:
        q, _ = robot.joint_states()
        p_ee = robot.link_pose()[0]
        v_ee = robot.link_velocity()[0]
        θ = q[2]  # yaw

        # desired linear velocity is along the path direction, with angular
        # velocity computed to make the robot oriented in the direction of the
        # path
        # TODO there is a lot of drift since we're not controlling the offset
        pathdir, off = path.compute_direction_and_offset(p_ee[:2])
        # print(f"off = {off}")
        θd = np.arctan2(pathdir[1], pathdir[0])
        ωd = kθ * fp.wrap_to_pi(θd - θ)
        ωd = min(ωu, max(ωl, ωd))
        V_ee_d = np.append(PUSH_SPEED * pathdir, ωd)

        # if abs(ωd) > ω_max:
        #     ω_max = ωd

        # print(f"dir = {pathdir}")
        # print(f"θd  = {θd}")
        # print(f"θ   = {θ}")

        # move the base so that the desired EE velocity is achieved
        C_wb = fp.rot2d(q[2])
        δ = np.append(fp.skew2d(V_ee_d[2]) @ C_wb @ r_be_b, 0)
        # u_base = V_ee_d + δ
        # u = np.concatenate((u_base, np.zeros(6)))

        d, closest = fp.line_segment_to_point_dist(obstacle.v1, obstacle.v2, q[:2])
        normal = None
        if d <= 0.75:
            normal = fp.unit(closest - q[:2])
        u_base = robot_controller.update(C_wb, V_ee_d, normal)
        u = np.concatenate((u_base, np.zeros(6)))

        # note that in simulation the mobile base takes commands in the world
        # frame, but the real mobile base takes commands in the body frame
        # (this is just an easy 2D rotation away)
        robot.command_velocity(u)

        # step the sim forward in time
        t = sim.step(t)
        # time.sleep(TIMESTEP)

    print(f"ω_max = {ω_max}")


main()
