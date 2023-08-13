#!/usr/bin/env python3
"""Push an object with mobile base + force-torque sensor."""
import argparse
import yaml

import rospy
import numpy as np

import mobile_manipulation_central as mm
import force_push as fp

import IPython


# Datasheet claims the sensor output rate is 100Hz, though rostopic says more
# like ~62Hz
RATE = 100  # Hz

# Origin is taken as the EE's starting position
STRAIGHT_DIRECTION = fp.rot2d(np.deg2rad(125)) @ np.array([1, 0])

# pushing speed
PUSH_SPEED = 0.1

# control gains
Kθ = 0.3
KY = 0.3
Kω = 1
CON_INC = 0.1
DIV_INC = 0.3  # NOTE

# base velocity bounds
VEL_UB = np.array([0.5, 0.5, 0.25])
VEL_LB = -VEL_UB

# only control based on force when it is high enough (i.e. in contact with
# something)
FORCE_MIN_THRESHOLD = 5
FORCE_MAX_THRESHOLD = 70  # TODO

# time constant for force filter
# FILTER_TIME_CONSTANT = 0.1
FILTER_TIME_CONSTANT = 0.05

# minimum obstacle distance
OBS_MIN_DIST = 0.75


# TODO do similar environment setup to the simulation
def main():
    np.set_printoptions(precision=6, suppress=True)

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--open-loop",
        help="Use open-loop pushing rather than closed-loop control",
        action="store_true",
        default=False,
    )
    parser.add_argument(
        "--environment",
        choices=["straight", "corner", "corridor"],
        help="Which environment to use",
        required=True,
    )
    args = parser.parse_args()
    open_loop = args.open_loop

    rospy.init_node("push_control_node", disable_signals=True)

    home = mm.load_home_position(name="pushing_diag", path=fp.HOME_CONFIG_FILE)
    model = mm.MobileManipulatorKinematics(tool_link_name="gripper")
    ft_idx = model.get_link_index("ft_sensor")
    q_arm = home[3:]

    # load calibrated offset between contact point and base frame origin
    with open(fp.CONTACT_POINT_CALIBRATION_FILE) as f:
        r_bc_b = np.array(yaml.safe_load(f)["r_bc_b"])

    # wait until robot feedback has been received
    robot = mm.RidgebackROSInterface()
    rate = rospy.Rate(RATE)
    while not rospy.is_shutdown() and not robot.ready():
        rate.sleep()

    # custom signal handler to brake the robot
    signal_handler = mm.RobotSignalHandler(robot)

    # zero the F-T sensor
    print("Estimating F-T sensor bias...")
    bias_estimator = fp.WrenchBiasEstimator()
    bias = bias_estimator.estimate(RATE)
    print(f"Done. Bias = {bias}")

    wrench_estimator = fp.WrenchEstimator(bias=bias, τ=FILTER_TIME_CONSTANT)

    # desired path
    q = robot.q
    r_bw_w = q[:2]
    C_wb = fp.rot2d(q[2])
    r_cw_w = r_bw_w - C_wb @ r_bc_b

    if args.environment == "straight":
        path = fp.SegmentPath.line(STRAIGHT_DIRECTION, origin=r_cw_w)
    else:
        path = fp.SegmentPath(
            [
                fp.LineSegment([0.0, 0], [0.0, 1]),
                fp.QuadBezierSegment([0.0, 1], [0.0, 3], [-2.0, 3]),
                fp.LineSegment([-2.0, 3], [-3.0, 3], infinite=True),
            ],
            origin=r_cw_w,
        )
    if args.environment == "corridor":
        obstacles = fp.translate_segments(
            [fp.LineSegment([-3.0, 3.5], [3.0, 3.5])], r_cw_w
        )
    else:
        obstacles = None

    # controllers
    push_controller = fp.PushController(
        speed=PUSH_SPEED,
        kθ=Kθ,
        ky=KY,
        path=path,
        con_inc=CON_INC,
        div_inc=DIV_INC,
        force_min=FORCE_MIN_THRESHOLD,
        force_max=FORCE_MAX_THRESHOLD,
    )
    robot_controller = fp.RobotController(
        -r_bc_b,
        lb=VEL_LB,
        ub=VEL_UB,
        vel_weight=1,
        acc_weight=0,
        obstacles=obstacles,
        min_dist=OBS_MIN_DIST,
    )

    cmd_vel = np.zeros(3)

    while not rospy.is_shutdown():
        q = np.concatenate((robot.q, q_arm))
        r_bw_w = q[:2]
        C_wb = fp.rot2d(q[2])
        r_cw_w = r_bw_w - C_wb @ r_bc_b

        model.forward(q)
        C_wf = model.link_pose(link_idx=ft_idx, rotation_matrix=True)[1]
        f_f = wrench_estimator.wrench_filtered[:3]
        f_w = C_wf @ f_f

        # force direction is negative to switch from sensed force to applied force
        f = -f_w[:2]
        print(np.linalg.norm(f))

        # direction of the path
        pathdir, _ = path.compute_direction_and_offset(r_cw_w)

        # in open-loop mode we just follow the path rather than controlling to
        # push the slider
        if open_loop:
            v_ee_cmd = PUSH_SPEED * pathdir
        else:
            v_ee_cmd = push_controller.update(r_cw_w, f)

        # desired angular velocity is calculated to align the robot with the
        # current path direction
        θd = np.arctan2(pathdir[1], pathdir[0])
        ωd = Kω * fp.wrap_to_pi(θd - q[2])
        V_ee_cmd = np.append(v_ee_cmd, ωd)

        # generate base input commands
        cmd_vel = robot_controller.update(r_bw_w, C_wb, V_ee_cmd, u_last=cmd_vel)
        if cmd_vel is None:
            print("Failed to solve QP!")
            break
        # print(f"cmd_vel = {cmd_vel}")
        # print(f"cmd_vel_xy_dir = {fp.unit(cmd_vel[:2])}")

        robot.publish_cmd_vel(cmd_vel, bodyframe=False)

        rate.sleep()

    robot.brake()


if __name__ == "__main__":
    main()
