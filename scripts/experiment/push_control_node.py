#!/usr/bin/env python3
"""Push an object with mobile base + force-torque sensor."""
import argparse
import datetime
import pickle
import time
import yaml

import rospy
import numpy as np

import mobile_manipulation_central as mm
import force_push as fp

import IPython


# Datasheet claims the F/T sensor output rate is 100Hz, though rostopic says
# more like ~62Hz
RATE = 100  # Hz
TIMESTEP = 1.0 / RATE

# Origin is taken as the EE's starting position
STRAIGHT_DIRECTION = fp.rot2d(np.deg2rad(125)) @ np.array([1, 0])
STRAIGHT_REV_DIRECTION = fp.rot2d(np.deg2rad(125 - 180)) @ np.array([1, 0])

# pushing speed
PUSH_SPEED = 0.1

# control gains
Kθ = 0.3
KY = 0.3
Kω = 1
KF = 0.003
CON_INC = 0.1
DIV_INC = 0.3

# base velocity bounds
VEL_UB = np.array([0.5, 0.5, 0.25])
VEL_LB = -VEL_UB

VEL_WEIGHT = 1.0
ACC_WEIGHT = 0.0

# only control based on force when it is high enough (i.e. in contact with
# something)
FORCE_MIN_THRESHOLD = 5
FORCE_MAX_THRESHOLD = 50

# time constant for force filter
# FILTER_TIME_CONSTANT = 0.1
FILTER_TIME_CONSTANT = 0.05

# minimum obstacle distance
BASE_OBS_MIN_DIST = 0.65  # 0.55 radius circle + 0.1 obstacle dist
EE_OBS_MIN_DIST = 0.1


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
        choices=["straight", "corner", "corridor", "straight_rev"],
        help="Which environment to use",
        required=True,
    )
    parser.add_argument("--save", help="Save data to this file.")
    parser.add_argument("--notes", help="Additional information written to notes.txt.")
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

    # desired path
    q = robot.q
    r_bw_w = q[:2]
    C_wb = fp.rot2d(q[2])
    r_cw_w = r_bw_w - C_wb @ r_bc_b

    if args.environment == "straight":
        path = fp.SegmentPath.line(STRAIGHT_DIRECTION, origin=r_cw_w)
    elif args.environment == "straight_rev":
        path = fp.SegmentPath.line(STRAIGHT_REV_DIRECTION, origin=r_cw_w)
    else:
        path = fp.SegmentPath(
            [
                fp.LineSegment([0.0, 0.0], [0.0, 2.0]),
                fp.CircularArcSegment(
                    center=[-2.0, 2.0], point=[0.0, 2.0], angle=np.pi / 2
                ),
                fp.LineSegment([-2.0, 4.0], [-4.0, 4.0], infinite=True),
            ],
            origin=r_cw_w,
        )
    if args.environment == "corridor":
        obstacles = fp.translate_segments(
            [fp.LineSegment([-3.5, 4.25], [1.0, 4.25])], r_cw_w
        )
    else:
        obstacles = None

    # push controller generates EE velocity commands to realize stable pushing
    push_controller = fp.PushController(
        speed=PUSH_SPEED,
        kθ=Kθ,
        ky=KY,
        path=path,
        con_inc=CON_INC,
        div_inc=DIV_INC,
        obstacles=obstacles,
        force_min=FORCE_MIN_THRESHOLD,
        force_max=np.inf,  # NOTE
        min_dist=EE_OBS_MIN_DIST,
    )

    # admittance control to comply with large forces
    force_controller = fp.AdmittanceController(kf=KF, force_max=FORCE_MAX_THRESHOLD)

    # generate joint commands to realize desired EE velocity
    robot_controller = fp.RobotController(
        r_cb_b=-r_bc_b,
        lb=VEL_LB,
        ub=VEL_UB,
        vel_weight=VEL_WEIGHT,
        acc_weight=ACC_WEIGHT,
        obstacles=obstacles,
        min_dist=BASE_OBS_MIN_DIST,
    )

    # Save the controller parameters
    params = {
        "environment": args.environment,
        "ctrl_freq": RATE,
        "push_speed": PUSH_SPEED,
        "open_loop": open_loop,
        "kθ": Kθ,
        "ky": KY,
        "kω": Kω,
        "kf": KF,
        "con_inc": CON_INC,
        "div_inc": DIV_INC,
        "vel_ub": VEL_UB,
        "vel_lb": VEL_LB,
        "force_min": FORCE_MIN_THRESHOLD,
        "force_max": FORCE_MAX_THRESHOLD,
        "filter_time_constant": FILTER_TIME_CONSTANT,
        "base_obs_min_dist": BASE_OBS_MIN_DIST,
        "ee_obs_min_dist": EE_OBS_MIN_DIST,
        "vel_weight": VEL_WEIGHT,
        "acc_weight": ACC_WEIGHT,
        "path": path,
        "obstacles": obstacles,
        "r_bc_b": r_bc_b,
    }

    # record data
    if args.save is not None:
        recorder = fp.DataRecorder(name=args.save, notes=args.notes, params=params)
        recorder.record()
        print(f"Recording data to {recorder.log_dir}")

    # zero the F-T sensor
    print("Estimating F-T sensor bias...")
    bias_estimator = fp.WrenchBiasEstimator()
    bias = bias_estimator.estimate(RATE)
    print(f"Done. Bias = {bias}")

    wrench_estimator = fp.WrenchEstimator(bias=bias, τ=FILTER_TIME_CONSTANT)

    if args.save is not None:
        # wait a bit to ensure bag is setup before we actually do anything
        # interesting
        time.sleep(3.0)

    cmd_vel = np.zeros(3)

    t = rospy.Time.now().to_sec()
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

        # direction of the path
        # TODO we should actually be tracking the minimum distance here, which
        # is annoying
        info = path.compute_closest_point_info(r_cw_w)
        pathdir, offset = info.direction, info.offset
        θd = np.arctan2(pathdir[1], pathdir[0])

        # in open-loop mode we just follow the path rather than controlling to
        # push the slider
        if open_loop:
            θp = θd - KY * offset
            v_ee_cmd = PUSH_SPEED * fp.rot2d(θp) @ [1, 0]
            # v_ee_cmd = PUSH_SPEED * pathdir
        else:
            v_ee_cmd = push_controller.update(r_cw_w, f)
            v_ee_cmd = force_controller.update(force=f, v_cmd=v_ee_cmd)

        # desired angular velocity is calculated to align the robot with the
        # current path direction
        ωd = Kω * fp.wrap_to_pi(θd - q[2])
        V_ee_cmd = np.append(v_ee_cmd, ωd)

        # generate base input commands
        cmd_vel = robot_controller.update(r_bw_w, C_wb, V_ee_cmd, u_last=cmd_vel)
        if cmd_vel is None:
            print("Failed to solve QP!")
            break

        robot.publish_cmd_vel(cmd_vel, bodyframe=False)

        rate.sleep()

        t_new = rospy.Time.now().to_sec()
        if t - t_new > TIMESTEP:
            print(f"Loop took {t - t_new} seconds")
        t = t_new

    robot.brake()
    if args.save is not None:
        recorder.close()


if __name__ == "__main__":
    main()
