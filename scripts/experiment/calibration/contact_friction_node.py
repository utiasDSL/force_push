#!/usr/bin/env python3
"""Push an object with mobile base + force-torque sensor."""
import argparse
import datetime
import pickle
import time
import yaml

import rospy
import numpy as np
from spatialmath import UnitQuaternion as UQ
from spatialmath.base import rotz
from std_msgs.msg import Time

import mobile_manipulation_central as mm
import force_push as fp

import IPython


# 1. first need to converge to path origin
# 2. zero out the force sensor
# 3. approach object
# 4. move sideways

# Datasheet claims the F/T sensor output rate is 100Hz, though rostopic says
# more like ~63Hz
RATE = 100  # Hz
TIMESTEP = 1.0 / RATE

# pushing speed
PUSH_SPEED = 0.01

# duration of initial acceleration phase
ACCELERATION_DURATION = 1.0

# acceleration value during acceleration phase (afterward it is zero)
ACCELERATION_MAGNITUDE = PUSH_SPEED / ACCELERATION_DURATION

ERR_THRESHOLD = 0.005

PATHDIR = np.array([0, 1])
ORTHDIR = np.array([-1, 0])

PATH_ANGLE = np.arctan2(PATHDIR[1], PATHDIR[0])
ORTH_ANGLE = np.arctan2(ORTHDIR[1], ORTHDIR[0])

# control gains
KY = 0.5
Kp = 1
Kω = 1

# only control based on force when it is high enough (i.e. in contact with
# something)
FORCE_MIN_THRESHOLD = 20

# time constant for force filter
FILTER_TIME_CONSTANT = 0.05


def main():
    np.set_printoptions(precision=6, suppress=True)

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--object",
        help="Name of the object being calibrated",
    )
    parser.add_argument("--save", help="Save data to this file.")
    parser.add_argument("--notes", help="Additional information written to notes.txt.")
    args = parser.parse_args()

    rospy.init_node("contact_friction_node", disable_signals=True)

    # publisher for the time when we start moving sideways
    time_pub = rospy.Publisher("/time_msg", Time, queue_size=1)

    home = mm.load_home_position(name="pushing_diag", path=fp.HOME_CONFIG_FILE)
    model = mm.MobileManipulatorKinematics(tool_link_name="gripper")
    ft_idx = model.get_link_index("ft_sensor")
    q_arm = home[3:]

    # load calibrated offset between contact point and base frame origin
    with open(fp.CONTACT_POINT_CALIBRATION_FILE) as f:
        r_bc_b = np.array(yaml.safe_load(f)["r_bc_b"])

    with open(fp.FORCE_ORN_CALIBRATION_FILE) as f:
        data = yaml.safe_load(f)
        ΔC = UQ(s=data["w"], v=[data["x"], data["y"], data["z"]]).R

    # wait until robot feedback has been received
    robot = mm.RidgebackROSInterface()
    rate = rospy.Rate(RATE)
    while not rospy.is_shutdown() and not robot.ready():
        rate.sleep()

    vicon_object = mm.ViconObjectInterface(args.object)
    while not rospy.is_shutdown() and not vicon_object.ready():
        rate.sleep()

    # custom signal handler to brake the robot
    signal_handler = mm.SimpleSignalHandler()

    # contact point position
    q = robot.q
    r_bw_w = q[:2]
    C_wb = fp.rot2d(q[2])
    r_cw_w = r_bw_w - C_wb @ r_bc_b

    # slider position
    r_sw_w = vicon_object.position[:2]
    origin = r_sw_w - (PATHDIR @ (r_sw_w - r_cw_w)) * PATHDIR

    # first we line up the robot with the slider in the normal direction
    print("Aligning robot with normal direction...")
    while not rospy.is_shutdown() and not signal_handler.received:
        q = robot.q
        r_bw_w = q[:2]
        C_wb = fp.rot2d(q[2])
        r_cw_w = r_bw_w - C_wb @ r_bc_b

        # steer toward path origin aligned with the path angle
        pos_error = origin - r_cw_w
        orn_error = fp.wrap_to_pi(PATH_ANGLE - q[2])
        if np.linalg.norm(pos_error) + np.abs(orn_error) <= ERR_THRESHOLD:
            break

        v_ee_cmd = Kp * pos_error
        if np.linalg.norm(v_ee_cmd) > PUSH_SPEED:
            v_ee_cmd = PUSH_SPEED * fp.unit(v_ee_cmd)
        ω_cmd = Kω * orn_error
        cmd_vel = np.append(v_ee_cmd, ω_cmd)

        # cmd_vel = robot_controller.update(r_bw_w, C_wb, V_ee_cmd)
        if cmd_vel is None:
            print("Failed to solve QP!")
            break

        robot.publish_cmd_vel(cmd_vel, bodyframe=False)
        rate.sleep()

    robot.brake()
    print("Aligned.")

    path = fp.SegmentPath.line(PATHDIR, origin=origin)

    if args.save is not None:
        recorder = fp.DataRecorder(name=args.save, notes=args.notes)
        recorder.record()
        print(f"Recording data to {recorder.log_dir}")

    # zero the F-T sensor
    print("Estimating F-T sensor bias...")
    bias_estimator = fp.WrenchBiasEstimator()
    bias = bias_estimator.estimate(RATE)
    print(f"Done. Bias = {bias}")

    wrench_estimator = fp.WrenchEstimator(bias=bias, τ=FILTER_TIME_CONSTANT)
    while not rospy.is_shutdown() and not signal_handler.received:
        if wrench_estimator.ready():
            break
        rate.sleep()

    if args.save is not None:
        # wait a bit to ensure bag is setup before we actually do anything
        # interesting
        time.sleep(3.0)

    path_switched = False
    path_angle = PATH_ANGLE

    t = rospy.Time.now().to_sec()
    t0 = t
    while not rospy.is_shutdown() and not signal_handler.received:
        # short acceleration phase to avoid excessive forces
        if t - t0 < ACCELERATION_DURATION:
            speed = (t - t0) * ACCELERATION_MAGNITUDE
        else:
            speed = PUSH_SPEED

        q = np.concatenate((robot.q, q_arm))
        r_bw_w = q[:2]
        C_wb = fp.rot2d(q[2])
        r_cw_w = r_bw_w - C_wb @ r_bc_b

        model.forward(q)
        C_wf = model.link_pose(link_idx=ft_idx, rotation_matrix=True)[1]
        f_f = wrench_estimator.wrench_filtered[:3]
        C_wb3 = rotz(q[2])
        f_w = C_wb3 @ ΔC @ C_wb3.T @ C_wf @ f_f

        f_norm = np.linalg.norm(f_w[:2])
        if not path_switched and f_norm > FORCE_MIN_THRESHOLD:
            print("minimum force threshold met")
            robot.brake()

            # update the path to go sideways now
            # TODO update from data

            # f_w_raw = C_wb3 @ ΔC @ C_wb3.T @ C_wf @ wrench_estimator.wrench[:3]
            # print(f"f_w_filt = {f_w[:2]}")
            # print(f"f_w_raw = {f_w_raw[:2]}")
            # path_angle = np.arctan2(pathdir[1], pathdir[0])

            rospy.sleep(3.0)

            f_f = wrench_estimator.wrench_filtered[:3]
            C_wb3 = rotz(q[2])
            f_w = C_wb3 @ ΔC @ C_wb3.T @ C_wf @ f_f

            pathdir = -fp.unit(f_w[:2])
            orthdir = fp.rot2d(np.pi / 2) @ pathdir

            path = fp.SegmentPath.line(orthdir, origin=r_cw_w)
            path_switched = True

            # publish message indicating we are moving sideways now
            msg = Time()
            msg.data = rospy.Time.now()
            time_pub.publish(msg)

        # direction of the path
        info = path.compute_closest_point_info(r_cw_w)
        θd = np.arctan2(info.direction[1], info.direction[0])

        # go in desired direction and correct for path deviations
        θp = θd - KY * info.offset
        v_ee_cmd = speed * fp.rot2d(θp) @ [1, 0]
        ω_cmd = Kω * fp.wrap_to_pi(PATH_ANGLE - q[2])
        cmd_vel = np.append(v_ee_cmd, ω_cmd)
        robot.publish_cmd_vel(cmd_vel, bodyframe=False)

        rate.sleep()
        t = rospy.Time.now().to_sec()

    params = {
        "pathdir0": PATHDIR,
        "orthdir0": ORTHDIR,
        "pathdir": pathdir,
        "orthdir": orthdir,
        "force_min": FORCE_MIN_THRESHOLD,
        "ΔC": ΔC,
        "r_bc_b": r_bc_b,
    }
    if args.save is not None:
        recorder.record_params(params)

    robot.brake()
    time.sleep(0.5)  # wait a bit to make sure brake is published

    if args.save is not None:
        recorder.close()


if __name__ == "__main__":
    main()
