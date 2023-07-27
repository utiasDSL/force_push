#!/usr/bin/env python3
"""Push an object with mobile base + force-torque sensor."""
import argparse
import yaml

import rospy
import numpy as np

from geometry_msgs.msg import Vector3, WrenchStamped

import mobile_manipulation_central as mm
import force_push as fp

import IPython


# Datasheet claims the sensor output rate is 100Hz, though rostopic says more
# like ~62Hz
RATE = 100  # Hz

# Direction to push
# Origin is taken as the EE's starting position
# DIRECTION = np.array([0, 1])
DIRECTION = fp.rot2d(np.deg2rad(125)) @ np.array([1, 0])

# pushing speed
PUSH_SPEED = 0.1

# control gains
Kθ = 0.3
KY = 0.3
Kω = 1

# angular velocity bounds
ANG_VEL_UB = 0.1
ANG_VEL_LB = -ANG_VEL_UB

# only control based on force when it is high enough (i.e. in contact with
# something)
FORCE_MIN_THRESHOLD = 5
FORCE_MAX_THRESHOLD = 50

# time constant for force filter
FILTER_TIME_CONSTANT = 0.1

WRENCH_TOPIC_NAME = "/robotiq_ft_wrench"


class WrenchBiasEstimator:
    def __init__(self, num_samples=100):
        self.num_samples = num_samples

        self.count = 0
        self.sum = np.zeros(6)

        self.wrench_sub = rospy.Subscriber(
            WRENCH_TOPIC_NAME, WrenchStamped, self._wrench_cb
        )

    def _wrench_cb(self, msg):
        if not self.done():
            f = msg.wrench.force
            τ = msg.wrench.torque
            wrench = np.array([f.x, f.y, f.z, τ.x, τ.y, τ.z])
            self.sum += wrench
            self.count += 1
        else:
            # unsubscribe once we have enough samples
            self.wrench_sub.unregister()

    def estimate(self):
        rate = rospy.Rate(RATE)
        while not rospy.is_shutdown() and not self.done():
            rate.sleep()
        return self.sum / self.num_samples

    def done(self):
        return self.count >= self.num_samples


def vec_msg_from_array(v):
    msg = Vector3()
    msg.x = v[0]
    msg.y = v[1]
    msg.z = v[2]
    return msg


class WrenchEstimator:
    def __init__(self, bias=None, τ=0, publish=True):
        if bias is None:
            bias = np.zeros(6)
        self.bias = bias
        self.wrench = np.zeros(6)
        self.wrench_filtered = np.zeros(6)

        self.smoother = mm.ExponentialSmoother(τ=τ, x0=np.zeros(6))

        # publish for logging purposes
        self.publish = publish
        if self.publish:
            self.wrench_raw_pub = rospy.Publisher(
                "/wrench/raw", WrenchStamped, queue_size=1
            )
            self.wrench_filt_pub = rospy.Publisher(
                "/wrench/filtered", WrenchStamped, queue_size=1
            )

        self.prev_time = rospy.Time.now().to_sec()
        self.wrench_sub = rospy.Subscriber(
            WRENCH_TOPIC_NAME, WrenchStamped, self._wrench_cb
        )

    def _publish_wrenches(self):
        """Publish raw and filtered wrenches."""
        now = rospy.Time.now()

        msg_raw = WrenchStamped()
        msg_raw.header.stamp = now
        msg_raw.wrench.force = vec_msg_from_array(self.wrench[:3])
        msg_raw.wrench.torque = vec_msg_from_array(self.wrench[3:])

        msg_filt = WrenchStamped()
        msg_filt.header.stamp = now
        msg_filt.wrench.force = vec_msg_from_array(self.wrench_filtered[:3])
        msg_filt.wrench.torque = vec_msg_from_array(self.wrench_filtered[3:])

        self.wrench_raw_pub.publish(msg_raw)
        self.wrench_filt_pub.publish(msg_filt)

    def _wrench_cb(self, msg):
        """Call back for wrench measurements received from FT sensor."""
        t = mm.ros_utils.msg_time(msg)
        dt = t - self.prev_time
        self.prev_time = t

        f = msg.wrench.force
        τ = msg.wrench.torque
        self.wrench = np.array([f.x, f.y, f.z, τ.x, τ.y, τ.z]) - self.bias
        self.wrench_filtered = self.smoother.update(self.wrench, dt)

        if self.publish:
            self._publish_wrenches()


def main():
    np.set_printoptions(precision=6, suppress=True)

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--open-loop",
        help="Use open-loop pushing rather than closed-loop control",
        action="store_true",
        default=False,
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
    bias_estimator = WrenchBiasEstimator()
    bias = bias_estimator.estimate()
    print(f"Done. Bias = {bias}")

    wrench_estimator = WrenchEstimator(bias=bias, τ=FILTER_TIME_CONSTANT)

    # desired path
    q = robot.q
    r_bw_w = q[:2]
    C_wb = fp.rot2d(q[2])
    r_cw_w = r_bw_w - C_wb @ r_bc_b

    path = fp.SegmentPath(
        [
            fp.LineSegment([0.0, 0], [0.0, 1]),
            fp.QuadBezierSegment([0.0, 1], [0.0, 3], [-2.0, 3]),
            fp.LineSegment([-2.0, 3], [-3.0, 3], infinite=True),
        ],
        origin=r_cw_w,
    )

    # pushing controller
    controller = fp.Controller(
        speed=PUSH_SPEED,
        kθ=Kθ,
        ky=KY,
        path=path,
        force_min=FORCE_MIN_THRESHOLD,
        force_max=FORCE_MAX_THRESHOLD,
    )

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
        pathdir, _ = path.compute_direction_and_offset(r_cw_w)

        # in open-loop mode we just follow the path rather than controlling to
        # push the slider
        if open_loop:
            v_ee_cmd = PUSH_SPEED * pathdir
        else:
            v_ee_cmd = controller.update(r_cw_w, f)

        # desired angular velocity is calculated to align the robot with the
        # current path direction
        θd = np.arctan2(pathdir[1], pathdir[0])
        ωd = Kω * fp.wrap_to_pi(θd - q[2])
        ωd = min(ANG_VEL_UB, max(ANG_VEL_LB, ωd))
        V_ee_cmd = np.append(v_ee_cmd, ωd)

        # move the base so that the desired EE velocity is achieved
        δ = np.append(fp.skew2d(V_ee_cmd[2]) @ C_wb @ r_bc_b, 0)
        cmd_vel = V_ee_cmd + δ

        # print(f"direction = {direction}")
        # print(f"Δy = {Δy}")
        # print(f"θf = {θf}")
        # print(f"θp = {θp}")
        print(f"cmd_vel = {cmd_vel}")

        robot.publish_cmd_vel(cmd_vel, bodyframe=False)

        rate.sleep()

    robot.brake()


if __name__ == "__main__":
    main()
