#!/usr/bin/env python3
"""Push an object with mobile base + force-torque sensor."""
import argparse
import rospy
import numpy as np

from geometry_msgs.msg import Vector3, WrenchStamped

import mobile_manipulation_central as mm
from mmpush import *

import IPython


# Datasheet claims the sensor output rate is 100Hz, though rostopic says more
# like ~62Hz
RATE = 100  # Hz

# Direction to push
# Origin is taken as the EE's starting position
DIRECTION = np.array([0, 1])

# pushing speed
SPEED = 0.1

# control gains
KF = 0.1
KY = 0.3

# only control based on force when it is high enough (i.e. in contact with
# something)
FORCE_THRESHOLD = 5

# time constant for force filter
FILTER_TIME_CONSTANT = 0.1

WRENCH_TOPIC_NAME = "/robotiq_ft_wrench"


# TODO to be moved to mm_central
class ExponentialSmoother:
    """Exponential smoothing filter with time constant τ."""

    def __init__(self, τ, x0):
        self.τ = τ  # time constant
        self.x = x0  # initial state/guess

    def update(self, y, dt):
        """Update state estimate with measurement y taken dt seconds after the
        previous update."""
        # zero time-constant means no filtering is done
        if self.τ <= 0:
            return y
        c = 1.0 - np.exp(-dt / self.τ)
        self.x = c * y + (1 - c) * self.x
        return self.x


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

        self.smoother = ExponentialSmoother(τ=τ, x0=np.zeros(6))

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
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--open-loop",
        help="Use open-loop pushing rather than closed-loop control",
        action="store_true",
        default=False,
    )
    args = parser.parse_args()
    open_loop = args.open_loop

    rospy.init_node("push_control_node")

    home = mm.load_home_position(name="pushing")
    model = mm.MobileManipulatorKinematics()
    ft_idx = model.get_link_index("ft_sensor")
    q_arm = home[3:]

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

    # desired path
    q = np.concatenate((robot.q, q_arm))
    model.forward(q)
    r_fw_w = model.link_pose(link_idx=ft_idx)[0]
    c = r_fw_w[:2]
    path = StraightPath(DIRECTION, origin=c)
    cmd_vel = SPEED * np.append(DIRECTION, 0)

    wrench_estimator = WrenchEstimator(bias=bias, τ=FILTER_TIME_CONSTANT)

    while not rospy.is_shutdown():
        q = np.concatenate((robot.q, q_arm))
        model.forward(q)
        r_fw_w, C_wf = model.link_pose(link_idx=ft_idx, rotation_matrix=True)
        c = r_fw_w[:2]  # contact point

        f_f = wrench_estimator.wrench_filtered[:3]
        f_w = C_wf @ f_f
        f = f_w[:2]

        # only control based on force when it is high enough (i.e. in contact
        # with something)
        if not open_loop and np.linalg.norm(f) > FORCE_THRESHOLD:

            # force direction is negative to switch from sensed force to applied force
            direction = path.compute_travel_direction(c)
            θf = signed_angle(direction, -unit(f))
            Δy = path.compute_lateral_offset(c)

            θp = (KF + 1) * θf + KY * Δy
            vp = SPEED * rot2d(θp) @ direction
            cmd_vel = np.append(vp, 0)

            print(f"direction = {direction}")
            print(f"Δy = {Δy}")
            print(f"θf = {θf}")
            print(f"θp = {θp}")
            print(f"vp = {vp}")

        robot.publish_cmd_vel(cmd_vel, bodyframe=False)

        rate.sleep()

    robot.brake()


if __name__ == "__main__":
    main()
