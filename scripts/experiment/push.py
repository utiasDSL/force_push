#!/usr/bin/env python3
"""Push an object with mobile base + force-torque sensor."""
import rospy
import numpy as np

from geometry_msgs.msg import WrenchStamped

import mobile_manipulation_central as mm
from mmpush import *

import IPython


class ExponentialSmoother:
    def __init__(self, τ, x0):
        self.τ = τ
        self.x = x0

    def update(self, y, dt):
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
            "/robotiq_ft_wrench", WrenchStamped, self._wrench_cb
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
        rate = rospy.Rate(0.1)
        while not rospy.is_shutdown() and not self.done():
            rate.sleep()
        return self.sum / self.num_samples

    def done(self):
        return self.count >= self.num_samples


class WrenchEstimator:
    def __init__(self, bias=None, τ=0):
        if bias is None:
            bias = np.zeros(6)
        self.bias = bias
        self.wrench = np.zeros(6)
        self.wrench_filtered = np.zeros(6)

        self.smoother = ExponentialSmoother(τ=τ, x0=np.zeros(6))

        self.prev_time = rospy.Time.now().to_sec()
        self.wrench_sub = rospy.Subscriber(
            "/robotiq_ft_wrench", WrenchStamped, self._wrench_cb
        )

    def _wrench_cb(self, msg):
        t = mm.ros_utils.msg_time(msg)
        dt = t - self.prev_time
        self.prev_time = t

        f = msg.wrench.force
        τ = msg.wrench.torque
        self.wrench = np.array([f.x, f.y, f.z, τ.x, τ.y, τ.z]) - self.bias
        self.wrench_filtered = self.smoother.update(self.wrench, dt)


def main():
    rospy.init_node("push_control_node")

    home = mm.load_home_position(name="pushing")
    model = mm.MobileManipulatorKinematics()
    ft_idx = model.get_link_index("ft_sensor")
    q_arm = home[3:]

    speed = 0.1
    kf = 0.1
    ky = 0.1

    robot = mm.RidgebackROSInterface()

    print("Estimating F-T sensor bias...")
    bias_estimator = WrenchBiasEstimator()
    bias = bias_estimator.estimate()
    print(f"Done. Bias = {bias}")

    # we want to steer toward the line y(t) = y0
    y0 = robot.q[1]
    direction = np.array([1, 0])
    cmd_vel = speed * np.append(direction, 0)

    wrench_estimator = WrenchEstimator(bias=bias, τ=0.1)

    # TODO what is FT rate?
    rate = rospy.Rate(100)
    while not rospy.is_shutdown():
        q = np.concatenate((robot.q, q_arm))
        model.forward(q)
        C_wf = model.link_pose(link_idx=ft_idx, rotation_matrix=True)[1]

        f_f = wrench_estimator.wrench_filtered[:3]
        f_w = C_wf @ f_f
        f = f_w[:2]

        # only control based on force when it is high enough (i.e. in contact
        # with something)
        if np.linalg.norm(f) > 5:

            # force direction is negative to switch from sensed force to applied force
            θf = signed_angle(direction, -unit(f))
            Δy = robot.q[1] - y0

            θp = (kf + 1) * θf + ky * Δy
            vp = speed * np.array([np.cos(θp), np.sin(θp)])
            cmd_vel = np.append(vp, 0)

        robot.publish_cmd_vel(cmd_vel, bodyframe=False)


        rate.sleep()

    # TODO properly handle ctrl-C
    robot.brake()


if __name__ == "__main__":
    main()
