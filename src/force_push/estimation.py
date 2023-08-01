import numpy as np
import rospy

from geometry_msgs.msg import Vector3, WrenchStamped

import mobile_manipulation_central as mm


WRENCH_TOPIC_NAME = "/robotiq_ft_wrench"


class WrenchBiasEstimator:
    """Estimator for the F/T sensor bias."""

    def __init__(self, num_samples=100, topic_name=WRENCH_TOPIC_NAME):
        self.num_samples = num_samples

        self.count = 0
        self.sum = np.zeros(6)

        self.wrench_sub = rospy.Subscriber(topic_name, WrenchStamped, self._wrench_cb)

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

    def estimate(self, hz):
        rate = rospy.Rate(hz)
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
    """Contact wrench estimator"""

    def __init__(self, bias=None, τ=0, publish=True, topic_name=WRENCH_TOPIC_NAME):
        """Initialize the estimator.

        Parameters:
            bias: sensor bias, removed from each measurement
            τ: filter time contast
            publish: True to publish the results to ROS, False otherwise (default: True)
            topic_name: topic to subscribe to raw wrench measurements
        """
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
        self.wrench_sub = rospy.Subscriber(topic_name, WrenchStamped, self._wrench_cb)

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
