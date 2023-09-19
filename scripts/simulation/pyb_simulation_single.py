#!/usr/bin/env python3
import time

import pybullet as pyb
import numpy as np
import matplotlib.pyplot as plt
import pyb_utils
from spatialmath.base import q2r, tr2rpy

import mobile_manipulation_central as mm
import force_push as fp

import IPython


# Hz
SIM_FREQ = 1000
CTRL_FREQ = 100

DURATION = 200

CONTACT_MU = 0.5
SURFACE_MU = 0.25

# slider params
SLIDER_MASS = 1.0
BOX_SLIDER_HALF_EXTENTS = (0.5, 0.5, 0.05)
CIRCLE_SLIDER_RADIUS = 0.5
CIRCLE_SLIDER_HEIGHT = 0.1
SLIDER_CONTACT_DAMPING = 100
SLIDER_CONTACT_STIFFNESS = 10000

# pusher params
PUSHER_MASS = 100
PUSHER_RADIUS = 0.05

CON_INC = 0.1
DIV_INC = 0.1

SLIDER_INIT_POS = np.array([0, 0, 0.05])
PUSHER_INIT_POS = np.array([-0.7, 0, 0.05])

# controller params
PUSH_SPEED = 0.1
Kθ = 0.5
KY = 0.1


def main():
    sim = mm.BulletSimulation(1.0 / SIM_FREQ)
    pyb.changeDynamics(sim.ground_uid, -1, lateralFriction=SURFACE_MU)

    # pusher = fp.BulletPusher([0, 0, 0.1], mu=CONTACT_MU)
    pusher = fp.BulletPusher(
        PUSHER_INIT_POS, mu=CONTACT_MU, mass=PUSHER_MASS, radius=PUSHER_RADIUS
    )

    # slider = fp.BulletCircleSlider([0.7, 0.25, 0.1])
    # slider = fp.BulletSquareSlider([1, 0.25, 0.1])
    slider = fp.BulletSquareSlider(
        SLIDER_INIT_POS, mass=SLIDER_MASS, half_extents=BOX_SLIDER_HALF_EXTENTS
    )

    # see e.g. <https://github.com/bulletphysics/bullet3/issues/4428>
    pyb.changeDynamics(slider.uid, -1, contactDamping=100, contactStiffness=10000)

    # constraint to keep the slider fixed in place (to test out recovery
    # mechanism)
    pyb.createConstraint(
        slider.uid,
        -1,
        -1,
        -1,
        pyb.JOINT_FIXED,
        # pyb.JOINT_PRISMATIC,
        jointAxis=[0, 1, 0],
        parentFramePosition=[0, 0, 0],
        childFramePosition=SLIDER_INIT_POS,
    )

    # block1 = fp.BulletBlock([2, 1.5, 0.5], [2, 0.5, 0.5], mu=0.5)
    # block2 = fp.BulletBlock([6, 0.5, 0.5], [0.5, 1.5, 0.5], mu=0.5)
    # vertices = np.array([[0, 0], [5, 0]])
    # path = fp.SegmentPath(vertices, final_direction=[0, 1])

    # pillar1 = fp.BulletPillar([3, 0.1, 0.5], radius=0.2, mu=0.5)
    # path = fp.SegmentPath.line(direction=[1, 0])

    path = fp.SegmentPath(
        [
            fp.QuadBezierSegment([-1.0, 0], [0, 0], [0, 1.0]),
            fp.LineSegment([0, 1.0], [0, 5], infinite=True),
        ],
    )
    for segment in path.segments:
        if type(segment) is fp.LineSegment:
            v1 = np.append(segment.v1, 0.1)
            pyb_utils.debug_frame_world(0.2, tuple(v1), line_width=3)
        v2 = np.append(segment.v2, 0.1)
        pyb_utils.debug_frame_world(0.2, tuple(v2), line_width=3)

    controller = fp.PushController(
        speed=PUSH_SPEED, kθ=Kθ, ky=KY, div_inc=DIV_INC, con_inc=CON_INC, path=path
    )

    r_pw_ws = []
    r_sw_ws = []
    ts = []

    t = 0
    steps = DURATION * SIM_FREQ
    for i in range(DURATION * SIM_FREQ):
        t = sim.timestep * i

        if i % CTRL_FREQ == 0:
            force = pusher.get_contact_force([slider.uid])
            r_pw_w = pusher.get_pose()[0]
            v_cmd = controller.update(r_pw_w[:2], force[:2])
            pusher.command_velocity(np.append(v_cmd, 0))

            # record information
            r_pw_ws.append(r_pw_w)
            r_sw_ws.append(slider.get_pose()[0])
            ts.append(t)

        # v_s = pyb.getBaseVelocity(slider.uid)[0]
        # pos, orn = pyb.getBasePositionAndOrientation(slider.uid)
        # yaw = tr2rpy(q2r(orn, order="xyzs"))[2]
        # pyb.resetBaseVelocity(slider.uid, linearVelocity=[-10 * pos[0], v_s[1], 0], angularVelocity=[0, 0, -10 * yaw])

        sim.step()
        time.sleep(sim.timestep)

    r_pw_ws = np.array(r_pw_ws)
    r_sw_ws = np.array(r_sw_ws)

    d = path.directions[-1, :]
    v = path.vertices[-1, :]
    dist = np.max(np.concatenate(((r_pw_ws[:, :2] - v) @ d, (r_sw_ws[:, :2] - v) @ d)))
    r_dw_ws = path.get_coords(dist)

    plt.figure()
    plt.plot(r_sw_ws[:, 0], r_sw_ws[:, 1], label="Slider")
    plt.plot(r_pw_ws[:, 0], r_pw_ws[:, 1], label="Pusher")
    plt.plot(r_dw_ws[:, 0], r_dw_ws[:, 1], "--", color="k", label="Desired")
    plt.legend()
    plt.grid()
    plt.show()


if __name__ == "__main__":
    main()
