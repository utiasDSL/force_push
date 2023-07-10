import time

import pybullet as pyb
import numpy as np
import matplotlib.pyplot as plt
from pyb_utils.frame import debug_frame_world

import mobile_manipulation_central as mm
import force_push as fp

import IPython


# Hz
SIM_FREQ = 1000
CTRL_FREQ = 100

# seconds
DURATION = 20

CONTACT_MU = 0.2
SURFACE_MU = 1.0

# controller params
PUSH_SPEED = 0.1
Kθ = 0.5
KY = 0.1
LOOKAHEAD = 2.0


def simulate(sim, pusher, slider, controller):
    r_pw_ws = []
    r_sw_ws = []
    ts = []

    t = 0
    steps = DURATION * SIM_FREQ
    for i in range(DURATION * SIM_FREQ):
        t = sim.timestep * i

        if i % CTRL_FREQ == 0:
            force = pusher.get_contact_force(slider)
            r_pw_w = pusher.get_position()
            v_cmd = controller.update(r_pw_w[:2], force[:2])
            pusher.command_velocity(np.append(v_cmd, 0))

            # record information
            r_pw_ws.append(r_pw_w)
            r_sw_ws.append(slider.get_position())
            ts.append(t)

        sim.step()
        # time.sleep(sim.timestep)

    ts = np.array(ts)
    r_pw_ws = np.array(r_pw_ws)
    r_sw_ws = np.array(r_sw_ws)
    return ts, r_pw_ws, r_sw_ws


def main():
    sim = mm.BulletSimulation(1.0 / SIM_FREQ)
    pyb.changeDynamics(sim.ground_uid, -1, lateralFriction=SURFACE_MU)

    pusher = fp.BulletPusher([0, 0, 0.1], mu=CONTACT_MU)
    # slider = fp.BulletCircleSlider([0.7, 0.25, 0.1])
    slider = fp.BulletSquareSlider([0.7, 0.25, 0.1])

    # see e.g. <https://github.com/bulletphysics/bullet3/issues/4428>
    pyb.changeDynamics(slider.uid, -1, contactDamping=100, contactStiffness=10000)

    # block1 = fp.BulletBlock([2, 1.5, 0.5], [2, 0.5, 0.5], mu=0.5)
    # block2 = fp.BulletBlock([6, 0.5, 0.5], [0.5, 1.5, 0.5], mu=0.5)
    # vertices = np.array([[0, 0], [5, 0]])
    # path = fp.SegmentPath(vertices, final_direction=[0, 1])

    path = fp.SegmentPath.line(direction=[1, 0])

    for vertex in path.vertices:
        r = np.append(vertex, 0.1)
        debug_frame_world(0.2, tuple(r), line_width=3)

    controller = fp.Controller(
        speed=PUSH_SPEED, kθ=Kθ, ky=KY, path=path, lookahead=LOOKAHEAD
    )

    ts, r_pw_ws, r_sw_ws = simulate(sim, pusher, slider, controller)
    pusher.reset()
    slider.reset()
    controller.reset()
    sim.step()
    ts, r_pw_ws, r_sw_ws = simulate(sim, pusher, slider, controller)

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
