import time
import itertools

import pybullet as pyb
import numpy as np
import matplotlib.pyplot as plt
from pyb_utils.frame import debug_frame_world
import tqdm
from spatialmath.base import rotz, r2q
import seaborn

import mobile_manipulation_central as mm
import force_push as fp

import IPython


# Hz
SIM_FREQ = 1000
CTRL_FREQ = 100

# seconds
DURATION = 200

CONTACT_MU = 0.2
SURFACE_MU = 1.0
OBSTACLE_MU = 0.25

# controller params
PUSH_SPEED = 0.1
# Kθ = 0.5
# KY = 0.1
Kθ = 0.3
KY = 0.1
LOOKAHEAD = 2.0


def simulate(sim, pusher, slider, controller, obstacles):
    r_pw_ws = []
    r_sw_ws = []
    ts = []

    uids = [slider.uid] + [o.uid for o in obstacles]

    t = 0
    steps = DURATION * SIM_FREQ
    for i in range(DURATION * SIM_FREQ):
        t = sim.timestep * i

        if i % CTRL_FREQ == 0:
            force = pusher.get_contact_force(uids)
            r_pw_w = pusher.get_position()
            v_cmd = controller.update(r_pw_w[:2], force[:2])
            pusher.command_velocity(np.append(v_cmd, 0))

            # record information
            r_pw_ws.append(r_pw_w)
            r_sw_ws.append(slider.get_position())
            ts.append(t)
        # pusher.control_velocity(np.append(v_cmd, 0))

        sim.step()
        # time.sleep(sim.timestep)

    ts = np.array(ts)
    r_pw_ws = np.array(r_pw_ws)
    r_sw_ws = np.array(r_sw_ws)
    return ts, r_pw_ws, r_sw_ws


def main():
    sim = mm.BulletSimulation(1.0 / SIM_FREQ, gui=True)
    pyb.changeDynamics(sim.ground_uid, -1, lateralFriction=SURFACE_MU)

    pusher = fp.BulletPusher([0, 0, 0.1], mu=CONTACT_MU)
    # slider = fp.BulletCircleSlider([0.7, 0.25, 0.1])
    slider = fp.BulletSquareSlider([0.7, 0.25, 0.1])

    # see e.g. <https://github.com/bulletphysics/bullet3/issues/4428>
    pyb.changeDynamics(slider.uid, -1, contactDamping=100, contactStiffness=10000)

    block1 = fp.BulletBlock([1, 4, 0.5], [2.5, 2.5, 0.5], mu=OBSTACLE_MU)
    block2 = fp.BulletBlock([7, 2.5, 0.5], [0.5, 5, 0.5], mu=OBSTACLE_MU)
    block3 = fp.BulletBlock([2.5, -2, 0.5], [4, 0.5, 0.5], mu=OBSTACLE_MU)
    obstacles = [block1, block2, block3]

    # corner path
    vertices = np.array([[0, 0], [5, 0]])
    path = fp.SegmentPath(vertices, final_direction=[0, 1])

    # straight path
    # path = fp.SegmentPath.line(direction=[1, 0])

    for vertex in path.vertices:
        r = np.append(vertex, 0.1)
        debug_frame_world(0.2, tuple(r), line_width=3)

    controller = fp.Controller(
        speed=PUSH_SPEED, kθ=Kθ, ky=KY, path=path, lookahead=LOOKAHEAD
    )

    # TODO we would also like to vary the inertia

    # y0s = [-0.4, 0, 0.4]
    # θ0s = [-np.pi / 8, 0, np.pi / 8]
    # s0s = [-0.4, 0, 0.4]
    # μ0s = [0, 0.5, 1.0]

    # TODO this one is problematic!
    # y0s = [-0.4]
    # θ0s = [-np.pi / 8]
    # s0s = [0]

    y0s = [-0.4, 0, 0.4]
    θ0s = [-np.pi / 8, 0, np.pi / 8]
    s0s = [-0.4, 0, 0.4]
    μ0s = [1.0]

    num_sims = len(y0s) * len(θ0s) * len(s0s) * len(μ0s)

    all_ts = []
    all_r_pw_ws = []
    all_r_sw_ws = []

    r_pw_w0 = np.array([0, 0, 0.1])
    r_sw_w0 = np.array([0.7, 0, 0.1])

    with tqdm.tqdm(total=num_sims) as progress:
        for (μ0, y0, θ0, s0) in itertools.product(μ0s, y0s, θ0s, s0s):
            # reset everything to proper states
            r_pw_w = r_pw_w0 + [0, s0 + y0, 0]
            r_sw_w = r_sw_w0 + [0, y0, 0]
            Q_ws = r2q(rotz(θ0), order="xyzs")
            pyb.changeDynamics(pusher.uid, -1, lateralFriction=μ0)

            pusher.reset(position=r_pw_w)
            slider.reset(position=r_sw_w, orientation=Q_ws)
            controller.reset()
            sim.step()

            # time.sleep(5.0)

            # run the sim
            ts, r_pw_ws, r_sw_ws = simulate(sim, pusher, slider, controller, obstacles)
            all_ts.append(ts)
            all_r_pw_ws.append(r_pw_ws)
            all_r_sw_ws.append(r_sw_ws)
            progress.update(1)

    # plotting
    palette = seaborn.color_palette("deep")

    d = path.directions[-1, :]
    v = path.vertices[-1, :]
    dist = np.max((np.vstack(all_r_sw_ws)[:, :2] - v) @ d)
    r_dw_ws = path.get_coords(dist=dist)

    plt.figure()
    plt.plot(r_dw_ws[:, 0], r_dw_ws[:, 1], "--", color="k")
    for i in range(num_sims):
        plt.plot(
            all_r_sw_ws[i][:, 0], all_r_sw_ws[i][:, 1], color=palette[0], alpha=0.2
        )
    plt.grid()
    plt.show()


if __name__ == "__main__":
    main()
