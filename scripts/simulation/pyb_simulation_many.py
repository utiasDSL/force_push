import argparse
import itertools
import pickle
import time

import matplotlib.pyplot as plt
import numpy as np
from pyb_utils.frame import debug_frame_world
import pybullet as pyb
import seaborn
from spatialmath.base import rotz, r2q
import tqdm

import mobile_manipulation_central as mm
import force_push as fp

import IPython


# Hz
SIM_FREQ = 1000
CTRL_FREQ = 100

# seconds
DURATION = 400

# friction
CONTACT_MU = 0.2
SURFACE_MU = 1.0
OBSTACLE_MU = 0.25

# controller params
PUSH_SPEED = 0.1
# Kθ = 0.5
# KY = 0.1

Kθ = 0.3
KY = 0.1

# Kθ = 0.15
# KY = 0.05

CORRIDOR_RADIUS = 1.4

# slider params
SLIDER_MASS = 1.0
BOX_SLIDER_HALF_EXTENTS = (0.5, 0.5, 0.1)
CIRCLE_SLIDER_RADIUS = 0.5
CIRCLE_SLIDER_HEIGHT = 0.2

SLIDER_INIT_POS = np.array([0, 0, 0.1])
PUSHER_INIT_POS = np.array([-0.7, 0, 0.1])

# if the closest distance between pusher and slider exceeds this amount, then
# the trial is considered to have failed
FAILURE_DIST = 0.5

μ0s = [0, 0.5]
# μ0s = [0, 0.3, 0.6, 0.8]
y0s = [-0.4, 0, 0.4]
# θ0s = [-np.pi / 8, 0, np.pi / 8]
θ0s = [0]
s0s = [-0.4, 0, 0.4]

# μ0s = [0.5]
# y0s = [0.4]
# θ0s = [0]
# s0s = [0.4]


def simulate(sim, pusher, slider, controller):
    success = True
    r_pw_ws = []
    r_sw_ws = []
    forces = []
    ts = []

    t = 0
    steps = DURATION * SIM_FREQ
    for i in range(DURATION * SIM_FREQ):
        t = sim.timestep * i

        if i % CTRL_FREQ == 0:
            force = pusher.get_contact_force([slider.uid])
            r_pw_w = pusher.get_position()
            v_cmd = controller.update(r_pw_w[:2], force[:2])  #, 1. / CTRL_FREQ)
            pusher.command_velocity(np.append(v_cmd, 0))

            # record information
            r_pw_ws.append(r_pw_w)
            r_sw_ws.append(slider.get_position())
            forces.append(force)
            ts.append(t)

        sim.step()

        # check if the trial has failed (pusher has lost the slider)
        pts = pyb.getClosestPoints(pusher.uid, slider.uid, distance=10)
        dist = pts[0][8]
        if dist > FAILURE_DIST:
            success = False
            break

        # time.sleep(0.0001)

    ts = np.array(ts)
    r_pw_ws = np.array(r_pw_ws)
    r_sw_ws = np.array(r_sw_ws)
    forces = np.array(forces)
    return ts, r_pw_ws, r_sw_ws, success, forces


def setup_box_slider(position):
    slider = fp.BulletSquareSlider(
        position, mass=SLIDER_MASS, half_extents=BOX_SLIDER_HALF_EXTENTS
    )

    slider_vertices = fp.cuboid_vertices(BOX_SLIDER_HALF_EXTENTS)
    slider_masses = (
        SLIDER_MASS * np.ones(slider_vertices.shape[0]) / slider_vertices.shape[0]
    )
    I_max = fp.point_mass_system_inertia(slider_masses, slider_vertices)
    I_uni = fp.uniform_cuboid_inertia(
        mass=SLIDER_MASS, half_extents=BOX_SLIDER_HALF_EXTENTS
    )
    I_low = 0.5 * I_uni
    inertias = [I_low, I_uni, I_max]

    return slider, inertias


def setup_circle_slider(position):
    slider = fp.BulletCircleSlider(position)

    I_max = fp.thin_walled_cylinder_inertia(
        SLIDER_MASS, CIRCLE_SLIDER_RADIUS, CIRCLE_SLIDER_HEIGHT
    )
    I_uni = fp.uniform_cylinder_inertia(
        SLIDER_MASS, CIRCLE_SLIDER_RADIUS, CIRCLE_SLIDER_HEIGHT
    )
    I_low = 0.5 * I_uni
    # I_low2 = fp.uniform_cylinder_inertia(
    #     SLIDER_MASS, 0.1 * CIRCLE_SLIDER_RADIUS, CIRCLE_SLIDER_HEIGHT
    # )
    inertias = [I_low, I_uni, I_max]

    return slider, inertias


def setup_straight_path():
    return fp.SegmentPath.line(direction=[1, 0])


def setup_corner_path(corridor=False):
    path = fp.SegmentPath(
        [
            fp.LineSegment([0., 0], [3., 0]),
            fp.QuadBezierSegment([3., 0], [5., 0], [5, 2]),
            fp.LineSegment([5., 2], [5., 5], infinite=True),
        ],
    )

    if corridor:
        block1 = fp.BulletBlock([1, 6.5, 0.5], [2.5, 5, 0.5], mu=OBSTACLE_MU)
        block2 = fp.BulletBlock([7, 4.5, 0.5], [0.5, 7, 0.5], mu=OBSTACLE_MU)
        block3 = fp.BulletBlock([2.5, -2, 0.5], [4, 0.5, 0.5], mu=OBSTACLE_MU)

    return path


def plot_results(data):
    all_r_sw_ws = data["slider_positions"]
    all_forces = data["forces"]
    ts = data["times"]
    r_dw_ws = data["path_positions"]

    μs = np.array([p[1] for p in data["parameters"]])

    n = len(all_r_sw_ws)

    # plotting
    palette = seaborn.color_palette("deep")

    plt.figure()
    for i in range(0, n):
        if μs[i] > 0.7:
            color = palette[1]
            continue
        else:
            color = palette[0]
        plt.plot(
            all_r_sw_ws[i][:, 0], all_r_sw_ws[i][:, 1], color=color, alpha=0.2
        )
    plt.plot(r_dw_ws[:, 0], r_dw_ws[:, 1], "--", color="k")
    ax = plt.gca()
    ax.set_aspect("equal")
    plt.grid()

    plt.figure()
    for i in range(n):
        plt.plot(ts[i], all_forces[i][:, 0], color="r", alpha=0.5)
        plt.plot(ts[i], all_forces[i][:, 1], color="b", alpha=0.5)
    plt.grid()
    plt.title("Forces vs. time")
    plt.xlabel("Time [s]")
    plt.ylabel("Force [N]")

    plt.show()


def main():
    np.set_printoptions(precision=6, suppress=True)

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--slider",
        choices=["box", "circle"],
        help="Type of slider to use.",
        default="box",
    )
    parser.add_argument(
        "--environment",
        choices=["straight", "corner", "corridor"],
        help="Which environment to use",
        default="straight",
    )
    parser.add_argument("--save", help="Save data to this file.")
    parser.add_argument("--load", help="Load data from this file.")
    parser.add_argument("--no-gui", action="store_true", help="Disable simulation GUI.")
    args = parser.parse_args()

    if args.load is not None:
        with open(args.load, "rb") as f:
            data = pickle.load(f)
        print(f"Loaded processed data from {args.load}")
        plot_results(data)
        return

    sim = mm.BulletSimulation(1.0 / SIM_FREQ, gui=not args.no_gui)
    pyb.changeDynamics(sim.ground_uid, -1, lateralFriction=SURFACE_MU)
    pyb.resetDebugVisualizerCamera(
        cameraDistance=5.6,
        cameraYaw=28,
        cameraPitch=-48.6,
        cameraTargetPosition=[3.66, 0.42, 0.49],
    )

    pusher = fp.BulletPusher(PUSHER_INIT_POS, mu=CONTACT_MU)
    if args.slider == "box":
        slider, slider_inertias = setup_box_slider(SLIDER_INIT_POS)
    elif args.slider == "circle":
        slider, slider_inertias = setup_circle_slider(SLIDER_INIT_POS)

    # see e.g. <https://github.com/bulletphysics/bullet3/issues/4428>
    # pyb.changeDynamics(slider.uid, -1, contactDamping=100, contactStiffness=10000)

    if args.environment == "straight":
        path = setup_straight_path()
        corridor_radius = np.inf
    elif args.environment == "corner":
        path = setup_corner_path(corridor=False)
        corridor_radius = np.inf
    elif args.environment == "corridor":
        path = setup_corner_path(corridor=True)
        corridor_radius = CORRIDOR_RADIUS

    # somewhat janky: for now, we show both vertices for lines and just the
    # middle one for quadratic bezier segments
    for segment in path.segments:
        if type(segment) is fp.LineSegment:
            v1 = np.append(segment.v1, 0.1)
            debug_frame_world(0.2, tuple(v1), line_width=3)
        v2 = np.append(segment.v2, 0.1)
        debug_frame_world(0.2, tuple(v2), line_width=3)

    controller = fp.Controller(
        speed=PUSH_SPEED,
        kθ=Kθ,
        ky=KY,
        path=path,
        corridor_radius=corridor_radius,
    )

    data = {
        "slider": args.slider,
        "environment": args.environment,
        "duration": DURATION,
        "sim_freq": SIM_FREQ,
        "ctrl_freq": CTRL_FREQ,
        "push_speed": PUSH_SPEED,
        "kθ": Kθ,
        "ky": KY,
        "inertias": slider_inertias,
        "y0s": y0s,
        "θ0s": θ0s,
        "s0s": s0s,
        "μ0s": μ0s,
    }

    num_sims = len(slider_inertias) * len(y0s) * len(θ0s) * len(s0s) * len(μ0s)

    all_ts = []
    all_r_pw_ws = []
    all_r_sw_ws = []
    successes = []
    all_forces = []
    parameters = []

    count = 0
    with tqdm.tqdm(total=num_sims) as progress:
        for (I, μ0, y0, θ0, s0) in itertools.product(
            slider_inertias, μ0s, y0s, θ0s, s0s
        ):
            # set the new parameters
            r_pw_w = PUSHER_INIT_POS + [0, s0 + y0, 0]
            r_sw_w = SLIDER_INIT_POS + [0, y0, 0]
            Q_ws = r2q(rotz(θ0), order="xyzs")
            pyb.changeDynamics(pusher.uid, -1, lateralFriction=μ0)
            slider.set_inertia_diagonal(I)

            # reset everything to initial states
            pusher.reset(position=r_pw_w)
            slider.reset(position=r_sw_w, orientation=Q_ws)
            controller.reset()
            sim.step()

            # time.sleep(5.0)

            # run the sim
            ts, r_pw_ws, r_sw_ws, success, forces = simulate(sim, pusher, slider, controller)
            if not success:
                print(f"Trial {count + 1} failed.")
                print(f"I = {np.diag(I)}\nμ = {μ0}\ny0 = {y0}\nθ0 = {θ0}\ns0 = {s0}")
                IPython.embed()

            all_ts.append(ts)
            all_r_pw_ws.append(r_pw_ws)
            all_r_sw_ws.append(r_sw_ws)
            successes.append(success)
            all_forces.append(forces)
            parameters.append([I, μ0, y0, θ0, s0])

            progress.update(1)
            count += 1

    # parse path points to plot
    d = path.segments[-1].direction
    v = path.segments[-1].v2
    dist = np.max((np.vstack(all_r_sw_ws)[:, :2] - v) @ d)

    # TODO this needs to be fixed
    r_dw_ws = path.get_plotting_coords(dist=dist)

    data["times"] = all_ts
    data["pusher_positions"] = all_r_pw_ws
    data["slider_positions"] = all_r_sw_ws
    data["successes"] = successes
    data["path_positions"] = r_dw_ws
    data["forces"] = all_forces
    data["parameters"] = parameters

    if args.save is not None:
        with open(args.save, "wb") as f:
            pickle.dump(data, f)
        print(f"Saved processed data to {args.save}")

    plot_results(data)


if __name__ == "__main__":
    main()