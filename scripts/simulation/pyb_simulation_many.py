#!/usr/bin/env python3
import argparse
import datetime
import itertools
import pickle
from pathlib import Path
import time

import matplotlib.pyplot as plt
import numpy as np
import pyb_utils
import pybullet as pyb
import seaborn
from spatialmath.base import rotz, r2q
import tqdm
import rospkg

import mobile_manipulation_central as mm
import force_push as fp

import IPython


# Hz
SIM_FREQ = 1000
CTRL_FREQ = 100
CTRL_STEP = SIM_FREQ // CTRL_FREQ

# seconds
DURATION = 300

# friction
# slider μ is set to 1
SURFACE_MU = 0.25
OBSTACLE_MU = 0.25
# SURFACE_MU = 0.5
# OBSTACLE_MU = 0.5

# controller params
PUSH_SPEED = 0.1
Kθ = 0.3
KY = 0.1
KF = 0.003
CON_INC = 0.1

FORCE_MIN_THRESHOLD = 1
FORCE_MAX_THRESHOLD = 50

# slider params
SLIDER_MASS = 1.0
BOX_SLIDER_HALF_EXTENTS = (0.5, 0.5, 0.06)
CIRCLE_SLIDER_RADIUS = 0.5
CIRCLE_SLIDER_HEIGHT = 0.12
DEFAULT_SLIDER_CONTACT_DAMPING = 1e2
DEFAULT_SLIDER_CONTACT_STIFFNESS = 1e4
SLIDER_LOW_INERTIA_MULT = 1.0 / 2.0

SLIDER_INIT_POS = np.array([0, 0, 0.06])

# starting position of pusher relative to the desired initial contact point
PUSHER_INIT_REL_POS = np.array([-0.5, 0, 0])

EE_OBS_MIN_DIST = 0.1

# if the distance between pusher and slider exceeds this amount, then the trial
# is considered to have failed
FAILURE_DIST = 2.0

# trial fails if no force occurs for this many seconds
FAILURE_TIME = 20.0

# variable parameters
I_mask = [True, True, True]
μ0s = [0, 0.5, 1.0]
y0s = [-0.4, 0, 0.4]
θ0s = [-np.pi / 8, 0, np.pi / 8]
s0s = [-0.4, 0, 0.4]

# I_mask = [False, True, False]
# μ0s = [0]
# y0s = [-0.4]
# θ0s = [-np.pi / 8]
# s0s = [-0.4]

# time constant for force filter
FILTER_TIME_CONSTANT = 0.05

START_AT_TRIAL = 0

FILM_FACE_SWITCHING_EXAMPLE = False

if FILM_FACE_SWITCHING_EXAMPLE:
    I_mask = [False, True, False]
    μ0s = [0]
    y0s = [-0.4]
    θ0s = [-np.pi / 8]
    s0s = [-0.4]


def simulate(sim, pusher, slider, push_controller, force_controller, blocks):
    success = True
    r_pw_ws = []
    r_sw_ws = []
    Q_wss = []
    v_cmds = []
    forces = []
    in_contacts = []
    ts = []

    last_force_time = 0

    smoother = mm.ExponentialSmoother(τ=FILTER_TIME_CONSTANT, x0=np.zeros(3))

    if FILM_FACE_SWITCHING_EXAMPLE:
        timestamp = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        camera = pyb_utils.Camera.from_distance_rpy(
            target_position=[4.14, 0.57, -0.16],
            distance=4,
            roll=0,
            pitch=-52.6,
            yaw=8,
        )
        recorder = pyb_utils.FrameRecorder(
            camera, fps=25, dirname=f"face_switching_frames_{timestamp}"
        )

    t = 0
    i = 0
    first_contact = False
    first_contact_index = 0
    max_pen_dist = 0
    while i <= first_contact_index + DURATION * SIM_FREQ:

        # TODO should this be done inside the below if? this would ensure the
        # correct overall duration as recorded
        # check if first contact has been made: we want to simulate DURATION
        # seconds after this time
        # if not first_contact:
        #     points = pyb_utils.getContactPoints(
        #         pusher.uid, slider.uid, pusher.tool_idx, -1
        #     )
        #     if len(points) > 0:
        #         first_contact = True
        #         first_contact_time = t

        if i % CTRL_STEP == 0:
            # force applied on slider
            points = pyb_utils.getContactPoints(
                slider.uid, pusher.uid, -1, pusher.tool_idx
            )
            assert len(points) <= 1
            force = pyb_utils.get_points_contact_wrench(points)[0]
            force = smoother.update(force, dt=CTRL_STEP * sim.timestep)
            f = force[:2]

            # get contact force and pusher position
            if np.linalg.norm(f) >= FORCE_MIN_THRESHOLD:
                last_force_time = t
            r_pw_w = pusher.get_joint_states()[0]

            # generate command
            v_cmd = push_controller.update(position=r_pw_w[:2], force=f)
            v_cmd = force_controller.update(force=f, v_cmd=v_cmd)
            pusher.command_velocity(v_cmd)

            # we can use this to check for penetration distance with the
            # obstacles
            # if blocks is not None:
            #     for block in blocks:
            #         pts = pyb_utils.getContactPoints(slider.uid, block.uid)
            #         for pt in pts:
            #             if pt.contactDistance < max_pen_dist:
            #                 max_pen_dist = pt.contactDistance
            #                 print(max_pen_dist)

            # contact info
            in_contact = len(points) > 0
            if not first_contact and in_contact:
                first_contact = True
                first_contact_index = i

            # record information
            r_sw_w, Q_ws = slider.get_pose()
            r_pw_ws.append(r_pw_w)
            r_sw_ws.append(r_sw_w)
            Q_wss.append(Q_ws)
            v_cmds.append(v_cmd)
            forces.append(force)
            in_contacts.append(in_contact)
            ts.append(t)

            # check if the trial has failed (pusher has lost the slider)
            if np.linalg.norm(r_pw_w - r_sw_w[:2]) > FAILURE_DIST:
                success = False
                print("Pusher and slider too far apart!")
                break

            # check if we've lost contact for too long
            if t - last_force_time > FAILURE_TIME:
                success = False
                print("Loss of contact for too long!")
                break

        if FILM_FACE_SWITCHING_EXAMPLE:
            recorder.capture_frame(t)

        i += 1
        t = i * sim.timestep
        sim.step()
        # time.sleep(0.001)

    first_contact_idx = np.argmax(in_contacts)
    t0 = ts[first_contact_idx]
    tf = ts[-1]
    print(f"duration = {tf - t0}")

    ts = np.array(ts)
    r_pw_ws = np.array(r_pw_ws)
    r_sw_ws = np.array(r_sw_ws)
    Q_wss = np.array(Q_wss)
    v_cmds = np.array(v_cmds)
    forces = np.array(forces)
    return ts, r_pw_ws, r_sw_ws, Q_wss, v_cmds, success, forces, in_contacts


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
    I_low = SLIDER_LOW_INERTIA_MULT * I_uni
    inertias = [I_low, I_uni, I_max]

    return slider, inertias


def setup_circle_slider(position):
    slider = fp.BulletCircleSlider(
        position, radius=CIRCLE_SLIDER_RADIUS, height=CIRCLE_SLIDER_HEIGHT
    )

    I_max = fp.thin_walled_cylinder_inertia(
        SLIDER_MASS, CIRCLE_SLIDER_RADIUS, CIRCLE_SLIDER_HEIGHT
    )
    I_uni = fp.uniform_cylinder_inertia(
        SLIDER_MASS, CIRCLE_SLIDER_RADIUS, CIRCLE_SLIDER_HEIGHT
    )
    I_low = SLIDER_LOW_INERTIA_MULT * I_uni
    inertias = [I_low, I_uni, I_max]

    return slider, inertias


def setup_straight_path():
    obstacles = None
    blocks = None
    return fp.SegmentPath.line(direction=[1, 0]), obstacles, blocks


def setup_corner_path(corridor=False):
    path = fp.SegmentPath(
        [
            fp.LineSegment([0.0, 0], [3.0, 0]),
            fp.CircularArcSegment(center=[3.0, 2.0], point=[3.0, 0], angle=np.pi / 2),
            fp.LineSegment([5.0, 2], [5.0, 5], infinite=True),
        ],
    )

    if corridor:
        block1 = fp.BulletBlock([1, 6.5, 0.5], [2.5, 5, 0.5], mu=OBSTACLE_MU)
        block2 = fp.BulletBlock([7, 4.5, 0.5], [0.5, 7, 0.5], mu=OBSTACLE_MU)
        block3 = fp.BulletBlock([2.5, -2, 0.5], [4, 0.5, 0.5], mu=OBSTACLE_MU)

        obstacles = [
            fp.LineSegment([-1.5, 1.5], [3.5, 1.5]),
            fp.LineSegment([-1.5, -1.5], [6.5, -1.5]),
            fp.LineSegment([3.5, 1.5], [3.5, 11.5]),
            fp.LineSegment([6.5, -1.5], [6.5, 11.5]),
        ]
        blocks = [block1, block2, block3]
    else:
        blocks = None
        obstacles = None

    return path, obstacles, blocks


def make_urdf_file():
    rospack = rospkg.RosPack()
    path = Path(rospack.get_path("force_push")) / "urdf/urdf/sim_pusher.urdf"
    if not path.parent.exists():
        path.parent.mkdir()

    includes = ["$(find force_push)/urdf/xacro/sim_pusher.urdf.xacro"]
    mm.XacroDoc.from_includes(includes).to_urdf_file(path)
    return path.as_posix()


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
    parser.add_argument(
        "--stiffness",
        help="Slider contact stiffness.",
        default=DEFAULT_SLIDER_CONTACT_STIFFNESS,
    )
    parser.add_argument(
        "--damping",
        help="Slider contact damping.",
        default=DEFAULT_SLIDER_CONTACT_DAMPING,
    )
    parser.add_argument("--save", help="Save data to this file.")
    parser.add_argument("--load", help="Load data from this file.")
    parser.add_argument("--no-gui", action="store_true", help="Disable simulation GUI.")
    args = parser.parse_args()

    if args.load is not None:
        with open(args.load, "rb") as f:
            data = pickle.load(f)
        print(f"Loaded processed data from {args.load}")
        fp.plot_simulation_results(data)
        plt.show()
        return

    sim = mm.BulletSimulation(1.0 / SIM_FREQ, gui=not args.no_gui, extra_gui=False)
    pyb.changeDynamics(sim.ground_uid, -1, lateralFriction=SURFACE_MU)
    pyb.resetDebugVisualizerCamera(
        cameraDistance=5.6,
        cameraYaw=28,
        cameraPitch=-48.6,
        cameraTargetPosition=[3.66, 0.42, 0.49],
    )

    urdf_path = make_urdf_file()
    pusher = fp.BulletPusher(urdf_path, SLIDER_INIT_POS + PUSHER_INIT_REL_POS)
    if args.slider == "box":
        slider, slider_inertias = setup_box_slider(SLIDER_INIT_POS)
    elif args.slider == "circle":
        slider, slider_inertias = setup_circle_slider(SLIDER_INIT_POS)
    slider_inertias = [slider_inertias[i] for i in range(3) if I_mask[i]]

    slider.set_contact_parameters(
        stiffness=float(args.stiffness), damping=float(args.damping)
    )
    # print(pyb_utils.getDynamicsInfo(slider.uid, -1))

    # pyb.changeDynamics(pusher.uid, -1, collisionMargin=0)
    # pyb.changeDynamics(slider.uid, -1, collisionMargin=0)

    if args.environment == "straight":
        path, obstacles, blocks = setup_straight_path()
    elif args.environment == "corner":
        path, obstacles, blocks = setup_corner_path(corridor=False)
    elif args.environment == "corridor":
        path, obstacles, blocks = setup_corner_path(corridor=True)

    # somewhat janky: for now, we show both vertices for lines and just the
    # middle one for quadratic bezier segments
    for segment in path.segments:
        if type(segment) is fp.LineSegment:
            v1 = np.append(segment.v1, 0.1)
            pyb_utils.debug_frame_world(0.2, tuple(v1), line_width=3)
        v2 = np.append(segment.v2, 0.1)
        pyb_utils.debug_frame_world(0.2, tuple(v2), line_width=3)

    if obstacles is not None:
        for obstacle in obstacles:
            v1 = np.append(obstacle.v1, 1.0)
            v2 = np.append(obstacle.v2, 1.0)
            pyb_utils.debug_frame_world(0.2, tuple(v1), line_width=3)
            pyb_utils.debug_frame_world(0.2, tuple(v2), line_width=3)

    push_controller = fp.PushController(
        speed=PUSH_SPEED,
        kθ=Kθ,
        ky=KY,
        path=path,
        con_inc=CON_INC,
        obstacles=obstacles,
        force_min=FORCE_MIN_THRESHOLD,
        min_dist=EE_OBS_MIN_DIST,
    )
    force_controller = fp.AdmittanceController(
        kf=KF, force_max=FORCE_MAX_THRESHOLD, vel_max=PUSH_SPEED
    )

    data = {
        "slider_type": args.slider,
        "environment": args.environment,
        "duration": DURATION,
        "sim_freq": SIM_FREQ,
        "ctrl_freq": CTRL_FREQ,
        "ctrl_step": CTRL_STEP,
        "push_speed": PUSH_SPEED,
        "kθ": Kθ,
        "ky": KY,
        "kf": KF,
        "con_inc": CON_INC,
        "force_min": FORCE_MIN_THRESHOLD,
        "force_max": FORCE_MAX_THRESHOLD,
        "ee_obs_min_dist": EE_OBS_MIN_DIST,
        "inertias": slider_inertias,
        "I_mask": I_mask,
        "y0s": y0s,
        "θ0s": θ0s,
        "s0s": s0s,
        "μ0s": μ0s,
        "path": path,
        "obstacles": obstacles,
        "slider": slider,
        "slider_stiffness": float(args.stiffness),
        "slider_damping": float(args.damping),
    }

    num_sims = len(slider_inertias) * len(y0s) * len(θ0s) * len(s0s) * len(μ0s)

    all_ts = []
    all_r_pw_ws = []
    all_r_sw_ws = []
    all_Q_wss = []
    all_v_cmds = []
    successes = []
    all_forces = []
    all_in_contacts = []
    parameters = []

    count = 0
    with tqdm.tqdm(total=num_sims, initial=START_AT_TRIAL) as progress:
        for (I, μ0, y0, θ0, s0) in itertools.product(
            slider_inertias, μ0s, y0s, θ0s, s0s
        ):
            # for debugging purposes, it may be useful to fast-forward to a
            # particular trial number
            if count < START_AT_TRIAL:
                count += 1
                continue

            # set the new parameters
            r_sw_w = SLIDER_INIT_POS + [0, y0, 0]
            C_ws = rotz(θ0)

            # calculate position of the contact point
            if args.slider == "box":
                r_cs_s = np.array([-BOX_SLIDER_HALF_EXTENTS[0], s0, 0])
                r_cw_w = r_sw_w + C_ws @ r_cs_s
            else:
                r_cs_s = np.array([-CIRCLE_SLIDER_RADIUS, 0, 0])
                angle = -s0 / CIRCLE_SLIDER_RADIUS
                r_cw_w = r_sw_w + rotz(angle) @ r_cs_s

            # pusher starts some ways back from the contact point
            r_pw_w = r_cw_w + PUSHER_INIT_REL_POS
            Q_ws = r2q(C_ws, order="xyzs")
            pusher.set_contact_friction(μ0)
            slider.set_inertia_diagonal(I)

            # reset everything to initial states
            pusher.reset(position=r_pw_w)
            slider.reset(position=r_sw_w, orientation=Q_ws)
            push_controller.reset()
            # force controller has no state, so nothing to reset
            sim.step()

            # run the sim
            (
                ts,
                r_pw_ws,
                r_sw_ws,
                Q_wss,
                v_cmds,
                success,
                forces,
                in_contacts,
            ) = simulate(sim, pusher, slider, push_controller, force_controller, blocks)
            if not success:
                print(f"Trial {count} failed.")
                print(f"I = {np.diag(I)}\nμ = {μ0}\ny0 = {y0}\nθ0 = {θ0}\ns0 = {s0}")
                IPython.embed()

            all_ts.append(ts)
            all_r_pw_ws.append(r_pw_ws)
            all_r_sw_ws.append(r_sw_ws)
            all_Q_wss.append(Q_wss)
            all_v_cmds.append(v_cmds)
            successes.append(success)
            all_forces.append(forces)
            all_in_contacts.append(in_contacts)
            parameters.append([I, μ0, y0, θ0, s0])

            progress.update(1)
            count += 1

    # parse path points to plot
    d = path.segments[-1].direction
    v = path.segments[-1].v2
    dist = np.max((np.vstack(all_r_sw_ws)[:, :2] - v) @ d)
    dist = max(0, dist)

    r_dw_ws = path.get_plotting_coords(dist=dist)

    data["times"] = all_ts
    data["pusher_positions"] = all_r_pw_ws
    data["slider_positions"] = all_r_sw_ws
    data["slider_orientations"] = all_Q_wss
    data["pusher_velocities"] = all_v_cmds
    data["successes"] = successes
    data["path_positions"] = r_dw_ws
    data["forces"] = all_forces
    data["in_contacts"] = all_in_contacts
    data["parameters"] = parameters

    if args.save is not None:
        with open(args.save, "wb") as f:
            pickle.dump(data, f)
        print(f"Saved processed data to {args.save}")

    fp.plot_simulation_results(data)
    plt.show()


if __name__ == "__main__":
    main()
