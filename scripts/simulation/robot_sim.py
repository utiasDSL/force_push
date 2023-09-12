#!/usr/bin/env python3
"""Simulation demonstrating QP-based robot controller."""
import argparse
from pathlib import Path
import time
import yaml

import rospkg
import numpy as np
import matplotlib.pyplot as plt
import pybullet as pyb
import pyb_utils

import mobile_manipulation_central as mm
import force_push as fp

import IPython


TIMESTEP = 0.01
TOOL_JOINT_NAME = "contact_ball_joint"
DURATION = 100

CONTACT_MU = 0.5
SURFACE_MU = 0.25
OBSTACLE_MU = 0.25

PUSH_SPEED = 0.1

# control gains
Kθ = 0.3
KY = 0.3
Kω = 1
Kf = 0.005  # N / (m/s)
CON_INC = 0.1
DIV_INC = 0.1

# only control based on force when it is high enough (i.e. in contact with
# something)
FORCE_MIN_THRESHOLD = 1
FORCE_MAX_THRESHOLD = 50

# base velocity bounds
VEL_UB = np.array([0.5, 0.5, 0.25])
VEL_LB = -VEL_UB

# slider params
SLIDER_MASS = 1.0
BOX_SLIDER_HALF_EXTENTS = (0.5, 0.5, 0.2)
CIRCLE_SLIDER_RADIUS = 0.5
CIRCLE_SLIDER_HEIGHT = 0.4
SLIDER_CONTACT_DAMPING = 100
SLIDER_CONTACT_STIFFNESS = 10000

# minimum obstacle distance
OBS_MIN_DIST = 0.75

# obstacle starts to influence at this distance
OBS_INFL_DIST = 1.5


def make_urdf_file():
    rospack = rospkg.RosPack()
    path = Path(rospack.get_path("force_push")) / "urdf/urdf/thing_pyb_pusher.urdf"
    if not path.parent.exists():
        path.parent.mkdir()

    includes = [
        "$(find mobile_manipulation_central)/urdf/xacro/thing_pyb.urdf.xacro",
        "$(find force_push)/urdf/xacro/contact_ball.urdf.xacro",
    ]
    mm.XacroDoc.from_includes(includes).to_urdf_file(path)
    return path.as_posix()


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

    # load initial joint configuration
    home = mm.load_home_position(name="pushing_corner", path=fp.HOME_CONFIG_FILE)

    # create the simulation
    urdf_path = make_urdf_file()
    sim = mm.BulletSimulation(TIMESTEP)
    robot = mm.BulletSimulatedRobot(urdf_path, TOOL_JOINT_NAME)
    robot.reset_joint_configuration(home)

    # load calibrated offset between contact point and base frame origin
    with open(fp.CONTACT_POINT_CALIBRATION_FILE) as f:
        r_bc_b = np.array(yaml.safe_load(f)["r_bc_b"])

    # initial contact position
    r_bw_w = home[:2]
    C_wb = fp.rot2d(home[2])
    r_cw_w = r_bw_w - C_wb @ r_bc_b

    # desired EE path
    path = fp.SegmentPath(
        [
            fp.LineSegment([0.0, 0], [0.0, 1]),
            fp.QuadBezierSegment([0.0, 1], [0.0, 3], [-2.0, 3]),
            fp.LineSegment([-2.0, 3], [-3.0, 3], infinite=True),
        ],
        origin=r_cw_w,
    )
    obstacles = fp.translate_segments([fp.LineSegment([-3.0, 3.5], [3.0, 3.5])], r_cw_w)
    block1 = fp.BulletBlock(
        np.append(r_cw_w, 0) + [0, 4, 0.5], [3, 0.5, 0.5], mu=OBSTACLE_MU
    )

    slider = fp.BulletSquareSlider(
        np.append(r_cw_w, 0) + [0, 0.6, 0.2],
        mass=SLIDER_MASS,
        half_extents=BOX_SLIDER_HALF_EXTENTS,
    )

    # set friction and contact properties
    pyb.changeDynamics(sim.ground_uid, -1, lateralFriction=SURFACE_MU)
    pyb.changeDynamics(robot.uid, robot.tool_idx, lateralFriction=CONTACT_MU)
    pyb.changeDynamics(
        slider.uid,
        -1,
        contactDamping=SLIDER_CONTACT_DAMPING,
        contactStiffness=SLIDER_CONTACT_STIFFNESS,
    )

    # controllers
    robot_controller = fp.RobotController(
        -r_bc_b,
        lb=VEL_LB,
        ub=VEL_UB,
        vel_weight=1,
        acc_weight=0,
        obstacles=obstacles,
        min_dist=OBS_MIN_DIST,
    )
    push_controller = fp.PushController(
        speed=PUSH_SPEED,
        kθ=Kθ,
        ky=KY,
        path=path,
        con_inc=CON_INC,
        div_inc=DIV_INC,
        force_min=FORCE_MIN_THRESHOLD,
        force_max=np.inf,
    )

    # admittance control to comply with large forces
    force_controller = fp.AdmittanceController(kf=Kf, force_max=FORCE_MAX_THRESHOLD)

    for obstacle in obstacles:
        pyb_utils.debug_frame_world(0.2, list(obstacle.v1) + [0.1], line_width=3)
        pyb_utils.debug_frame_world(0.2, list(obstacle.v2) + [0.1], line_width=3)

    cmd_vel = np.zeros(3)

    ts = []
    qs = []
    r_cw_ws = []
    r_sw_ws = []
    cmd_vels = []
    forces = []

    t = 0
    while t <= DURATION:
        q, _ = robot.joint_states()
        r_bw_w = q[:2]
        C_wb = fp.rot2d(q[2])
        r_cw_w = r_bw_w - C_wb @ r_bc_b

        pathdir, _ = path.compute_direction_and_offset(r_cw_w)
        f = fp.get_contact_force(robot.uid, slider.uid)[:2]

        if open_loop:
            v_ee_cmd = PUSH_SPEED * pathdir
        else:
            v_ee_cmd = push_controller.update(r_cw_w, f)
            v_ee_cmd = force_controller.update(force=f, v_cmd=v_ee_cmd)

        θd = np.arctan2(pathdir[1], pathdir[0])
        ωd = Kω * fp.wrap_to_pi(θd - q[2])
        V_ee_cmd = np.append(v_ee_cmd, ωd)

        # move the base so that the desired EE velocity is achieved
        cmd_vel = robot_controller.update(r_bw_w, C_wb, V_ee_cmd, u_last=cmd_vel)
        if cmd_vel is None:
            print("Failed to solve QP!")
            break

        # use P control on the arm joints to keep them in place
        u = np.concatenate((cmd_vel, 10 * (home[3:] - q[3:])))

        # record data
        ts.append(t)
        qs.append(q[:3])
        r_cw_ws.append(r_cw_w)
        r_sw_ws.append(slider.get_pose()[0])
        forces.append(f)
        cmd_vels.append(cmd_vel)

        # note that in simulation the mobile base takes commands in the world
        # frame, but the real mobile base takes commands in the body frame
        # (this is just an easy 2D rotation away)
        robot.command_velocity(u)

        # step the sim forward in time
        t = sim.step(t)
        # time.sleep(TIMESTEP)

    ts = np.array(ts)
    qs = np.array(qs)
    r_cw_ws = np.array(r_cw_ws)
    r_sw_ws = np.array(r_sw_ws)
    forces = np.array(forces)
    cmd_vels = np.array(cmd_vels)
    path_xy = path.get_plotting_coords()

    plt.figure()
    plt.plot(ts, qs[:, 0], label="x")
    plt.plot(ts, qs[:, 1], label="y")
    plt.plot(ts, qs[:, 2], label="θ")
    plt.legend()
    plt.grid()
    plt.xlabel("Time [s]")
    plt.title("Base position vs. time")

    plt.figure()
    plt.plot(path_xy[:, 0], path_xy[:, 1], "--", color="k", label="Desired")
    plt.plot(qs[:, 0], qs[:, 1], label="Base")
    plt.plot(r_cw_ws[:, 0], r_cw_ws[:, 1], label="Contact")
    plt.plot(r_sw_ws[:, 0], r_sw_ws[:, 1], label="Slider")
    plt.legend()
    plt.grid()
    plt.xlabel("x [m]")
    plt.ylabel("y [m]")
    plt.title("Paths")

    plt.figure()
    plt.plot(ts, forces[:, 0], label="x")
    plt.plot(ts, forces[:, 1], label="y")
    plt.plot(ts, np.linalg.norm(forces, axis=1), label="Magnitude")
    plt.legend()
    plt.grid()
    plt.xlabel("Time [s]")
    plt.title("Contact force vs. time")

    plt.figure()
    plt.plot(ts, cmd_vels[:, 0], label="x")
    plt.plot(ts, cmd_vels[:, 1], label="y")
    plt.plot(ts, cmd_vels[:, 2], label="θ")
    plt.legend()
    plt.grid()
    plt.xlabel("Time [s]")
    plt.title("Base velocity commands vs. time")

    plt.show()


main()
