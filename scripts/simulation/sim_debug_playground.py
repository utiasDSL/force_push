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
from spatialmath.base import rotz

import mobile_manipulation_central as mm
import force_push as fp

import IPython


USE_URDF_SLIDER = False
COMMAND_SLIDER = False
STOP_AT_TIME = 5
USE_BOX_GROUND = False

TIMESTEP = 0.01
TOOL_LINK_NAME = "contact_ball"
DURATION = 50

CONTACT_MU = 0.5
SURFACE_MU = 0.25
OBSTACLE_MU = 0.25

STRAIGHT_ANGLE = np.deg2rad(0)
C_wd = fp.rot2d(STRAIGHT_ANGLE)
C_dw = C_wd.T
STRAIGHT_DIRECTION = C_wd @ np.array([1, 0])

PUSH_SPEED = 0.1

# control gains
Kθ = 0.3
KY = 0.1
Kω = 1
Kf = 0.003  # N / (m/s)
CON_INC = 0.1

# only control based on force when it is high enough (i.e. in contact with
# something)
FORCE_MIN_THRESHOLD = 1
FORCE_MAX_THRESHOLD = 50

# base velocity bounds
VEL_UB = np.array([0.5, 0.5, 0.25])
VEL_LB = -VEL_UB

# slider params
SLIDER_MASS = 1.0
# BOX_SLIDER_HALF_EXTENTS = (0.5, 0.5, 0.2)
BOX_SLIDER_HALF_EXTENTS = (0.5, 0.5, 0.06)
CIRCLE_SLIDER_RADIUS = 0.5
CIRCLE_SLIDER_HEIGHT = 0.12
SLIDER_CONTACT_DAMPING = 100
SLIDER_CONTACT_STIFFNESS = 10000
# SLIDER_CONTACT_DAMPING = 100
# SLIDER_CONTACT_STIFFNESS = 1e9

# minimum obstacle distance
OBS_MIN_DIST = 0.75


def box_mesh(position, half_extents, orientation=None):
    x, y, z = half_extents
    vertices = [
        [x, y, z],  # 0
        [x, y, -z],  # 1
        [x, -y, z],  # 2
        [x, -y, -z],  # 3
        [-x, y, z],  # 4
        [-x, y, -z],  # 5
        [-x, -y, z],  # 6
        [-x, -y, -z],  # 7
    ]

    # sets of vertices making up triangular faces
    # counter-clockwise winding about normal facing out of the shape
    # fmt: off
    indices = np.array([
        [0, 2, 1], [2, 3, 1], # +x
        [4, 6, 5], [6, 7, 5], # -x
        [0, 5, 4], [0, 1, 5], # +y
        [2, 7, 6], [2, 3, 7], # -y
        [0, 4, 2], [4, 6, 2], # +z
        [1, 5, 3], [5, 7, 3], # -z
    ])
    # fmt: on
    indices = list(indices.flatten())

    collision_uid = pyb.createCollisionShape(
        pyb.GEOM_MESH, vertices=vertices, indices=indices
    )
    visual_uid = pyb.createVisualShape(
        pyb.GEOM_MESH,
        vertices=vertices,
        indices=indices,
        rgbaColor=(0, 0, 1, 1),
    )
    return pyb_utils.BulletBody(
        position, collision_uid, visual_uid, orientation=orientation, mass=SLIDER_MASS
    )


# TODO make this a general utility
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


def make_pusher_urdf_file():
    rospack = rospkg.RosPack()
    path = Path(rospack.get_path("force_push")) / "urdf/urdf/sim_pusher.urdf"
    if not path.parent.exists():
        path.parent.mkdir()

    includes = ["$(find force_push)/urdf/xacro/sim_pusher.urdf.xacro"]
    mm.XacroDoc.from_includes(includes).to_urdf_file(path)
    return path.as_posix()


# def make_slider_urdf_file():
#     rospack = rospkg.RosPack()
#     path = Path(rospack.get_path("force_push")) / "urdf/urdf/sim_slider.urdf"
#     if not path.parent.exists():
#         path.parent.mkdir()
#
#     includes = ["$(find force_push)/urdf/xacro/sim_slider.urdf.xacro"]
#     mm.XacroDoc.from_includes(includes).to_urdf_file(path)
#     return path.as_posix()


def compile_xacro_urdf_file(name):
    xacro_name = name
    parts = xacro_name.split(".")
    assert parts[-1] == "xacro"
    urdf_name = ".".join(parts[:-1])

    rospack = rospkg.RosPack()
    path = Path(rospack.get_path("force_push")) / "urdf/urdf" / urdf_name
    if not path.parent.exists():
        path.parent.mkdir()

    includes = ["$(find force_push)/urdf/xacro/" + xacro_name]
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
    parser.add_argument(
        "--environment",
        choices=["straight", "corner"],
        help="Which environment to use",
        required=True,
    )
    args = parser.parse_args()
    open_loop = args.open_loop

    # load initial joint configuration
    if args.environment == "straight":
        home = mm.load_home_position(name="pushing_diag", path=fp.HOME_CONFIG_FILE)
    else:
        home = mm.load_home_position(name="pushing_corner", path=fp.HOME_CONFIG_FILE)

    # create the simulation
    urdf_path = make_urdf_file()
    sim = mm.BulletSimulation(TIMESTEP)

    # try replacing the ground plane with a large flat box
    if USE_BOX_GROUND:
        pyb.resetBasePositionAndOrientation(
            sim.ground_uid, posObj=(0, 0, -0.2), ornObj=(0, 0, 0, 1)
        )
        ground = pyb_utils.BulletBody.box(
            position=(0, 0, -0.1), half_extents=(2, 2, 0.1), mass=1
        )
        pyb.changeDynamics(ground.uid, -1, lateralFriction=SURFACE_MU)
        pyb.changeDynamics(sim.ground_uid, -1, lateralFriction=10)
        sim.ground_uid = ground.uid

    # robot_id = pyb.loadURDF(
    #     urdf_path,
    #     [0, 0, 0],
    #     useFixedBase=True,
    # )
    # robot = pyb_utils.Robot(robot_id, tool_link_name=TOOL_LINK_NAME)
    # robot.reset_joint_configuration(home)

    # initial contact position
    # r_bw_w = home[:2]
    # r_cw_w = robot.get_link_frame_pose()[0][:2]
    r_cw_w = np.array([0, 0])
    # C_wb = fp.rot2d(home[2])
    # r_bc_b = -C_wb.T @ (r_cw_w - r_bw_w)

    pusher_urdf_path = make_pusher_urdf_file()
    pusher = fp.BulletPusher(
        pusher_urdf_path,
        np.append(r_cw_w + [0, 0], 0),
        mu=CONTACT_MU,
    )
    # pusher.set_joint_friction_forces([0, 0])
    pyb.setCollisionFilterPair(pusher.uid, sim.ground_uid, pusher.tool_idx, -1, 0)

    # NOTE for visualizing the circular approximation to the base for obstacle
    # avoidance
    # pyb_utils.BulletBody.cylinder([home[0], home[1], 0], radius=0.55, height=0.2)
    # IPython.embed()
    # return

    # desired EE path
    if args.environment == "straight":
        path = fp.SegmentPath.line(STRAIGHT_DIRECTION, origin=r_cw_w)
        C_ws0 = rotz(STRAIGHT_ANGLE)
        r_sw_w0 = np.append(r_cw_w, 0) + C_ws0 @ [0.8, 0, 0.06]
        Q_ws0 = pyb_utils.matrix_to_quaternion(C_ws0)
    else:
        path = fp.SegmentPath(
            [
                fp.LineSegment([0.0, 0.0], [0.0, 2.0]),
                fp.CircularArcSegment(
                    center=[-2.0, 2.0], point=[0.0, 2.0], angle=np.pi / 2
                ),
                fp.LineSegment([-2.0, 4.0], [-4.0, 4.0], infinite=True),
            ],
            origin=r_cw_w,
        )
        r_sw_w0 = np.append(r_cw_w, 0) + [0, 0.8, 0.06]
        Q_ws0 = (0, 0, 0, 1)
    obstacles = []
    # obstacles = fp.translate_segments(
    #     [fp.LineSegment([-3.5, 4.25], [1.0, 4.25])], r_cw_w
    # )
    #
    # block1 = fp.BulletBlock(
    #     np.append(r_cw_w, 0) + [0, 4.25, 0.5], [3, 0.5, 0.5], mu=OBSTACLE_MU
    # )

    if USE_URDF_SLIDER:
        slider_urdf_path = compile_xacro_urdf_file("sim_slider.urdf.xacro")
        slider_uid = pyb.loadURDF(
            slider_urdf_path,
            [0, 0, 0],
            useFixedBase=True,
        )
        slider = pyb_utils.Robot(
            slider_uid,
            tool_link_name="slider_link",
            actuated_joint_names=[
                "slider_joint_x",
                "slider_joint_y",
                "slider_joint_yaw",
            ],
        )
        slider.reset_joint_configuration([r_sw_w0[0], r_sw_w0[1], 0, STRAIGHT_ANGLE])
        slider.set_joint_friction_forces([0, 0, 0, 0])
    else:
        slider = fp.BulletSquareSlider(
            position=r_sw_w0,
            orientation=Q_ws0,
            mass=SLIDER_MASS,
            half_extents=BOX_SLIDER_HALF_EXTENTS,
        )
        # slider = fp.BulletCircleSlider(
        #     position=r_sw_w0,
        #     mass=SLIDER_MASS,
        #     radius=CIRCLE_SLIDER_RADIUS,
        #     height=CIRCLE_SLIDER_HEIGHT,
        # )
        # slider = box_mesh(
        #     position=r_sw_w0, orientation=Q_ws0, half_extents=BOX_SLIDER_HALF_EXTENTS
        # )
        # I_diag = np.diag(
        #     fp.uniform_cuboid_inertia(
        #         mass=SLIDER_MASS, half_extents=BOX_SLIDER_HALF_EXTENTS
        #     )
        # )
        # pyb.changeDynamics(slider.uid, -1, localInertiaDiagonal=list(I_diag))

    # set friction and contact properties
    pyb.changeDynamics(
        sim.ground_uid,
        -1,
        lateralFriction=SURFACE_MU,
        spinningFriction=0,
        rollingFriction=0,
        # contactDamping=SLIDER_CONTACT_DAMPING,
        # contactStiffness=SLIDER_CONTACT_STIFFNESS,
    )
    # pyb.changeDynamics(robot.uid, robot.tool_idx, lateralFriction=CONTACT_MU)

    # mass = 10
    I_diag = np.array(pyb_utils.getDynamicsInfo(slider.uid, -1).localInertiaDiagonal)
    # pyb.changeDynamics(ground.uid, -1, localInertiaDiagonal=I_diag)
    #
    pyb.changeDynamics(
        slider.uid,
        slider.tool_idx if USE_URDF_SLIDER else -1,
        lateralFriction=1.0,
        # mass=10,
        contactDamping=SLIDER_CONTACT_DAMPING,
        contactStiffness=SLIDER_CONTACT_STIFFNESS,
        # localInertiaDiagonal=list(10*I_diag),
        # collisionMargin=0,
        # linearDamping=1,
        # angularDamping=10,
        # rollingFriction=1,
        spinningFriction=0.0,
        # anisotropicFriction=(1, 1, 0),
    )
    # pyb.enableJointForceTorqueSensor(slider.uid, slider.tool_idx, enableSensor=1)
    # for i in range(4):
    #     link_idx = slider.get_link_index(f"slider_foot{i+1}_link")
    #     pyb.changeDynamics(slider.uid, link_idx, lateralFriction=1.0, spinningFriction=0, rollingFriction=0)

    # link_idx = slider.get_link_index(f"slider_foot1_link")
    # pyb.changeDynamics(
    #     slider.uid,
    #     link_idx,
    #     lateralFriction=1.0,
    #     spinningFriction=0.0,
    #     rollingFriction=0,
    #     contactDamping=SLIDER_CONTACT_DAMPING,
    #     contactStiffness=SLIDER_CONTACT_STIFFNESS,
    # )

    pyb.changeDynamics(
        pusher.uid,
        pusher.tool_idx,
        contactDamping=SLIDER_CONTACT_DAMPING,
        contactStiffness=SLIDER_CONTACT_STIFFNESS,
    )

    # pyb.setPhysicsEngineParameter(
    #     enableConeFriction=1, numSolverIterations=500,
    # )

    # pyb.changeDynamics(pusher.uid, pusher.tool_idx, collisionMargin=0)

    # controllers
    # robot_controller = fp.RobotController(
    #     -r_bc_b,
    #     lb=VEL_LB,
    #     ub=VEL_UB,
    #     vel_weight=1,
    #     acc_weight=0,
    #     obstacles=obstacles,
    #     min_dist=OBS_MIN_DIST,
    # )
    push_controller = fp.PushController(
        speed=PUSH_SPEED,
        kθ=Kθ,
        ky=KY,
        path=path,
        con_inc=CON_INC,
        force_min=FORCE_MIN_THRESHOLD,
    )

    # admittance control to comply with large forces
    force_controller = fp.AdmittanceController(kf=Kf, force_max=FORCE_MAX_THRESHOLD)

    smoother = mm.ExponentialSmoother(τ=0.05, x0=np.zeros(3))

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
        # q, _ = robot.get_joint_states()
        # r_bw_w = q[:2]
        # C_wb = fp.rot2d(q[2])
        # r_cw_w = robot.get_link_frame_pose()[0][:2]
        r_cw_w, v = pusher.get_joint_states()
        # print(f"v = {v}")

        # pts = pyb_utils.getContactPoints(slider.uid, sim.ground_uid)
        # if t > 5:
        #     IPython.embed()
        #     return

        info = path.compute_closest_point_info(r_cw_w)
        pathdir, offset = info.direction, info.offset
        # f = fp.get_contact_force(robot.uid, slider.uid, robot.tool_idx, -1)
        f = pusher.get_contact_force([slider.uid], max_contacts=2)
        # f = smoother.update(f, dt=sim.timestep)
        # f_b = rotz(q[2]).T @ f
        # print(f_b)
        f = f[:2]
        print(f"f sensed = {f}")
        θd = np.arctan2(pathdir[1], pathdir[0])

        # print(f"pathdir = {pathdir}")
        # print(f"θd = {np.rad2deg(θd)}")
        # print(np.arccos(fp.unit(f) @ pathdir))

        if open_loop:
            θp = θd - KY * offset
            v_ee_cmd = PUSH_SPEED * fp.rot2d(θp) @ [1, 0]
        else:
            v_ee_cmd = push_controller.update(position=r_cw_w, force=f)
            # v_ee_cmd = force_controller.update(force=f, v_cmd=v_ee_cmd)
        # print(f"v_cmd = {v_ee_cmd}")

        # ωd = Kω * fp.wrap_to_pi(θd - q[2])
        # V_ee_cmd = np.append(v_ee_cmd, ωd)

        # move the base so that the desired EE velocity is achieved
        # cmd_vel = robot_controller.update(r_bw_w, C_wb, V_ee_cmd, u_last=cmd_vel)
        # if cmd_vel is None:
        #     print("Failed to solve QP!")
        #     break

        # cmd_vel = np.append(v_ee_cmd, 0)
        # cmd_vel = v_ee_cmd

        # use P control on the arm joints to keep them in place
        # u = np.concatenate((cmd_vel, 10 * (home[3:] - q[3:])))
        # u = np.concatenate((cmd_vel, np.zeros(6)))

        # record data
        ts.append(t)
        # qs.append(q[:3])
        r_cw_ws.append(r_cw_w)
        if USE_URDF_SLIDER:
            r_sw_ws.append(slider.get_link_frame_pose()[0])
        else:
            r_sw_ws.append(slider.get_pose()[0])
        forces.append(f)
        cmd_vels.append(v_ee_cmd)

        # note that in simulation the mobile base takes commands in the world
        # frame, but the real mobile base takes commands in the body frame
        # (this is just an easy 2D rotation away)
        # robot.command_velocity(u)
        if not COMMAND_SLIDER:
            pusher.command_velocity(v_ee_cmd)

        # vs, ωs = slider.get_velocity()
        # if np.linalg.norm(vs[:2]) > 0:
        #     nf = 9.81 * SLIDER_MASS
        #     μ = 0.25
        #     ff = np.append(-0.25 * nf * fp.unit(vs[:2]), 0)
        #     slider.apply_wrench(force=ff, frame=pyb.WORLD_FRAME)
        #     print(f"ff = {np.linalg.norm(ff)}")
        # if np.abs(ωs[2]) > 0.0:
        #     # TODO
        #     τz = -10 * ωs[2]
        #     if np.abs(τz) > 1:
        #         τz = np.sign(τz) * 1
        #     τf = np.array([0, 0, τz])
        #     slider.apply_wrench(torque=τf)
        #     print(f"ωs = {ωs}")
        #     print(f"τf = {τf}")

        f_con = fp.get_contact_force(sim.ground_uid, slider.uid, max_contacts=10)
        # f_con = fp.get_contact_force(slider.uid, ground.uid, max_contacts=4)
        # f_con = fp.get_contact_force(pusher.uid, slider.uid, pusher.tool_idx, -1, max_contacts=4)
        # print(f_con)
        print(f"friction norm = {np.linalg.norm(f_con[:2])}")

        # vs, ωs = slider.get_velocity()
        # slider.set_velocity(angular=[0, 0, ωs[2]])

        if COMMAND_SLIDER:
            if USE_URDF_SLIDER:
                slider.command_velocity([0.1, 0.1, 0])
                # slider.command_effort([2, 2, 0])
            else:
                slider.set_velocity(linear=[0.1, 0.1, 0])
                # slider.apply_wrench(force=[2.5, 0, 0], frame=pyb.LINK_FRAME)
        pts = pyb_utils.getContactPoints(slider.uid, sim.ground_uid)
        ff_sum = 0
        ff1_sum = 0
        ff2_sum = 0
        ff_total = np.zeros(3)
        τ = np.zeros(3)
        for pt in pts:
            nf = pt.normalForce * np.array(pt.contactNormalOnB)
            ff = pt.lateralFriction1 * np.array(
                pt.lateralFrictionDir1
            ) + pt.lateralFriction2 * np.array(pt.lateralFrictionDir2)
            ff_sum += np.linalg.norm(ff)
            ff1_sum += pt.lateralFriction1
            ff2_sum += pt.lateralFriction2

            # TODO this is wrong
            if USE_URDF_SLIDER:
                posA = slider.get_link_frame_pose()[0]
            else:
                posA = slider.get_pose()[0]
            τ += np.cross(pt.positionOnA - posA, ff)
            ff_total += ff
            print(f"nf = {pt.normalForce}, ff = {ff}")
        print(f"τ = {τ}")
        print(f"ff_total = {ff_total}")

        # pyb.resetBasePositionAndOrientation(sim.ground_uid, posObj=(0, 0, 0), ornObj=slider.get_link_frame_pose()[1])

        if STOP_AT_TIME is not None and t > STOP_AT_TIME:
            # pts = pyb_utils.getContactPoints(pusher.uid)
            IPython.embed()
            return

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

    # plt.figure()
    # plt.plot(ts, qs[:, 0], label="x")
    # plt.plot(ts, qs[:, 1], label="y")
    # plt.plot(ts, qs[:, 2], label="θ")
    # plt.legend()
    # plt.grid()
    # plt.xlabel("Time [s]")
    # plt.title("Base position vs. time")

    plt.figure()
    plt.plot(path_xy[:, 0], path_xy[:, 1], "--", color="k", label="Desired")
    # plt.plot(qs[:, 0], qs[:, 1], label="Base")
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
    # plt.plot(ts, cmd_vels[:, 2], label="θ")
    plt.legend()
    plt.grid()
    plt.xlabel("Time [s]")
    plt.title("Base velocity commands vs. time")

    plt.show()


main()
