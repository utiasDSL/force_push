#!/usr/bin/env python3
"""Simulation demonstrating QP-based robot controller."""
import argparse
from pathlib import Path
import time
import yaml

import numpy as np
import matplotlib.pyplot as plt
import pybullet as pyb
import pyb_utils
from spatialmath.base import rotz
from xacrodoc import XacroDoc

import mobile_manipulation_central as mm
import force_push as fp

import IPython


STOP_AT_TIME = None

TIMESTEP = 0.01
DURATION = 10

CONTACT_MU = 0.5
SURFACE_MU = 0.25

ANGLES = np.pi * np.arange(8) / 4

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

# slider params
SLIDER_MASS = 1.0
BOX_SLIDER_HALF_EXTENTS = (0.5, 0.5, 0.06)
SLIDER_CONTACT_DAMPING = 100
SLIDER_CONTACT_STIFFNESS = 10000


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

    # create the simulation
    sim = mm.BulletSimulation(TIMESTEP)
    pyb.changeDynamics(
        sim.ground_uid,
        -1,
        lateralFriction=SURFACE_MU,
    )

    r_cw_w = np.array([0, 0])

    pusher_xacro_doc = XacroDoc.from_includes(
        ["$(find force_push)/urdf/xacro/sim_pusher.urdf.xacro"]
    )
    with pusher_xacro_doc.temp_urdf_file_path() as pusher_urdf_path:
        pusher = fp.BulletPusher(
            pusher_urdf_path,
            [0, 0, 0],
            mu=CONTACT_MU,
        )

    slider = fp.BulletSquareSlider(
        position=[0.5, 0, 0.06],
        mass=SLIDER_MASS,
        half_extents=BOX_SLIDER_HALF_EXTENTS,
    )
    pyb.changeDynamics(
        slider.uid,
        -1,
        contactDamping=SLIDER_CONTACT_DAMPING,
        contactStiffness=SLIDER_CONTACT_STIFFNESS,
    )

    for angle in ANGLES:
        path = fp.SegmentPath.line(fp.rot2d(angle) @ [1, 0])
        push_controller = fp.PushController(
            speed=PUSH_SPEED,
            kθ=Kθ,
            ky=KY,
            path=path,
            con_inc=CON_INC,
            force_min=FORCE_MIN_THRESHOLD,
        )

        r_cw_w0 = np.array([0, 0])

        C_ws0 = rotz(angle)
        offset = np.array(
            [BOX_SLIDER_HALF_EXTENTS[0] + 0.5, 0, BOX_SLIDER_HALF_EXTENTS[2]]
        )
        r_sw_w0 = C_ws0 @ offset
        Q_ws0 = pyb_utils.matrix_to_quaternion(C_ws0)

        pusher.reset(position=r_cw_w0)
        slider.reset(position=r_sw_w0, orientation=Q_ws0)
        sim.step()

        t = 0
        while t <= DURATION:
            r_cw_w, v = pusher.get_joint_states()

            info = path.compute_closest_point_info(r_cw_w)
            pathdir, offset = info.direction, info.offset
            # f = fp.get_contact_force(robot.uid, slider.uid, robot.tool_idx, -1)
            f_old = pusher.get_contact_force([slider.uid], max_contacts=2)
            f = pyb_utils.get_total_contact_wrench(
                slider.uid, pusher.uid, -1, pusher.tool_idx, max_contacts=1
            )[0]
            assert np.allclose(f_old, f)
            f = f[:2]
            θd = np.arctan2(pathdir[1], pathdir[0])

            if open_loop:
                θp = θd - KY * offset
                v_ee_cmd = PUSH_SPEED * fp.rot2d(θp) @ [1, 0]
            else:
                v_ee_cmd = push_controller.update(position=r_cw_w, force=f)

            pusher.command_velocity(v_ee_cmd)
            # slider.apply_wrench(force=[3, 0, 0])

            if STOP_AT_TIME is not None and t > STOP_AT_TIME:
                # pts = pyb_utils.getContactPoints(pusher.uid)
                f_con1 = fp.get_contact_force(
                    slider.uid, sim.ground_uid, max_contacts=4
                )
                f_con2, _ = pyb_utils.get_total_contact_wrench(
                    slider.uid, sim.ground_uid
                )
                IPython.embed()
                return

            t = sim.step(t)


main()
