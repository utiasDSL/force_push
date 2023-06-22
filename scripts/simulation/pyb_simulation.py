import time

import pybullet as pyb
import pybullet_data
import numpy as np
import matplotlib.pyplot as plt
from pyb_utils.frame import debug_frame_world

import mobile_manipulation_central as mm
import force_push as fp

import IPython


# Hz
SIM_FREQ = 1000
CTRL_FREQ = 100

DURATION = 60

CONTACT_MU = 0.2
SURFACE_MU = 1.0

# controller params
PUSH_SPEED = 0.5
Kθ = 0.5
KY = 0.1
LOOKAHEAD = 2.0


class BulletBody:
    def __init__(self, position, collision_uid, visual_uid, mass=0, orientation=None):
        if orientation is None:
            orientation = (0, 0, 0, 1)

        self.uid = pyb.createMultiBody(
            baseMass=mass,
            baseCollisionShapeIndex=collision_uid,
            baseVisualShapeIndex=visual_uid,
            basePosition=tuple(position),
            baseOrientation=tuple(orientation),
        )

    def get_position(self):
        pos, _ = pyb.getBasePositionAndOrientation(self.uid)
        return np.array(pos)

    def command_velocity(self, v):
        assert len(v) == 3
        pyb.resetBaseVelocity(self.uid, linearVelocity=list(v))


class Pusher(BulletBody):
    def __init__(self, position, radius=0.05):
        collision_uid = pyb.createCollisionShape(
            shapeType=pyb.GEOM_SPHERE,
            radius=radius,
        )
        visual_uid = pyb.createVisualShape(
            shapeType=pyb.GEOM_SPHERE,
            radius=radius,
            rgbaColor=[1, 0, 0, 1],
        )
        super().__init__(position, collision_uid, visual_uid)

        pyb.changeDynamics(self.uid, -1, lateralFriction=CONTACT_MU)

    def get_contact_force(self, slider):
        """Return contact force, expressed in the world frame."""
        pts = pyb.getContactPoints(self.uid, slider.uid, -1, -1)
        assert len(pts) <= 1
        if len(pts) == 0:
            return np.zeros(3)

        pt = pts[0]
        # pos = np.array(pt[5])
        normal = -np.array(pt[7])
        nf = pt[9] * normal

        # TODO not sure if friction directions should also be negated to get
        # the force applied by pusher on slider
        ff1 = -pt[10] * np.array(pt[11])
        ff2 = -pt[12] * np.array(pt[13])
        force = nf + ff1 + ff2
        return force


class SquareSlider(BulletBody):
    def __init__(self, position, mass=1, half_extents=(0.5, 0.5, 0.1)):
        collision_uid = pyb.createCollisionShape(
            shapeType=pyb.GEOM_BOX,
            halfExtents=tuple(half_extents),
        )
        visual_uid = pyb.createVisualShape(
            shapeType=pyb.GEOM_BOX,
            halfExtents=tuple(half_extents),
            rgbaColor=[0, 0, 1, 1],
        )
        super().__init__(position, collision_uid, visual_uid, mass=mass)
        pyb.changeDynamics(self.uid, -1, lateralFriction=1.0)


class CircleSlider(BulletBody):
    def __init__(self, position, mass=1, radius=0.5, height=0.2):
        collision_uid = pyb.createCollisionShape(
            shapeType=pyb.GEOM_CYLINDER,
            radius=radius,
            height=height,
        )
        visual_uid = pyb.createVisualShape(
            shapeType=pyb.GEOM_CYLINDER,
            radius=radius,
            length=height,
            rgbaColor=[0, 0, 1, 1],
        )
        super().__init__(position, collision_uid, visual_uid, mass=mass)
        pyb.changeDynamics(self.uid, -1, lateralFriction=1.0)


def main():
    sim = mm.BulletSimulation(1.0 / SIM_FREQ)
    pyb.changeDynamics(sim.ground_uid, -1, lateralFriction=SURFACE_MU)

    pusher = Pusher([0, 0, 0.1])
    slider = SquareSlider([0.7, 0.4, 0.1])

    # desired path
    # path = fp.SegmentPath.line(direction=[1, 0])
    vertices = np.array([[0, 0], [5, 0]])
    path = fp.SegmentPath(vertices, final_direction=[0, 1])

    for vertex in path.vertices:
        r = np.append(vertex, 0.1)
        debug_frame_world(0.2, tuple(r), line_width=3)

    controller = fp.Controller(
        speed=PUSH_SPEED, kθ=Kθ, ky=KY, path=path, lookahead=LOOKAHEAD
    )

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

        sim.step(t)
        # time.sleep(SIM_TIMESTEP)

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
