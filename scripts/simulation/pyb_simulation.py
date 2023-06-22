import time

import pybullet as pyb
import pybullet_data
import numpy as np

import mobile_manipulation_central as mm
import force_push as fp

import IPython


SIM_TIMESTEP = 0.01

CONTACT_MU = 0.5
SURFACE_MU = 0.5
PUSH_SPEED = 0.1


class Pusher:
    def __init__(self, position, radius=0.05):

        # create the object
        collision_uid = pyb.createCollisionShape(
            shapeType=pyb.GEOM_SPHERE,
            radius=radius,
        )
        visual_uid = pyb.createVisualShape(
            shapeType=pyb.GEOM_SPHERE,
            radius=radius,
            rgbaColor=[1, 0, 0, 1],
        )
        self.uid = pyb.createMultiBody(
            baseMass=0,
            baseCollisionShapeIndex=collision_uid,
            baseVisualShapeIndex=visual_uid,
            basePosition=tuple(position),
        )

        pyb.changeDynamics(self.uid, -1, lateralFriction=CONTACT_MU)

    def get_position(self):
        pos, _ = pyb.getBasePositionAndOrientation(self.uid)
        return np.array(pos)

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

    def command_velocity(self, v):
        assert len(v) == 3
        pyb.resetBaseVelocity(self.uid, linearVelocity=list(v))


class Slider:
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
        self.uid = pyb.createMultiBody(
            baseMass=mass,
            baseCollisionShapeIndex=collision_uid,
            baseVisualShapeIndex=visual_uid,
            basePosition=tuple(position),
        )

        pyb.changeDynamics(self.uid, -1, lateralFriction=1.)


def main():
    sim = mm.BulletSimulation(SIM_TIMESTEP)
    pyb.changeDynamics(sim.ground_id, -1, lateralFriction=SURFACE_MU)

    pusher = Pusher([0, 0, 0.1])
    slider = Slider([0.7, 0.4, 0.1])

    # desired path
    path = fp.SegmentPath.line(direction=[1, 0])

    # control gains
    kθ = 0.3
    ky = 0.1

    t = 0
    while True:
        force = pusher.get_contact_force(slider)
        r_cw_w = pusher.get_position()

        # angle-based control law
        Δ, yc = path.compute_direction_and_offset(r_cw_w[:2], lookahead=0)
        θy = ky * yc
        θd = fp.signed_angle(Δ, fp.unit(force[:2]))
        θp = (1 + kθ) * θd + ky * yc
        vp_w = PUSH_SPEED * fp.rot2d(θp) @ Δ

        pusher.command_velocity(np.append(vp_w, 0))

        t = sim.step(t)
        time.sleep(SIM_TIMESTEP)


if __name__ == "__main__":
    main()
