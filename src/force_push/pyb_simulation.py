"""PyBullet simulation code for the pusher-slider system."""
import numpy as np
import pybullet as pyb


class BulletBody:
    """Generic rigid body in PyBullet."""

    def __init__(
        self, position, collision_uid, visual_uid, mass=0, mu=1.0, orientation=None
    ):
        if orientation is None:
            orientation = (0, 0, 0, 1)

        self.uid = pyb.createMultiBody(
            baseMass=mass,
            baseCollisionShapeIndex=collision_uid,
            baseVisualShapeIndex=visual_uid,
            basePosition=tuple(position),
            baseOrientation=tuple(orientation),
        )
        pyb.changeDynamics(self.uid, -1, lateralFriction=mu)

    def get_position(self):
        pos, _ = pyb.getBasePositionAndOrientation(self.uid)
        return np.array(pos)

    def command_velocity(self, v):
        assert len(v) == 3
        pyb.resetBaseVelocity(self.uid, linearVelocity=list(v))


class BulletPusher(BulletBody):
    def __init__(self, position, mu, radius=0.05):
        collision_uid = pyb.createCollisionShape(
            shapeType=pyb.GEOM_SPHERE,
            radius=radius,
        )
        visual_uid = pyb.createVisualShape(
            shapeType=pyb.GEOM_SPHERE,
            radius=radius,
            rgbaColor=[1, 0, 0, 1],
        )
        super().__init__(position, collision_uid, visual_uid, mu=mu)

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


class BulletSquareSlider(BulletBody):
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


class BulletCircleSlider(BulletBody):
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


class BulletBlock(BulletBody):
    def __init__(self, position, half_extents, mu=1.0, orientation=None):
        collision_uid = pyb.createCollisionShape(
            shapeType=pyb.GEOM_BOX,
            halfExtents=tuple(half_extents),
        )
        visual_uid = pyb.createVisualShape(
            shapeType=pyb.GEOM_BOX,
            halfExtents=tuple(half_extents),
            rgbaColor=[0, 1, 0, 1],
        )
        super().__init__(
            position, collision_uid, visual_uid, mu=mu, orientation=orientation
        )
