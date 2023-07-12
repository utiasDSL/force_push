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

        self.pos_init = np.copy(position)
        self.orn_init = np.copy(orientation)

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

    def set_inertia_diagonal(self, I):
        # take the inertia diagonal
        if len(I.shape) > 1:
            I = np.diag(I)
        assert len(I) == 3
        pyb.changeDynamics(self.uid, -1, localInertiaDiagonal=list(I))

    def reset(self, position=None, orientation=None):
        """Reset the body to initial pose and zero velocity."""
        if position is not None:
            self.pos_init = position
        if orientation is not None:
            self.orn_init = orientation

        pyb.resetBaseVelocity(
            self.uid, linearVelocity=[0, 0, 0], angularVelocity=[0, 0, 0]
        )
        pyb.resetBasePositionAndOrientation(
            self.uid, posObj=list(self.pos_init), ornObj=list(self.orn_init)
        )


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
        super().__init__(position, collision_uid, visual_uid, mass=0, mu=mu)

        self.v_last = np.zeros(3)
        self.v_err_int = np.zeros(3)

    def get_contact_force(self, uids):
        """Return contact force, expressed in the world frame."""
        force = np.zeros(3)
        for uid in uids:
            pts = pyb.getContactPoints(self.uid, uid, -1, -1)
            assert len(pts) <= 1
            if len(pts) == 0:
                continue

            pt = pts[0]
            # pos = np.array(pt[5])
            normal = -np.array(pt[7])
            nf = pt[9] * normal

            # TODO not sure if friction directions should also be negated to get
            # the force applied by pusher on slider
            ff1 = -pt[10] * np.array(pt[11])
            ff2 = -pt[12] * np.array(pt[13])
            force += nf + ff1 + ff2
        return force

    def control_velocity(self, vd):
        # TODO thus far unsuccessful attempts at force control (which would
        # allow the pusher to interact with the walls)
        B = np.diag([200, 200, 1])
        k = 1
        v, _ = pyb.getBaseVelocity(self.uid)
        p = self.get_position()

        # a = 1 * (np.array(v) - self.v_last)

        v_err = vd - v
        self.v_err_int += 0.001 * v_err

        f = B @ v_err + 0.01 * B @ self.v_err_int
        f[2] += k * (self.pos_init[2] - p[2]) + 9.81
        print(f)
        pyb.applyExternalForce(
            self.uid, -1, forceObj=list(f), posObj=[0, 0, 0], flags=pyb.LINK_FRAME
        )

        self.v_last = v


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


class BulletPillar(BulletBody):
    def __init__(self, position, radius, height=1.0, mu=1.0):
        collision_uid = pyb.createCollisionShape(
            shapeType=pyb.GEOM_CYLINDER,
            radius=radius,
            height=height,
        )
        visual_uid = pyb.createVisualShape(
            shapeType=pyb.GEOM_CYLINDER,
            radius=radius,
            length=height,
            rgbaColor=[0, 1, 0, 1],
        )
        super().__init__(position, collision_uid, visual_uid, mu=mu, orientation=None)
