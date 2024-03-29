"""PyBullet simulation code for the pusher-slider system."""
import numpy as np
import pybullet as pyb
import pyb_utils


def get_contact_force(uid1, uid2, linkIndexA=-2, linkIndexB=-2, max_contacts=1):
    """Get the contact force between two PyBullet bodies."""
    points = pyb_utils.getContactPoints(uid1, uid2, linkIndexA, linkIndexB)
    assert len(points) <= max_contacts, f"Found {len(points)} contact points."
    if len(points) == 0:
        return np.zeros(3)

    force = np.zeros(3)
    for point in points:
        normal = -np.array(point.contactNormalOnB)
        nf = point.normalForce * normal
        # ff1 = -point.lateralFriction1 * np.array(point.lateralFrictionDir1)
        # ff2 = -point.lateralFriction2 * np.array(point.lateralFrictionDir2)
        ff1 = -point.lateralFriction1 * np.array(point.lateralFrictionDir1)
        ff2 = -point.lateralFriction2 * np.array(point.lateralFrictionDir2)
        # print(f"fric force = {ff1 + ff2}")
        # print(f"norm force = {nf}")
        force += nf + ff1 + ff2
        # print(f"pb = {point.positionOnB}")
    return force


class BulletBody(pyb_utils.BulletBody):
    """Generic rigid body in PyBullet."""

    def __init__(
        self, position, collision_uid, visual_uid, mass=0, mu=1.0, orientation=None
    ):
        if orientation is None:
            orientation = (0, 0, 0, 1)
        self.pos_init = np.copy(position)
        self.orn_init = np.copy(orientation)

        super().__init__(
            position=position,
            collision_uid=collision_uid,
            visual_uid=visual_uid,
            mass=mass,
            orientation=orientation,
        )
        pyb.changeDynamics(self.uid, -1, lateralFriction=mu)

    def set_inertia_diagonal(self, I):
        # take the inertia diagonal
        if I.ndim > 1:
            I = np.diag(I)
        assert I.shape == (3,)
        pyb.changeDynamics(self.uid, -1, localInertiaDiagonal=list(I))

    def set_contact_parameters(self, stiffness=0, damping=0):
        # see e.g. <https://github.com/bulletphysics/bullet3/issues/4428>
        pyb.changeDynamics(
            self.uid, -1, contactDamping=damping, contactStiffness=stiffness
        )

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


# class BulletPusher(BulletBody):
#     """Spherical pusher"""
#
#     def __init__(self, position, mass=100, mu=1, radius=0.1):
#         collision_uid = pyb.createCollisionShape(
#             shapeType=pyb.GEOM_SPHERE,
#             radius=radius,
#         )
#         visual_uid = pyb.createVisualShape(
#             shapeType=pyb.GEOM_SPHERE,
#             radius=radius,
#             rgbaColor=[1, 0, 0, 1],
#         )
#         super().__init__(position, collision_uid, visual_uid, mass=mass, mu=mu)
#
#     def command_velocity(self, v):
#         """Send a linear velocity command."""
#         self.set_velocity(linear=v)
#
#     def get_contact_force(self, uids):
#         """Return contact force, expressed in the world frame."""
#         return sum([get_contact_force(self.uid, uid) for uid in uids])


class BulletPusher(pyb_utils.Robot):
    """Pusher based on a URDF."""

    def __init__(self, urdf_path, position, mu=1):
        # here we use a fixed base so that the "base" just stays at the world
        # frame origin while the link moves via two prismatic joints
        uid = pyb.loadURDF(urdf_path, [0, 0, 0], [0, 0, 0, 1], useFixedBase=True)
        super().__init__(uid)
        assert self.num_total_joints == 2
        assert self.tool_idx == 1

        # zero friction on the floor, variable contact friction
        pyb.changeDynamics(self.uid, -1, lateralFriction=0)
        self.set_contact_friction(mu)
        self.reset(position=position, orientation=[0, 0, 0, 1])

    def set_contact_friction(self, μ):
        """Set the friction for the link in contact with the slider."""
        pyb.changeDynamics(self.uid, self.tool_idx, lateralFriction=μ)

    def get_contact_force(self, uids, max_contacts=1):
        """Return contact force, expressed in the world frame."""
        return sum(
            [
                get_contact_force(
                    self.uid, uid, self.tool_idx, max_contacts=max_contacts
                )
                for uid in uids
            ]
        )

    def reset(self, position=None, orientation=None):
        """Reset the body to initial pose and zero velocity."""
        if position is not None:
            self.pos_init = position
        if orientation is not None:
            self.orn_init = orientation

        self.command_velocity([0, 0])
        self.reset_joint_configuration(self.pos_init)


class BulletSquareSlider(BulletBody):
    """Square slider"""

    def __init__(
        self, position, mass=1, half_extents=(0.5, 0.5, 0.1), orientation=None
    ):
        collision_uid = pyb.createCollisionShape(
            shapeType=pyb.GEOM_BOX,
            halfExtents=tuple(half_extents),
        )
        visual_uid = pyb.createVisualShape(
            shapeType=pyb.GEOM_BOX,
            halfExtents=tuple(half_extents),
            rgbaColor=[0, 0, 1, 1],
        )
        super().__init__(
            position, collision_uid, visual_uid, mass=mass, orientation=orientation
        )


class BulletCircleSlider(BulletBody):
    """Circular slider"""

    def __init__(self, position, mass=1, radius=0.5, height=0.2, orientation=None):
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
        super().__init__(
            position, collision_uid, visual_uid, mass=mass, orientation=orientation
        )


class BulletBlock(BulletBody):
    """Fixed block obstacle."""

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
    """Fixed cylindrical pillar obstacle."""

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
