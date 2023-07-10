import numpy as np
from force_push import util


def circle_r_tau(radius):
    """r_tau for a circular support area with uniform friction."""
    return 2.0 * radius / 3


def _alpha_rect(w, h):
    # alpha_rect for half of the rectangle
    d = np.sqrt(h * h + w * w)
    return (w * h * d + w * w * w * (np.log(h + d) - np.log(w))) / 12.0


def rectangle_r_tau(width, height):
    """r_tau for a rectangular support area with uniform friction."""
    # see pushing notes
    area = width * height
    return (_alpha_rect(width, height) + _alpha_rect(height, width)) / area


def uniform_cuboid_inertia(mass, half_lengths):
    """Inertia matrix of a cuboid with given half lengths."""
    lx, ly, lz = 2 * np.array(half_lengths)
    xx = ly**2 + lz**2
    yy = lx**2 + lz**2
    zz = lx**2 + ly**2
    return mass * np.diag([xx, yy, zz]) / 12.0


def point_mass_system_inertia(masses, points):
    """Inertia matrix corresponding to a finite set of point masses."""
    I = np.zeros((3, 3))
    for m, p in zip(masses, points):
        P = util.skew3d(p)
        I += -m * P @ P
    return I


def thick_walled_cylinder_inertia(mass, r1, r2, h):
    """Inertia matrix of a thick-walled cylinder of height h. The inner wall
    has radius r1 and the outer wall has radius r2 >= r1.

    The cylinder is oriented along the z-axis.
    """
    assert r2 >= r1
    p2 = r1**2 + r2**2
    h2 = h**2
    return mass * np.diag([(3 * p2 + h2) / 12, (3 * p2 + h2) / 12, p2 / 2])


def thin_walled_cylinder_inertia(mass, r, h):
    """Inertia matrix of a cylinder of radius r and height h with a wall of
    negligible thickness.

    In other words, all mass is concentrated at radius r from the central axis.

    The cylinder is oriented along the z-axis.
    """
    return thick_walled_cylinder_inertia(mass, r, r, h)


def uniform_cylinder_inertia(mass, r, h):
    """Inertia matrix of a uniform density cylinder of radius r and height h.

    The cylinder is oriented along the z-axis.
    """
    return thick_walled_cylinder_inertia(mass, 0, r, h)
