import numpy as np
from scipy import sparse
import osqp


def wrap_to_pi(x):
    """Wrap a value to [-π, π]"""
    while x > np.pi:
        x -= 2 * np.pi
    while x < -np.pi:
        x += 2 * np.pi
    return x


def rot2d(θ):
    """2D rotation matrix."""
    return np.array([[np.cos(θ), -np.sin(θ)], [np.sin(θ), np.cos(θ)]])


def signed_angle(a, b):
    """Angle to rotate a to b.

    See <https://stackoverflow.com/a/2150111/5145874>"""
    return wrap_to_pi(np.arctan2(b[1], b[0]) - np.arctan2(a[1], a[0]))


def unit(x):
    """Normalize a vector."""
    norm = np.linalg.norm(x)
    if norm > 0:
        return x / norm
    return x


def pursuit(p, lookahead):
    """Pure pursuit along the x-axis."""
    if np.abs(p[1]) >= lookahead:
        return np.array([0, -np.sign(p[1]) * lookahead])
    x = lookahead**2 - p[1] ** 2
    return np.array([x, -p[1]])


def perp2d(x):
    return np.array([-x[1], x[0]])


class QuadSlider:
    """Quadrilateral slider with half lengths hx and hy."""

    def __init__(self, hx, hy, cof=None):
        self.hx = hx
        self.hy = hy

        # center of friction (relative to the centroid of the shape)
        if cof is None:
            cof = np.zeros(2)
        self.cof = np.array(cof)

    def contact_point(self, s, check=True):
        """Contact point given parameter s.

        s is relative to the centroid of the shape
        """
        # for a quad, s is the y position along the face
        if check and np.abs(s) > self.hy:
            raise ValueError("Pusher lost contact with slider.")
        return np.array([-self.hx, s])  # - self.cof

    def contact_normal(self, s):
        """Inward-facing contact normal."""
        # constant for a quad
        return np.array([1, 0])

    def s_dot(self, α):
        """Time derivative of s."""
        return α


class CircleSlider:
    """Circular slider with radius r."""

    def __init__(self, r, cof=None):
        self.r = r

        if cof is None:
            cof = np.zeros(2)
        self.cof = np.array(cof)

    def contact_point(self, s, check=True):
        # for a quad, s is the angle of the contact point
        # TODO nothing to check?
        return rot2d(s) @ [-self.r, 0]

    def contact_normal(self, s):
        return np.array([np.cos(s), np.sin(s)])

    def s_dot(self, α):
        return α / self.r


class QPMotion:
    """Quadratic program-based equations of motion."""

    def __init__(self, M, μ):
        self.M = M
        self.μ = μ

        # cost is constant
        P = np.diag([1.0, 0, 0])

        # establish constraint sparsity pattern
        A, self.A_idx = self._A_sparsity_pattern()

        # initial problem setup
        self.problem = osqp.OSQP()
        self.problem.setup(
            P=sparse.csc_matrix(P),
            l=np.zeros(5),
            u=np.zeros(5),
            A=sparse.csc_matrix(A),
            verbose=False,
            eps_abs=1e-10,
            eps_rel=1e-10,
        )

    @staticmethod
    def _A_sparsity_pattern():
        """Compute the sparsity pattern of A."""
        A = np.ones((5, 3))
        A[2:, 0] = 0
        # indices are taken w.r.t. column-wise flattened data
        return A, np.nonzero(A.T.flatten())

    def _A_data(self, A):
        """Get data from A as required by CSC sparse matrix format."""
        return A.T.flatten()[self.A_idx]

    def _compute_constraints(self, vp, W, nc, nc_perp):
        """Compute updated constraint matrix and vectors l <= Ax <= u"""
        # motion cone constraint
        Lv = vp
        Uv = vp
        Av = np.hstack((nc_perp[:, None], W.T @ self.M @ W))

        # friction cone constraint
        Lf = -np.inf * np.ones(3)
        Uf = np.zeros(3)
        # fmt: off
        Af = np.hstack((np.zeros((3, 1)), np.vstack((
            -nc,
            -nc_perp - self.μ * nc,
            nc_perp - self.μ * nc,
        ))))
        # fmt: on

        # put them together
        L = np.concatenate((Lv, Lf))
        U = np.concatenate((Uv, Uf))
        A = np.vstack((Av, Af))

        return A, L, U

    def solve(self, vp, W, nc):
        """Update and solve the problem with the new data."""
        nc_perp = perp2d(nc)
        A, L, U = self._compute_constraints(vp, W, nc, nc_perp)

        self.problem.update(Ax=self._A_data(A), l=L, u=U)
        res = self.problem.solve()

        α = res.x[0]
        f = res.x[1:]
        v_slip = α * nc_perp

        # recover object velocity
        vo = vp - v_slip
        Vo = self.M @ W @ f

        return Vo, f, α


class CirclePath:
    """Circle path to track."""

    def __init__(self, radius):
        self.radius = radius

    def compute_closest_point(self, p):
        return self.radius * self.compute_normal(p)

    def compute_normal(self, p):
        return unit(p)

    def compute_lateral_offset(self, p):
        c = self.compute_closest_point(p)
        n = self.compute_normal(p)
        return -n @ (p - c)

    def compute_travel_direction(self, p):
        return rot2d(np.pi / 2) @ self.compute_normal(p)


class StraightPath:
    """Straight path to track."""

    def __init__(self, direction, origin=None):
        if origin is None:
            origin = np.zeros(2)
        self.origin = origin
        self.direction = direction
        self.perp = rot2d(np.pi / 2) @ direction

    def compute_closest_point(self, p):
        dist = (p - self.origin) @ self.direction
        return dist * self.direction + self.origin

    def compute_travel_direction(self, p):
        return self.direction

    def compute_lateral_offset(self, p):
        return self.perp @ (p - self.origin)
