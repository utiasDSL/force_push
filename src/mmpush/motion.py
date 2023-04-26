"""Equations of motion for quasistatic single-point contact pusher-slider system."""
import numpy as np
from scipy import sparse
import osqp

from mmpush import util


def limit_surface_ellipsoid(f_max, τ_max):
    return np.diag([1.0 / f_max**2, 1.0 / f_max**2, 1.0 / τ_max**2])


class QPPusherSliderMotion:
    """Quadratic program-based equations of motion."""

    def __init__(self, f_max, τ_max, μ):
        self.M = limit_surface_ellipsoid(f_max, τ_max)
        self.μ = μ

        # Optimization variables are (α, f_x, f_y), with all quantities in the
        # body frame. If μ = 0, then we actually only need a single scalar
        # variable to represent the force, since it must act along the contact
        # normal.

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

    def solve(self, vp, rc, nc):
        """Update and solve the problem with the new data.

        vp is the pusher velocity (at the contact) in the body frame
        W is the matrix mapping the velocity at the contact to the body frame origin
        nc is the contact normal (in the body frame)

        Returns a tuple (Vo, f, α), where Vo is the generalized velocity of the
        body about the body frame origin, f is the force at the contact, and α
        is the slip velocity.
        """
        if vp @ nc < 0:
            raise ValueError("Pusher pulling away from slider.")

        W = np.array([[1, 0], [0, 1], [-rc[1], rc[0]]])
        nc_perp = util.perp2d(nc)
        A, L, U = self._compute_constraints(vp, W, nc, nc_perp)

        self.problem.update(Ax=self._A_data(A), l=L, u=U)
        res = self.problem.solve()

        if None in res.x:
            raise ValueError("QP solver failed!")

        α = res.x[0]
        ρ = res.x[1:]
        v_slip = α * nc_perp

        # recover object velocity
        vo = vp - v_slip
        Vo = self.M @ W @ ρ

        # normalize ρ to obtain the contact force
        f = ρ / np.sqrt(ρ @ W.T @ self.M @ W @ ρ)

        return Vo, f, α


class QPPusherSliderMotionZeroFriction:
    """Quadratic program-based equations of motion, specialized for zero
    contact friction.

    When μ = 0, this can be more numerically stable than the general version.
    """

    def __init__(self, f_max, τ_max):
        self.M = limit_surface_ellipsoid(f_max, τ_max)

        # Optimization variable is just α and scalar force f

        # cost is constant
        P = np.diag([1.0, 0.0])

        # establish constraint sparsity pattern
        A, self.A_idx = self._A_sparsity_pattern()

        # initial problem setup
        self.problem = osqp.OSQP()
        self.problem.setup(
            P=sparse.csc_matrix(P),
            l=np.zeros(3),
            u=np.zeros(3),
            A=sparse.csc_matrix(A),
            verbose=False,
            eps_abs=1e-10,
            eps_rel=1e-10,
        )

    @staticmethod
    def _A_sparsity_pattern():
        """Compute the sparsity pattern of A."""
        A = np.ones((3, 2))
        # indices are taken w.r.t. column-wise flattened data
        return A, np.nonzero(A.T.flatten())

    def _A_data(self, A):
        """Get data from A as required by CSC sparse matrix format."""
        return A.T.flatten()[self.A_idx]

    def _compute_constraints(self, vp, W, nc, nc_perp):
        """Compute updated constraint matrix and vectors l <= Ax <= u"""
        # motion cone constraint
        L = np.append(vp, 0)
        U = np.append(vp, np.inf)
        # fmt: off
        A = np.vstack((
            np.hstack((nc_perp[:, None], W.T @ self.M @ W @ nc[:, None])),
            [0, 1]))
        # fmt: on
        return A, L, U

    def solve(self, vp, rc, nc):
        """Update and solve the problem with the new data.

        vp is the pusher velocity (at the contact) in the body frame
        W is the matrix mapping the velocity at the contact to the body frame origin
        nc is the contact normal (in the body frame)

        Returns a tuple (Vo, f, α), where Vo is the generalized velocity of the
        body about the body frame origin, f is the force at the contact, and α
        is the slip velocity.
        """
        if vp @ nc < 0:
            raise ValueError("Pusher pulling away from slider.")

        W = np.array([[1, 0], [0, 1], [-rc[1], rc[0]]])
        nc_perp = util.perp2d(nc)
        A, L, U = self._compute_constraints(vp, W, nc, nc_perp)

        self.problem.update(Ax=self._A_data(A), l=L, u=U)
        res = self.problem.solve()

        if None in res.x:
            raise ValueError("QP solver failed!")

        α = res.x[0]
        ρ = res.x[1] * nc
        v_slip = α * nc_perp

        # recover object velocity
        vo = vp - v_slip
        Vo = self.M @ W @ ρ

        # normalize ρ to obtain the contact force
        f = ρ / np.sqrt(ρ @ W.T @ self.M @ W @ ρ)

        return Vo, f, α


def motion_cone(M, W, nc, μ):
    """Compute the boundaries of the motion cone.

    M defines the limit surface ellipsoid
    W maps the velocity at the contact to the body frame origin
    nc is the contact normal (in the body frame)
    μ is the contact friction coefficient

    Returns a tuple of two unit vectors representing the left and right
    boundaries of the motion cone at the body frame origin.
    """
    θc = np.arctan(μ)
    Rl = util.rot2d(θc)
    Rr = util.rot2d(-θc)

    Vl = M @ W @ Rl @ nc
    Vr = M @ W @ Rr @ nc
    return util.unit(Vl), util.unit(Vr)


def sliding(vp, W, Vi, nc):
    """Dynamics for sliding mode."""
    vi = W.T @ Vi
    κ = (vp @ nc) / (vi @ nc)
    vo = κ * vi
    Vo = κ * Vi
    α = util.perp2d(nc) @ (vp - vo)
    return Vo, α


def sticking(vp, c, rc):
    """Dynamics for sticking mode."""
    d = c**2 + rc @ rc
    xc, yc = rc
    vx = np.array([c**2 + xc**2, xc * yc]) @ vp / d
    vy = np.array([xc * yc, c**2 + yc**2]) @ vp / d
    ω = (xc * vy - yc * vx) / c**2
    Vo = np.array([vx, vy, ω])
    α = 0  # sticking ==> no slip
    return Vo, α


class PusherSliderMotion:
    """Analytical pusher-slider motion."""

    def __init__(self, f_max, τ_max, μ):
        self.M = limit_surface_ellipsoid(f_max, τ_max)
        self.μ = μ
        self.c = τ_max / f_max

    def solve(self, vp, rc, nc):
        if vp @ nc < 0:
            raise ValueError("Pusher pulling away from slider.")

        W = np.array([[1, 0], [0, 1], [-rc[1], rc[0]]])

        # compute edges of the motion cone at the contact point
        Vl, Vr = motion_cone(self.M, W, nc, self.μ)
        vl = W.T @ Vl
        vr = W.T @ Vr

        # if the pusher velocity is inside the motion cone we have sticking
        # contact, otherwise we have sliding contact
        if util.signed_angle(vp, vl) < 0:
            Vo, α = sliding(vp, W, Vl, nc)
        elif util.signed_angle(vp, vr) > 0:
            Vo, α = sliding(vp, W, Vr, nc)
        else:
            Vo, α = sticking(vp, self.c, rc)

        Vf = np.array([Vo[0], Vo[1], util.perp2d(rc) @ Vo[:2]])
        f = Vo[:2] / np.sqrt(Vf @ self.M @ Vf)

        return Vo, f, α
