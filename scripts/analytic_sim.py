"""Simulation of pushing based on quasistatic model and my control law."""
import numpy as np
from scipy import sparse
import osqp
import matplotlib.pyplot as plt
from mmpush import *
import IPython


class QuadSlider:
    def __init__(self, hx, hy):
        """Quadrilateral slider with half lengths hx and hy."""
        self.hx = hx
        self.hy = hy

    def contact_point(self, s):
        """Contact point given parameter s."""
        # for a quad, s is the y position along the face
        return np.array([-self.hx, s])

    def contact_normal(self, s):
        """Inward-facing contact normal."""
        # constant for a quad
        return np.array([1, 0])

    def s_dot(self, α):
        """Time derivative of s."""
        return α


class CircleSlider:
    def __init__(self, r):
        """Circle slider with radius r."""
        self.r = r

    def contact_point(self, s):
        # for a quad, s is the angle of the contact point
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
            eps_abs=1e-6,
            eps_rel=1e-6,
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


def simulate_pushing(motion, slider, kθ, ky, x0, duration, timestep):
    x = x0.copy()
    xs = [x0.copy()]
    ts = [0]

    Δ = unit([1, 0])

    t = 0
    while t < duration:
        # compute required quantities
        φ = x[2]
        s = x[3]
        C_wb = rot2d(φ)
        f = C_wb.T @ x[4:]
        nc = slider.contact_normal(s)
        r_co_o = slider.contact_point(s)
        W = np.array([[1, 0], [0, 1], [-r_co_o[1], r_co_o[0]]])

        # term to correct deviations from desired line
        # this is simpler than pure pursuit!
        r_cw_w = x[:2] + C_wb @ r_co_o
        θy = ky * r_cw_w[1]

        # angle-based control law
        θd = signed_angle(Δ, C_wb @ unit(f))
        θ = (1 + kθ) * θd + θy
        vp = 1 * C_wb.T @ rot2d(θ) @ Δ

        # equations of motion
        Vo, f, α = motion.solve(vp, W, nc)

        # update state
        x[:2] += timestep * C_wb @ Vo[:2]
        x[2] += timestep * Vo[2]
        x[3] += timestep * slider.s_dot(α)
        x[4:] = C_wb @ f

        t += timestep

        xs.append(x.copy())
        ts.append(t)

    return np.array(ts), np.array(xs)


def main():
    f_max = 5
    τ_max = 0.1
    M = np.diag([1.0 / f_max**2, 1 / f_max**2, 1.0 / τ_max**2])
    μ = 0.2

    # control gains
    kθ = 0.2
    ky = 0.2

    x0 = np.array([0.0, 0, 0, 0.1, 1, 0])

    motion = QPMotion(M, μ)
    # slider = QuadSlider(0.5, 0.5)
    slider = CircleSlider(0.5)

    duration = 100
    timestep = 0.1

    ts, xs = simulate_pushing(motion, slider, kθ, ky, x0, duration, timestep)

    plt.figure()
    plt.plot(ts, xs[:, 0], label="x")
    plt.plot(ts, xs[:, 1], label="y")
    plt.plot(ts, xs[:, 2], label="φ")
    plt.plot(ts, xs[:, 3], label="s")
    plt.plot(ts, xs[:, 4], label="fx")
    plt.plot(ts, xs[:, 5], label="fy")
    plt.legend()
    plt.grid()
    plt.show()


if __name__ == "__main__":
    main()
