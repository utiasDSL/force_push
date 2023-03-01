"""Quasistatic pushing equations of motion from Lynch (1992).

Equivalently implemented as a QP.
"""
import numpy as np
from scipy import sparse
import osqp
import matplotlib.pyplot as plt
from mmpush import *
import IPython


def motion_cone(M, W, nc, μ):
    """Compute the boundaries of the motion cone."""
    θc = np.arctan(μ)
    Rl = rot2d(θc)
    Rr = rot2d(-θc)

    Vl = M @ W @ Rl @ nc
    Vr = M @ W @ Rr @ nc
    return unit(Vl), unit(Vr)


def sliding(vp, W, Vi, nc):
    """Dynamics for sliding mode."""
    vi = W.T @ Vi
    κ = (vp @ nc) / (vi @ nc)
    vo = κ * vp
    Vo = κ * Vi
    return Vo


def sticking(vp, M, r_co_o):
    """Dynamics for sticking mode."""
    c = M[2, 2] / M[0, 0]
    d = c**2 + r_co_o @ r_co_o
    xc, yc = r_co_o
    vx = np.array([c**2 + xc**2, xc * yc]) @ vp / d
    vy = np.array([xc * yc, c**2 + yc**2]) @ vp / d
    ω = (xc * vy - yc * vx) / c**2
    return np.array([vx, vy, ω])


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


def qp_form(vp, W, M, nc, μ):
    # cost
    P = np.diag([1, 0, 0])
    q = np.zeros(3)

    # constraints
    # motion cone constraint
    Lv = vp
    Uv = vp
    Av = np.hstack((perp2d(nc)[:, None], W.T @ M @ W))

    # friction cone constraint
    Lf = -np.inf * np.ones(3)
    Uf = np.zeros(3)
    Af = np.array([[0, -μ, 1], [0, -μ, -1], [0, -1, 0]])

    L = np.concatenate((Lv, Lf))
    U = np.concatenate((Uv, Uf))
    A = np.vstack((Av, Af))

    # solve the problem
    # reducing tolerances appears necessary for accuracy of solution in the
    # long term
    m = osqp.OSQP()
    m.setup(
        P=sparse.csc_matrix(P),
        A=sparse.csc_matrix(A),
        l=L,
        u=U,
        verbose=False,
        eps_abs=1e-6,
        eps_rel=1e-6,
    )
    res = m.solve()

    α = res.x[0]
    f = res.x[1:]
    v_slip = α * perp2d(nc)

    # recover object velocity
    vo = vp - v_slip
    Vo = M @ W @ f
    # Vo_hat = unit(M @ W @ f)
    # Vo_mag = np.linalg.norm(vo) / np.linalg.norm(W.T @ Vo_hat)
    # Vo = Vo_mag * Vo_hat

    return Vo, f, α


def simulate():
    f_max = 5
    τ_max = 0.1
    M = np.diag([1.0 / f_max**2, 1 / f_max**2, 1.0 / τ_max**2])
    μ = 0.2

    # control gains
    kθ = 0.2
    ky = 0.2

    Δ = unit([1, 0])
    # Ro = rot2d(np.pi / 2)
    # Δ_perp = Ro @ Δ
    # Np = np.outer(Δ, Δ)
    # Np_perp = np.outer(Δ_perp, Δ_perp)
    # K = Np + (kθ + 1) * Np_perp

    x = np.array([0.0, 0, 0, 0.1, 1, 0])
    xs = [x.copy()]

    motion = QPMotion(M, μ)
    # slider = QuadSlider(0.5, 0.5)
    slider = CircleSlider(0.5)

    T = 100
    N = 1000
    dt = T / N
    for i in range(N):
        t = i / N

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

        # other control formulations I tried
        # r_cw_w = x[:2] + C_wb @ r_co_o
        # Δ = unit(pursuit(r_cw_w, 1))
        # Δ_perp = Ro @ Δ
        # Np = np.outer(Δ, Δ)
        # Np_perp = np.outer(Δ_perp, Δ_perp)
        # K = Np + (kθ + 1) * Np_perp

        # vp = 1 * C_wb.T @ K @ C_wb @ unit(f) + ky * r_cw_w[1]
        # vp = C_wb.T @ K @ C_wb @ f #+ ky * r_cw_w[1]
        # fw = C_wb @ f
        # vp = [1, 0.1 * fw[1] + ky * r_cw_w[1]]
        # vp = 1 * unit([1, -0.1 * fw[1]])

        # vp = 1 * unit(C_wb.T @ K @ C_wb @ f)
        # vp = 0.1 * C_wb.T @ K @ C_wb @ f
        # vp = C_wb.T @ np.array([0.1, 0.1 * Δ_perp @ C_wb @ f])
        # print(Δ_perp @ C_wb @ f)

        # equations of motion
        Vo, f, α = motion.solve(vp, W, nc)

        # update state
        x[:2] += dt * C_wb @ Vo[:2]
        x[2] += dt * Vo[2]
        x[3] += dt * slider.s_dot(α)
        x[4:] = C_wb @ f

        xs.append(x.copy())
    xs = np.array(xs)

    # NOTE: it appears to be fy (in the world frame) and yc that converge
    plt.figure()
    ts = dt * np.arange(N)
    plt.plot(ts, xs[:-1, 0], label="x")
    plt.plot(ts, xs[:-1, 1], label="y")
    plt.plot(ts, xs[:-1, 2], label="φ")
    plt.plot(ts, xs[:-1, 3], label="s")
    plt.plot(ts, xs[:-1, 4], label="fx")
    plt.plot(ts, xs[:-1, 5], label="fy")
    plt.legend()
    plt.grid()
    plt.show()


def main():
    f_max = 5
    τ_max = 0.1
    M = np.diag([1.0 / f_max**2, 1 / f_max**2, 1.0 / τ_max**2])
    μ = 0.2

    r_co_o = np.array([-0.5, -0])
    nc = np.array([1, 0])
    vp = rot2d(0.1) @ np.array([1, 0])
    W = np.array([[1, 0], [0, 1], [-r_co_o[1], r_co_o[0]]])

    A = W.T @ M @ W

    Vl, Vr = motion_cone(M, W, nc, μ)
    vl = W.T @ Vl
    vr = W.T @ Vr

    if signed_angle(vp, vl) < 0:
        Vo = sliding(vp, W, Vl, nc)
    elif signed_angle(vp, vr) > 0:
        Vo = sliding(vp, W, Vr, nc)
    else:
        Vo = sticking(vp, M, r_co_o)

    print("Analytical")
    print(f"Vo = {Vo}")
    print(f"vo = {W.T @ Vo}")

    Vo, f, α = qp_form(vp, W, M, nc, μ)

    print("\nQP")
    print(f"Vo = {Vo}")
    print(f"vo = {W.T @ Vo}")


def force_iteration():
    f_max = 1
    τ_max = 0.1
    M = np.diag([1.0 / f_max**2, 1 / f_max**2, 1.0 / τ_max**2])
    μ = 0.2

    r_co_o = np.array([-0.5, 0.1])
    nc = np.array([1, 0])
    vp = rot2d(0) @ np.array([1, 0])
    W = np.array([[1, 0], [0, 1], [-r_co_o[1], r_co_o[0]]])

    kθ = 0.1
    Δ = unit([1, 0])
    φ = 0.1
    C_wb = rot2d(φ)
    A = W.T @ M @ W
    f = np.linalg.solve(A, vp)

    # print(f)
    # for i in range(10):
    #     # TODO enforce limits
    #     θd = signed_angle(Δ, C_wb @ unit(f))
    #     θ = (1 + kθ) * θd + φ
    #     vp = 1 * C_wb.T @ rot2d(θ) @ Δ
    #     f_new = np.linalg.solve(A, vp)
    #     print(f_new - f)
    #     f = f_new

    IPython.embed()


if __name__ == "__main__":
    # main()
    simulate()
    # force_iteration()
