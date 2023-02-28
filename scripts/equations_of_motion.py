"""Quasistatic pushing equations of motion from Lynch (1992).

Equivalently implemented as a QP.
"""
import numpy as np
from scipy import sparse
import osqp
import matplotlib.pyplot as plt
import IPython


def rot2d(θ):
    """2D rotation matrix."""
    return np.array([[np.cos(θ), -np.sin(θ)], [np.sin(θ), np.cos(θ)]])


def signed_angle(a, b):
    """Angle to rotate a to b.

    See <https://stackoverflow.com/a/2150111/5145874>"""
    return np.arctan2(b[1], b[0]) - np.arctan2(a[1], a[0])


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


def motion_cone(M, W, nc, μ):
    θc = np.arctan(μ)
    Rl = rot2d(θc)
    Rr = rot2d(-θc)

    Vl = M @ W @ Rl @ nc
    Vr = M @ W @ Rr @ nc
    return unit(Vl), unit(Vr)


def sliding(vp, W, Vi, nc):
    vi = W.T @ Vi
    κ = (vp @ nc) / (vi @ nc)
    vo = κ * vp
    Vo = κ * Vi
    return Vo


def sticking(vp, M, r_co_o):
    c = M[2, 2] / M[0, 0]
    d = c**2 + r_co_o @ r_co_o
    xc, yc = r_co_o
    vx = np.array([c**2 + xc**2, xc * yc]) @ vp / d
    vy = np.array([xc * yc, c**2 + yc**2]) @ vp / d
    ω = (xc * vy - yc * vx) / c**2
    return np.array([vx, vy, ω])


class QPMotion:
    def __init__(self, M, nc, μ):
        self.M = M
        self.nc = nc
        self.μ = μ

        P = np.diag([1., 0, 0])

        # setup placeholders to establish the sparsity pattern of A
        vp = np.array([1, 0])
        W = np.array([[1, 0], [0, 1], [1, 1]])
        A, L, U = self._compute_constraints(vp, W)

        self.problem = osqp.OSQP()
        self.problem.setup(
            P=sparse.csc_matrix(P),
            l=np.zeros(5),
            u=np.zeros(5),
            A=sparse.csc_matrix(A),
            verbose=False,
            eps_abs=1e-5,
            eps_rel=1e-5,
        )

    def _compute_constraints(self, vp, W):
        # motion cone constraint
        Lv = vp
        Uv = vp
        Av = np.hstack((perp2d(self.nc)[:, None], W.T @ self.M @ W))

        # friction cone constraint
        Lf = -np.inf * np.ones(3)
        Uf = np.zeros(3)
        Af = np.array([[0, -self.μ, 1], [0, -self.μ, -1], [0, -1, 0]])

        # put them together
        L = np.concatenate((Lv, Lf))
        U = np.concatenate((Uv, Uf))
        A = np.vstack((Av, Af))

        return A, L, U

    def solve(self, vp, W):
        A, L, U = self._compute_constraints(vp, W)
        self.problem.update(Ax=sparse.csc_matrix(A).data, l=L, u=U)
        res = self.problem.solve()

        α = res.x[0]
        f = res.x[1:]
        v_slip = α * perp2d(self.nc)

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
        eps_abs=1e-5,
        eps_rel=1e-5,
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
    f_max = 1
    τ_max = 0.1
    M = np.diag([1.0 / f_max**2, 1 / f_max**2, 1.0 / τ_max**2])
    μ = 0.2

    xbar = -0.5
    nc = np.array([1, 0])

    kθ = 0.1
    ky = 0.1

    Ro = rot2d(np.pi / 2)
    Δ = unit([1, 0])
    Δ_perp = Ro @ Δ
    Np = np.outer(Δ, Δ)
    Np_perp = np.outer(Δ_perp, Δ_perp)
    K = Np + (kθ + 1) * Np_perp

    x = np.array([0.0, 0, 0, -0.2, 1, 0])
    xs = [x.copy()]

    motion = QPMotion(M, nc, μ)

    T = 100
    N = 1000
    dt = T / N
    for i in range(N):
        t = i / N

        φ = x[2]
        s = x[3]
        f = x[4:]
        C_wb = rot2d(φ)
        r_co_o = np.array([xbar, s])
        W = np.array([[1, 0], [0, 1], [-r_co_o[1], r_co_o[0]]])

        # TODO to try pure pursuit
        # r_cw_w = x[:2] + C_wb @ r_co_o
        # Δ = unit(pursuit(r_cw_w, 1))
        # Δ_perp = Ro @ Δ
        # Np = np.outer(Δ, Δ)
        # Np_perp = np.outer(Δ_perp, Δ_perp)
        # K = Np + (kθ + 1) * Np_perp

        # term to correct deviations from desired line
        # this is simpler than pure pursuit!
        r_cw_w = x[:2] + C_wb @ r_co_o
        φ = ky * r_cw_w[1]

        # angle-based control law
        θd = signed_angle(Δ, C_wb @ unit(f))
        θ = (1 + kθ) * θd + φ
        vp = 1 * C_wb.T @ rot2d(θ) @ Δ

        # control input
        # vp = 0.1 * C_wb.T @ K @ C_wb @ unit(f)
        # vp = 1 * unit(C_wb.T @ K @ C_wb @ f)
        # vp = 0.1 * C_wb.T @ K @ C_wb @ f
        # vp = C_wb.T @ np.array([0.1, 0.1 * Δ_perp @ C_wb @ f])
        # print(Δ_perp @ C_wb @ f)

        # equations of motion
        # Vo, f, α = qp_form(vp, W, M, nc, μ)
        Vo, f, α = motion.solve(vp, W)

        # update state
        x[:2] += dt * C_wb @ Vo[:2]
        x[2] += dt * Vo[2]
        x[3] += dt * α
        x[4:] = f

        xs.append(x.copy())
    xs = np.array(xs)

    plt.figure()
    ts = dt * np.arange(N)
    plt.plot(ts, xs[:-1, 0], label="x")
    plt.plot(ts, xs[:-1, 1], label="y")
    plt.plot(ts, xs[:-1, 3], label="s")
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


if __name__ == "__main__":
    simulate()
    # main()
