"""Quasistatic pushing equations of motion from Lynch (1992).

Equivalently implemented as a QP.
"""
import numpy as np
from scipy import sparse
import osqp
import matplotlib.pyplot as plt
from mmpush import *
import IPython


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

    # compute edges of the motion cone at the contact point
    Vl, Vr = motion_cone(M, W, nc, μ)
    vl = W.T @ Vl
    vr = W.T @ Vr

    # if the pusher velocity is inside the motion cone we have sticking
    # contact, otherwise we have sliding contact
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
    main()
