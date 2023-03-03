import sympy
import numpy as np
import IPython


def rot2d(θ):
    """2D rotation matrix."""
    return sympy.Matrix([[sympy.cos(θ), -sympy.sin(θ)], [sympy.sin(θ), sympy.cos(θ)]])


def main():
    f_max = 5
    τ_max = 0.1
    xbar = -0.5
    v = 0.1  # velocity magnitude
    M = sympy.diag([1.0 / f_max**2, 1 / f_max**2, 1.0 / τ_max**2], unpack=True)
    nc = sympy.Matrix([1, 0])  # contact normal
    nc_perp = sympy.Matrix([0, 1])
    μ = 0.2

    stick = True

    # control gains
    kθ = 0.1
    ky = 0.1

    # desired direction
    Δ = sympy.Matrix([1, 0])

    # equilibrium state x = (yc, θo, s, θf)
    yc, θo, s, θf = sympy.symbols("yc, θo, s, θf")
    dt = sympy.symbols("dt")
    x = sympy.Matrix([yc, θo, s, θf])

    r_co_o = sympy.Matrix([xbar, s])
    W = sympy.Matrix([[1, 0], [0, 1], [-r_co_o[1], r_co_o[0]]])
    A = W.T * M * W

    # control law
    θv = (1 + kθ) * θf + ky * yc
    vp_world = v * sympy.Matrix([sympy.cos(θv), sympy.sin(θv)])
    # vp_world = v * sympy.Matrix([1, 0])  # open-loop version
    vp_body = rot2d(θo).T * vp_world

    if stick:
        f_next = A.inv() * vp_body

        # TODO we may be able to say that since s is just constant in mode 1, then
        # we don't care about it?
        θf_next = sympy.atan2(f_next[1], f_next[0]) + θo
        dsdt_next = nc_perp.dot(vp_body - A * f_next)
        s_next = s + dt * dsdt_next
        θo_next = θo + dt * (M * W * f_next)[2]
        yc_next = yc + dt * v * sympy.sin(θv)
        # θo_next = (M * W * f_next)[2]
        # yc_next = v * sympy.sin(θv)
    else:
        # assume μ = 0
        xhat = sympy.Matrix([1, 0])
        f_next = v * xhat.dot(A.inv() * vp_body) * xhat
        θf_next = sympy.atan2(f_next[1], f_next[0]) + θo
        dsdt_next = nc_perp.dot(vp_body - A * f_next)
        s_next = s + dt * dsdt_next
        θo_next = θo + dt * (M * W * f_next)[2]
        yc_next = yc + dt * v * sympy.sin(θv)

    x_next = sympy.Matrix([yc_next, θo_next, s_next, θf_next])
    dfdx = x_next.jacobian(x)
    J = np.array(dfdx.subs({yc: 0, θo: 0, s: 0.5, θf: 0, dt: 0.1})).astype(np.float64)
    e, v = np.linalg.eig(J)
    print(np.real(e))

    # IPython.embed()


if __name__ == "__main__":
    main()
