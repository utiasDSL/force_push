#!/usr/bin/env python
import jax.numpy as jnp
import jax
import numpy as np
import matplotlib.pyplot as plt

from mm2d.plotter import RealtimePlotter, ThreeInputRenderer, TrajectoryRenderer
from mm2d.trajectory import Circle, Polygon, PointToPoint, CubicTimeScaling, QuinticTimeScaling, LinearTimeScaling, CubicBezier
import qpoases

import IPython


# robot parameters
L1 = 1
L2 = 1
VEL_LIM = 1
ACC_LIM = 1

# tray parameters
RADIUS = 0.5
MASS = 1.0
MOMENT_INTERTIA = I = 0.5*MASS*RADIUS**2
MU = 0.8

DT = 0.01        # timestep (s)
DURATION = 1.0  # duration of trajectory (s)

NUM_WSR = 100     # number of working set recalculations


class TrayRenderer(object):
    def __init__(self, length, a, p0):
        self.length = length
        self.a = a
        self.p = p0

    def set_state(self, p):
        self.p = p

    def render(self, ax):
        # origin
        xo = self.p[0]
        yo = self.p[1]

        # sides
        xl = xo - 0.5*self.length * np.cos(self.p[2])
        yl = yo - 0.5*self.length * np.sin(self.p[2])
        xr = xo + 0.5*self.length * np.cos(self.p[2])
        yr = yo + 0.5*self.length * np.sin(self.p[2])

        # contact points
        x1 = xo - 0.5*self.a * np.cos(self.p[2])
        y1 = yo - 0.5*self.a * np.sin(self.p[2])
        x2 = xo + 0.5*self.a * np.cos(self.p[2])
        y2 = yo + 0.5*self.a * np.sin(self.p[2])

        self.plot, = ax.plot([xl, xr], [yl, yr], color='k')
        self.points, = ax.plot([x1, x2], [y1, y2], 'o', color='k')

    def update_render(self):
        # origin
        xo = self.p[0]
        yo = self.p[1]

        # sides
        xl = xo - 0.5*self.length * np.cos(self.p[2])
        yl = yo - 0.5*self.length * np.sin(self.p[2])
        xr = xo + 0.5*self.length * np.cos(self.p[2])
        yr = yo + 0.5*self.length * np.sin(self.p[2])

        # contact points
        x1 = xo - 0.5*self.a * np.cos(self.p[2])
        y1 = yo - 0.5*self.a * np.sin(self.p[2])
        x2 = xo + 0.5*self.a * np.cos(self.p[2])
        y2 = yo + 0.5*self.a * np.sin(self.p[2])

        self.plot.set_xdata([xl, xr])
        self.plot.set_ydata([yl, yr])

        self.points.set_xdata([x1, x2])
        self.points.set_ydata([y1, y2])


def rot2d(a):
    c = np.cos(a)
    s = np.sin(a)
    R = np.array([[c, -s], [s, c]])
    return R


def skew1(x):
    return np.array([[0, -x], [x, 0]])


def perp(v):
    return skew1(1).dot(v)




def main():
    N = int(DURATION / DT) + 1

    gravity = 9.81

    # tray params
    l = 1
    r = 0.5 * l
    a = 0.4
    b = 0.1
    m = 1
    I = 0.5*m*r**2
    M = np.diag([m, m, I])
    mu = 0.8

    e_p_t = np.array([b, 0])
    t_p_1 = np.array([-0.5*a, 0])
    t_p_2 = np.array([0.5*a, 0])

    # control params
    kp = 1
    kv = 0.1

    # cost parameters
    Q = np.zeros((7, 7))
    Q[0, 0] = 1
    Q[1, 1] = 1
    R = np.diag([0.1, 0.1, 0.1, 0.0001, 0.0001, 0.0001, 0.0001])

    # constant optimization matrices
    E = np.array([[0, 0, 0, 0, -1, 0, 0],
                  [0, 0, 0, 0, 0, 0, -1],
                  [0, 0, 0, 1, -mu, 0, 0],
                  [0, 0, 0, -1, -mu, 0, 0],
                  [0, 0, 0, 0, 0, 1, -mu],
                  [0, 0, 0, 0, 0, -1, -mu]])
    ubA_E = np.zeros(6)
    lbA_E = -np.infty * np.ones(6)

    ts = np.array([i * DT for i in range(N)])
    us = np.zeros((N, 3))
    pes = np.zeros((N, 3))
    ves = np.zeros((N, 3))
    pts = np.zeros((N, 3))
    fs = np.zeros((N, 4))

    pe = np.array([0, 0, 0])
    ve = np.array([0, 0, 0])
    pes[0, :] = pe

    timescaling = CubicTimeScaling(DURATION)
    trajectory = PointToPoint(pe, pe + [2, 0, 0], timescaling, DURATION)

    pds, *other = trajectory.sample(ts)

    tray_renderer = TrayRenderer(l, a, pe)
    trajectory_renderer = TrajectoryRenderer(trajectory, ts)
    plotter = RealtimePlotter([tray_renderer, trajectory_renderer])
    plotter.start()

    def force_balance_equations(x, u, f):
        θ, dθ = x[2], x[5]
        w_R_e = jnp.array([[jnp.cos(θ), -jnp.sin(θ)],
                           [jnp.sin(θ), jnp.cos(θ)]])
        D = jnp.block([[w_R_e,       w_R_e],
                      [perp(t_p_1), perp(t_p_2)]])
        M = jnp.block([[m*jnp.eye(2), jnp.dot(jnp.dot(skew1(1), w_R_e), e_p_t).reshape(2, 1)],
                      [0, 0,        I]])
        rhs = jnp.array([0, -m*gravity, 0]) + jnp.append(m*dθ**2*jnp.dot(w_R_e, e_p_t), 0)
        return jnp.dot(M, u) - jnp.dot(D, f) - rhs

    Z = jnp.zeros((3, 3))
    A = jnp.eye(6) + 0.1*jnp.block([[Z, jnp.eye(3)], [Z, Z]])
    B = jnp.block([[Z], [jnp.eye(3)]])

    def force_balance_equations_unrolled(x0, xd, u, f):
        # TODO need to figure out if it is possible to have a variable n
        # is it actually needed? what if we optimize by propagating out the
        # last state over the horizon?
        n = 3
        ns = 6
        ni = 3
        nc = 3

        e = jnp.zeros(n * ns)
        F = jnp.zeros(n * nc)
        xi = x0

        for i in range(n):
            ui = u[i*ni:(i+1)*ni]
            Fi = force_balance_equations(xi, ui, f)
            xi = jnp.dot(A, xi) + jnp.dot(B, ui)
            ei = xd[i*ns:(i+1)*ns] - xi
            e = jax.ops.index_update(e, jax.ops.index[i*ns:(i+1)*ns], ei)
            F = jax.ops.index_update(F, jax.ops.index[i*nc:(i+1)*nc], Fi)

        # inequality constraints are all linear already

        return F, e

        # θ, dθ = x2[2], x2[5]
        # w_R_e = jnp.array([[jnp.cos(θ), -jnp.sin(θ)],
        #                    [jnp.sin(θ), jnp.cos(θ)]])
        # D = jnp.block([[w_R_e,       w_R_e],
        #               [perp(t_p_1), perp(t_p_2)]])
        # M = jnp.block([[m*jnp.eye(2), jnp.dot(jnp.dot(skew1(1), w_R_e), e_p_t).reshape(2, 1)],
        #               [0, 0,        I]])
        # rhs = jnp.array([0, -m*gravity, 0]) + jnp.append(m*dθ**2*jnp.dot(w_R_e, e_p_t), 0)
        # return jnp.dot(M, u2) - jnp.dot(D, f) - rhs

    n = 3
    x0 = np.zeros(6)
    xd = np.ones(6*n)
    u = 0.1*np.ones(3*n)
    f = np.ones(4)
    Ju = jax.jit(jax.jacfwd(force_balance_equations_unrolled, argnums=2))
    J = Ju(x0, xd, u, f)
    IPython.embed()

    for i in range(N - 1):
        t = ts[i]

        pd, vd, ad = trajectory.sample(t, flatten=True)

        # solve opt problem
        nv = 7
        nc = 9

        # Reference values determined by PD with feedforward
        ddx_ref = kp*(pd[0] - pe[0]) + kv*(vd[0] - ve[0]) + ad[0]
        ddy_ref = kp*(pd[1] - pe[1]) + kv*(vd[1] - ve[1]) + ad[1]

        # Construct reference for opt variables: first 3 are x, y, θ; last four
        # are the two contact forces (2 components each). For now, we only care about x
        # and y and let the angle be a free DOF.
        Xref = np.zeros(nv)
        Xref[0] = ddx_ref
        Xref[1] = ddy_ref

        # cost
        H = Q + R
        g = -Q.dot(Xref)

        # constraints
        theta = pe[2]
        w_R_e = rot2d(theta)


        # LHS of force balance equality constraints
        D = np.block([[w_R_e,       w_R_e],
                      [perp(t_p_1), perp(t_p_2)]])
        M2 = np.block([[m*np.eye(2), skew1(1).dot(w_R_e).dot(e_p_t)[:, None]],
                       [0, 0,        I]])
        A = np.block([[M2, -D],
                      [E]])

        # RHS of force balance equality constraints
        eqA = np.array([0, -m*gravity, 0]) + np.append(m*ve[2]**2*w_R_e.dot(e_p_t), 0)

        lbA = np.concatenate((eqA, lbA_E))
        ubA = np.concatenate((eqA, ubA_E))

        # solve the QP
        qp = qpoases.PyQProblem(nv, nc)
        options = qpoases.PyOptions()
        options.printLevel = qpoases.PyPrintLevel.NONE
        qp.setOptions(options)
        ret = qp.init(H, g, A, None, None, lbA, ubA, np.array([NUM_WSR]))

        X = np.zeros(7)
        qp.getPrimalSolution(X)
        f = X[3:]

        # integrate the system
        u = X[:3]  # EE accel input is first three values
        ve = ve + DT * u
        pe = pe + DT * ve

        # tray position is a constant offset from EE frame
        theta = pe[2]
        w_R_e = rot2d(theta)
        pt = pe + np.append(w_R_e.dot(e_p_t), 0)

        # record
        us[i, :] = u
        pes[i+1, :] = pe
        ves[i+1, :] = ve
        pts[i+1, :] = pt
        pds[i, :] = pd
        fs[i, :] = f

        tray_renderer.set_state(pt)
        plotter.update()
    plotter.done()

    # IPython.embed()

    # xe = pds[1:, 0] - ps[1:, 0]
    # ye = pds[1:, 1] - ps[1:, 1]
    # print('RMSE(x) = {}'.format(rms(xe)))
    # print('RMSE(y) = {}'.format(rms(ye)))

    plt.figure()
    plt.plot(ts, pds[:, 0], label='$x_d$', color='b', linestyle='--')
    plt.plot(ts, pds[:, 1], label='$y_d$', color='r', linestyle='--')
    plt.plot(ts, pes[:, 0],  label='$x$', color='b')
    plt.plot(ts, pes[:, 1],  label='$y$', color='r')
    plt.grid()
    plt.legend()
    plt.xlabel('Time (s)')
    plt.ylabel('Position')
    plt.title('End effector position')
    #
    # plt.figure()
    # plt.plot(ts, dqs[:, 0], label='$\\dot{q}_x$')
    # plt.plot(ts, dqs[:, 1], label='$\\dot{q}_1$')
    # plt.plot(ts, dqs[:, 2], label='$\\dot{q}_2$')
    # plt.grid()
    # plt.legend()
    # plt.title('Actual joint velocity')
    # plt.xlabel('Time (s)')
    # plt.ylabel('Velocity')
    #
    # plt.figure()
    # plt.plot(ts, us[:, 0], label='$u_x$')
    # plt.plot(ts, us[:, 1], label='$u_1$')
    # plt.plot(ts, us[:, 2], label='$u_2$')
    # plt.grid()
    # plt.legend()
    # plt.title('Commanded joint velocity')
    # plt.xlabel('Time (s)')
    # plt.ylabel('Velocity')
    #
    # plt.figure()
    # plt.plot(ts, qs[:, 0], label='$q_x$')
    # plt.plot(ts, qs[:, 1], label='$q_1$')
    # plt.plot(ts, qs[:, 2], label='$q_2$')
    # plt.grid()
    # plt.legend()
    # plt.title('Joint positions')
    # plt.xlabel('Time (s)')
    # plt.ylabel('Joint positions')
    #
    plt.show()


if __name__ == '__main__':
    main()
