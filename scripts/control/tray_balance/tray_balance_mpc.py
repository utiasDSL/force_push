#!/usr/bin/env python
import jax.numpy as jnp
import jax
import numpy as np
import matplotlib.pyplot as plt

from mm2d import plotter as plotting
from mm2d import trajectory as trajectories
from mm2d import util
from tray_balance import TrayRenderer
import qpoases

import IPython


# robot parameters
L1 = 1
L2 = 1
VEL_LIM = 1
ACC_LIM = 1

# tray parameters
GRAVITY = 9.81
RADIUS = 0.5
MASS = 1.0
MOMENT_INERTIA = 0.5*MASS*RADIUS**2
MU = 10.0

DT = 0.05        # timestep (s)
DURATION = 2.0  # duration of trajectory (s)

NUM_ITER = 3
NUM_WSR = 100     # number of working set recalculations

n = 3   # num iterations
ns = 6  # num states
ni = 3  # num inputs
nf = 4  # num force variables
nc_eq = 3
nc_ineq = 6
nv = ni + nf  # num opt vars
nc = nc_eq + nc_ineq  # num constraints


def skew1(x):
    return np.array([[0, -x], [x, 0]])


def perp(v):
    return skew1(1).dot(v)


class MPC(object):
    ''' Model predictive controller. '''
    def __init__(self, fun, jac, hess):
        self.fun = fun
        self.jac = jac
        self.hess = hess

    def _lookahead(self, x0, xd, var):
        ''' Generate lifted matrices proprogating the state N timesteps into the
            future. '''
        # TODO I think I'll need to restructure things a bit: divide up the
        # functions for these different things
        obj, con = self.fun(x0, vd, var)
        J_obj, J_con = self.jac(x0, vd, var)
        H = self.hess(x0, vd, var)
        
        # ni = self.model.ni  # number of joints
        # no = self.model.no  # number of Cartesian outputs
        #
        # fbar = np.zeros(no*N)         # Lifted forward kinematics
        # Jbar = np.zeros((no*N, ni*N))  # Lifted Jacobian
        # Qbar = np.kron(np.eye(N), self.Q)
        # Rbar = np.kron(np.eye(N), self.R)
        #
        # # lower triangular matrix of ni*ni identity matrices
        # Ebar = np.kron(np.tril(np.ones((N, N))), np.eye(ni))
        #
        # # Integrate joint positions from the last iteration
        # qbar = np.tile(q0, N+1)
        # qbar[ni:] = qbar[ni:] + self.dt * Ebar.dot(u)
        #
        # for k in range(N):
        #     q = qbar[(k+1)*ni:(k+2)*ni]
        #     p = self.model.forward(q)
        #     J = self.model.jacobian(q)
        #
        #     fbar[k*no:(k+1)*no] = p
        #     Jbar[k*no:(k+1)*no, k*ni:(k+1)*ni] = J
        #
        # dbar = fbar - pr
        # H = Rbar + self.dt**2*Ebar.T.dot(Jbar.T).dot(Qbar).dot(Jbar).dot(Ebar)
        # g = u.T.dot(Rbar) + self.dt*dbar.T.dot(Qbar).dot(Jbar).dot(Ebar)
        #
        # return H, g

    def _iterate(self, x0, xd, var):
        qp = qpoases.PySQProblem(nv, nc)
        options = qpoases.PyOptions()
        options.printLevel = qpoases.PyPrintLevel.NONE
        qp.setOptions(options)

        # Initial opt problem.
        H, g = self._lookahead(q0, pr, u, N)
        ret = qp.init(H, g, A, lb, ub, lbA, ubA, np.array([NUM_WSR]))
        delta = np.zeros(nv * n)
        qp.getPrimalSolution(delta)
        var = var + delta

        # Remaining sequence is hotstarted from the first.
        for i in range(NUM_ITER - 1):
            H, g = self._lookahead(q0, pr, u, N)
            qp.hotstart(H, g, A, lb, ub, lbA, ubA, np.array([NUM_WSR]))
            qp.getPrimalSolution(delta)
            var = var + delta

        return var

    def solve(self, x0, xd):
        ''' Solve the MPC problem at current state x0 given desired trajectory
            xd. '''
        # initialize decision variables
        var = np.zeros(nv * n)

        # iterate to final solution
        var = self._iterate(x0, xd, var)

        # return first optimal input
        return var[:ni]


def main():
    N = int(DURATION / DT) + 1

    # tray params
    a = 0.4
    b = 0.0

    e_p_t = np.array([b, 0])
    e_p_1 = np.array([-0.5*a, 0])
    e_p_2 = np.array([0.5*a, 0])
    t_p_1 = e_p_1 - e_p_t
    t_p_2 = e_p_2 - e_p_t


    # linear system
    Z = jnp.zeros((3, 3))
    A = jnp.eye(ns) + 0.1*jnp.block([[Z, jnp.eye(3)], [Z, Z]])
    B = jnp.block([[Z], [jnp.eye(3)]])

    # MPC weights
    Q = np.eye(ns)
    R = 0.1*np.eye(ni)

    # constant optimization matrices
    E = np.array([[0, 0, 0, 0, -1, 0, 0],
                  [0, 0, 0, 0, 0, 0, -1],
                  [0, 0, 0, 1, -MU, 0, 0],
                  [0, 0, 0, -1, -MU, 0, 0],
                  [0, 0, 0, 0, 0, 1, -MU],
                  [0, 0, 0, 0, 0, -1, -MU]])

    # ts = DT * np.arange(N)
    # us = np.zeros((N, 3))
    # pes = np.zeros((N, 3))
    # ves = np.zeros((N, 3))
    # pts = np.zeros((N, 3))
    # fs = np.zeros((N, 4))
    #
    # pe = np.array([0, 0, 0])
    # ve = np.array([0, 0, 0])
    # pes[0, :] = pe
    #
    # timescaling = trajectories.CubicTimeScaling(0.5*DURATION)
    # traj1 = trajectories.PointToPoint(pe, pe + [2, 0, 0], timescaling, 0.5*DURATION)
    # traj2 = trajectories.PointToPoint(pe + [2, 0, 0], pe, timescaling, 0.5*DURATION)
    # trajectory = trajectories.Chain([traj1, traj2])
    # # timescaling = trajectories.CubicTimeScaling(DURATION)
    # # trajectory = trajectories.PointToPoint(pe, pe + [2, 0, 0], timescaling, DURATION)
    #
    # pds, *other = trajectory.sample(ts)
    #
    # tray_renderer = TrayRenderer(RADIUS, e_p_t, e_p_1, e_p_2, pe)
    # trajectory_renderer = plotting.TrajectoryRenderer(trajectory, ts)
    # video = plotting.Video(name='tray_balance.mp4', fps=1./DT)
    # plotter = plotting.RealtimePlotter([tray_renderer, trajectory_renderer], video=None)
    # plotter.start()


    def force_balance_equations(x, u, f):
        θ, dθ = x[2], x[5]
        w_R_e = jnp.array([[jnp.cos(θ), -jnp.sin(θ)],
                           [jnp.sin(θ),  jnp.cos(θ)]])
        D = jnp.block([[w_R_e,       w_R_e],
                      [perp(t_p_1), perp(t_p_2)]])
        M = jnp.block([[MASS*jnp.eye(2), jnp.dot(jnp.dot(skew1(1), w_R_e), e_p_t).reshape(2, 1)],
                      [0, 0,             MOMENT_INERTIA]])
        rhs = jnp.array([0, -MASS*GRAVITY, 0]) + jnp.append(MASS*dθ**2*jnp.dot(w_R_e, e_p_t), 0)
        return jnp.dot(M, u) - jnp.dot(D, f) - rhs

    def objective_unrolled(x0, xd, var):
        ''' Unroll the objective. '''
        obj = 0
        x = x0

        for i in range(n):
            u = var[i*nv:i*nv+ni]
            x = jnp.dot(A, x) + jnp.dot(B, u)
            e = xd[i*ns:(i+1)*ns] - x
            obj = obj + jnp.linalg.multi_dot([e, Q, e]) + jnp.linalg.multi_dot([u, R, u])

        return obj

    def opt_unrolled(x0, xd, var):
        # TODO need to figure out if it is possible to have a variable n
        # is it actually needed? what if we optimize by propagating out the
        # last state over the horizon?
        # var is nv * n, where nv = 3 + 4
        obj = 0
        con = jnp.zeros(n * nc)
        xi = x0

        for i in range(n):
            vari = var[i*nv:(i+1)*nv]
            ui = vari[:ni]
            fi = vari[ni:]

            # constraints
            # TODO we could keep all the eq constraints first and add the
            # (linear) ineq constraints later (w.o the auto-diff fuss)
            eq_con = force_balance_equations(xi, ui, fi)
            ineq_con = jnp.dot(E, vari)
            con = jax.ops.index_update(con, jax.ops.index[i*nc:i*nc+nc_eq], eq_con)
            con = jax.ops.index_update(con, jax.ops.index[i*nc+nc_eq:(i+1)*nc], ineq_con)

            # propagate state and update objective
            xi = jnp.dot(A, xi) + jnp.dot(B, ui)  # propagate state
            ei = xd[i*ns:(i+1)*ns] - xi
            obj = obj + jnp.linalg.multi_dot([ei, Q, ei]) + jnp.linalg.multi_dot([ui, R, ui])

        return obj, con

    x0 = np.zeros(ns)
    xd = np.ones(n*ns)
    # u = 0.1*np.ones(3*n)
    # f = np.ones(4)
    var = np.ones(n*nv)

    jac = jax.jit(jax.jacfwd(opt_unrolled, argnums=2))

    J_obj, J_con = jac(x0, xd, var)
    IPython.embed()
    return

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
        w_R_e = util.rotation_matrix(theta)

        # LHS of force balance equality constraints
        D = np.block([[w_R_e,       w_R_e],
                      [perp(t_p_1), perp(t_p_2)]])
        M2 = np.block([[MASS*np.eye(2), skew1(1).dot(w_R_e).dot(e_p_t)[:, None]],
                       [0, 0,        MOMENT_INERTIA]])
        A = np.block([[M2, -D],
                      [E]])

        # RHS of force balance equality constraints
        eqA = np.array([0, -MASS*GRAVITY, 0]) + np.append(MASS*ve[2]**2*w_R_e.dot(e_p_t), 0)

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
        f1 = f[:2]
        f2 = f[2:]
        print(f'f1 = {f1}; f2 = {f2}')

        # integrate the system
        u = X[:3]  # EE accel input is first three values
        ve = ve + DT * u
        pe = pe + DT * ve

        # tray position is a constant offset from EE frame
        theta = pe[2]
        w_R_e = util.rotation_matrix(theta)
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
