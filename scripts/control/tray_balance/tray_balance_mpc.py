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

DT = 0.1        # timestep (s)
DURATION = 5.0  # duration of trajectory (s)

NUM_ITER = 3
NUM_WSR = 100     # number of working set recalculations

n = 5   # num horizon
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


def hessian(f, argnums=0):
    return jax.jacfwd(jax.jacrev(f, argnums=argnums), argnums=argnums)


class MPC(object):
    ''' Model predictive controller. '''
    def __init__(self, obj_fun, obj_jac, obj_hess, eq_fun, eq_jac, ineq_fun, ineq_jac):
        self.obj_fun = obj_fun
        self.obj_jac = obj_jac
        self.obj_hess = obj_hess

        self.eq_fun = eq_fun
        self.eq_jac = eq_jac

        self.ineq_fun = ineq_fun
        self.ineq_jac = ineq_jac

    def _lookahead(self, x0, xd, var):
        ''' Generate lifted matrices proprogating the state N timesteps into the
            future. '''
        H = self.obj_hess(x0, xd, var)
        g = self.obj_jac(x0, xd, var)

        A_eq = self.eq_jac(x0, xd, var)
        A_ineq = self.ineq_jac(x0, xd, var)
        A = np.vstack((A_eq, A_ineq))

        lbA_eq = ubA_eq = -self.eq_fun(x0, xd, var)
        ubA_ineq = -self.ineq_fun(x0, xd, var)
        lbA_ineq = -np.infty * np.ones(nc_ineq * n)  # no lower bound
        lbA = np.concatenate((lbA_eq, lbA_ineq))
        ubA = np.concatenate((ubA_eq, ubA_ineq))


        return np.array(H, dtype=np.float64), np.array(g, dtype=np.float64), np.array(A, dtype=np.float64), np.array(lbA, dtype=np.float64), np.array(ubA, dtype=np.float64)

    def _iterate(self, x0, xd, var):
        qp = qpoases.PySQProblem(nv*n, nc*n)
        options = qpoases.PyOptions()
        options.printLevel = qpoases.PyPrintLevel.NONE
        qp.setOptions(options)

        # Initial opt problem.
        H, g, A, lbA, ubA = self._lookahead(x0, xd, var)
        qp.init(H, g, A, None, None, lbA, ubA, np.array([NUM_WSR]))
        delta = np.zeros(nv * n)
        qp.getPrimalSolution(delta)
        var = var + delta

        # Remaining sequence is hotstarted from the first.
        for i in range(NUM_ITER - 1):
            H, g, A, lbA, ubA = self._lookahead(x0, xd, var)
            qp.hotstart(H, g, A, None, None, lbA, ubA, np.array([NUM_WSR]))
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
    Z = np.zeros((3, 3))
    A = np.eye(ns) + DT*np.block([[Z, np.eye(3)], [Z, Z]])
    B = DT*np.block([[Z], [np.eye(3)]])

    # MPC weights
    Q = np.diag([1, 1, 0, 0, 0, 0])
    # R = 0.1*np.eye(ni)
    R = np.diag([0.01, 0.01, 0.01, 0.0001, 0.0001, 0.0001, 0.0001])

    # constant optimization matrices
    E = np.array([[0, 0, 0, 0, -1, 0, 0],
                  [0, 0, 0, 0, 0, 0, -1],
                  [0, 0, 0, 1, -MU, 0, 0],
                  [0, 0, 0, -1, -MU, 0, 0],
                  [0, 0, 0, 0, 0, 1, -MU],
                  [0, 0, 0, 0, 0, -1, -MU]])
    Ebar = np.kron(np.eye(n), E)

    ts = DT * np.arange(N+n)
    us = np.zeros((N, 3))
    pes = np.zeros((N, 3))
    pds = np.zeros((N, 3))
    ves = np.zeros((N, 3))
    pts = np.zeros((N, 3))
    fs = np.zeros((N, 4))

    x = np.zeros(ns)
    pe = x[:3]
    ve = x[3:]
    pes[0, :] = pe
    ves[0, :] = ve

    # timescaling = trajectories.CubicTimeScaling(0.5*DURATION)
    # traj1 = trajectories.PointToPoint(pe, pe + [2, 0, 0], timescaling, 0.5*DURATION)
    # traj2 = trajectories.PointToPoint(pe + [2, 0, 0], pe, timescaling, 0.5*DURATION)
    # trajectory = trajectories.Chain([traj1, traj2])
    timescaling = trajectories.CubicTimeScaling(DURATION)
    trajectory = trajectories.PointToPoint(pe, pe + [2, 0, 0], timescaling, DURATION)

    tray_renderer = TrayRenderer(RADIUS, e_p_t, e_p_1, e_p_2, pe)
    trajectory_renderer = plotting.TrajectoryRenderer(trajectory, ts)
    video = plotting.Video(name='tray_balance_mpc.mp4', fps=1./DT)
    plotter = plotting.RealtimePlotter([tray_renderer, trajectory_renderer], video=None)
    plotter.start()


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
        ''' Unroll the objective over n timesteps. '''
        # TODO this could fairly easily be done without autodiff
        obj = 0
        x = x0

        for i in range(n):
            v = var[i*nv:(i+1)*nv]
            # u = var[i*nv:i*nv+ni]
            u = v[:ni]
            x = jnp.dot(A, x) + jnp.dot(B, u)
            e = xd[i*ns:(i+1)*ns] - x
            obj = obj + jnp.linalg.multi_dot([e, Q, e]) + jnp.linalg.multi_dot([v, R, v])

        return obj

    def eq_con_unrolled(x0, xd, var):
        ''' Unroll the equality (force balance) constraints over n timesteps. '''
        eq_con = jnp.zeros(n * nc_eq)
        x = x0

        for i in range(n):
            vari = var[i*nv:(i+1)*nv]
            u = vari[:ni]
            f = vari[ni:]

            eq_coni = force_balance_equations(x, u, f)
            eq_con = jax.ops.index_update(eq_con, jax.ops.index[i*nc_eq:(i+1)*nc_eq], eq_coni)

            x = jnp.dot(A, x) + jnp.dot(B, u)  # propagate state

        return eq_con

    def ineq_con_unrolled(x0, xd, var):
        return np.dot(Ebar, var)

    def ineq_con_unrolled_jac(x0, xd, var):
        # trivial Jacobian since ineq constraints are already linear
        return Ebar

    obj_fun = jax.jit(objective_unrolled)
    obj_jac = jax.jit(jax.jacfwd(objective_unrolled, argnums=2))
    obj_hess = jax.jit(hessian(objective_unrolled, argnums=2))
    eq_fun = jax.jit(eq_con_unrolled)
    eq_jac = jax.jit(jax.jacfwd(eq_con_unrolled, argnums=2))

    controller = MPC(obj_fun, obj_jac, obj_hess, eq_fun, eq_jac,
                     ineq_con_unrolled, ineq_con_unrolled_jac)

    for i in range(N - 1):
        t = np.minimum(ts[i+1:i+1+n], DURATION)
        pd, vd, _ = trajectory.sample(t, flatten=True)
        xd = np.zeros(ns*n)
        try:
            for j in range(n):
                xd[j*ns:j*ns+3] = pd[j*3:(j+1)*3]
                xd[j*ns+3:(j+1)*ns] = vd[j*3:(j+1)*3]
        except ValueError:
            IPython.embed()
        u = controller.solve(x, xd)

        # integrate the system
        x = A.dot(x) + B.dot(u)
        pe = x[:3]
        ve = x[3:]

        # tray position is a constant offset from EE frame
        θ = pe[2]
        w_R_e = util.rotation_matrix(θ)
        pt = pe + np.append(w_R_e.dot(e_p_t), 0)

        # record
        us[i, :] = u
        pes[i+1, :] = pe
        ves[i+1, :] = ve
        pts[i+1, :] = pt
        pds[i+1, :] = pd[:3]
        # fs[i, :] = f

        tray_renderer.set_state(pt)
        plotter.update()
    plotter.done()

    # xe = pds[1:, 0] - ps[1:, 0]
    # ye = pds[1:, 1] - ps[1:, 1]
    # print('RMSE(x) = {}'.format(rms(xe)))
    # print('RMSE(y) = {}'.format(rms(ye)))

    plt.figure()
    plt.plot(ts[:N], pds[:, 0], label='$x_d$', color='b', linestyle='--')
    plt.plot(ts[:N], pds[:, 1], label='$y_d$', color='r', linestyle='--')
    plt.plot(ts[:N], pes[:, 0],  label='$x$', color='b')
    plt.plot(ts[:N], pes[:, 1],  label='$y$', color='r')
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
