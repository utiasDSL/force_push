#!/usr/bin/env python
import time
import jax.numpy as jnp
import jax
import numpy as np
import matplotlib.pyplot as plt
import qpoases

from mm2d import plotter as plotting
from mm2d import trajectory as trajectories
from mm2d import util, models
from mm2d.control import sqp

from tray_balance import TrayRenderer
from mm_model import ThreeInputModel

import IPython


# robot parameters
L1 = 1
L2 = 1
VEL_LIM = 1
ACC_LIM = 2

# tray parameters
GRAVITY = 9.81
RADIUS = 0.5
MASS = 0.5
MOMENT_INERTIA = 0.5*MASS*RADIUS**2
MU = 1.0

SIM_DT = 0.05        # timestep (s)
MPC_DT = 0.1
DURATION = 5.0  # duration of trajectory (s)

n = 5   # num horizon
ns_ee = 6  # num EE states
ns_q = 8   # num joint states
ni = 4  # num inputs
nf = 4  # num force variables
nc_eq = 3
nc_ineq = nf + ni + 1  # nf force + ni velocity + 1 constraint on q2
nv = ni + nf  # num opt vars
nc = nc_eq + nc_ineq  # num constraints


def skew1(x):
    return np.array([[0, -x], [x, 0]])


def perp(v):
    return skew1(1).dot(v)


def hessian(f, argnums=0):
    return jax.jacfwd(jax.jacrev(f, argnums=argnums), argnums=argnums)


def main():
    N = int(DURATION / SIM_DT) + 1

    # tray params
    a = 0.4
    b = 0.0

    e_p_t = np.array([b, 0])
    e_p_1 = np.array([-0.5*a, 0])
    e_p_2 = np.array([0.5*a, 0])
    t_p_1 = e_p_1 - e_p_t
    t_p_2 = e_p_2 - e_p_t

    model = ThreeInputModel(L1, L2, VEL_LIM, ACC_LIM)

    # MPC weights
    Q = np.diag([1, 1, 1, 0, 0, 0])
    R = np.diag([0.01, 0.01, 0.01, 0.01, 0.0001, 0.0001, 0.0001, 0.0001])

    E = np.array([[0, 0, 0, 0, 1, -MU, 0, 0],
                  [0, 0, 0, 0, -1, -MU, 0, 0],
                  [0, 0, 0, 0, 0, 0, 1, -MU],
                  [0, 0, 0, 0, 0, 0, -1, -MU]])

    ts = SIM_DT * np.arange(N)
    us = np.zeros((N, ni))
    pes = np.zeros((N, 3))
    pds = np.zeros((N, 3))
    ves = np.zeros((N, 3))
    pts = np.zeros((N, 3))
    fs = np.zeros((N, nf))

    # state of joints
    X_q = np.array([0, 0, 0.25*np.pi, -0.25*np.pi, 0, 0, 0, 0])

    pe = model.ee_position(X_q)
    ve = model.ee_velocity(X_q)
    pes[0, :] = pe
    ves[0, :] = ve
    pds[0, :] = pe

    # nominal bounds
    lb0 = np.concatenate((-ACC_LIM*np.ones(ni), np.array([-np.infty, 0, -np.infty, 0])))
    ub0 = np.concatenate((ACC_LIM*np.ones(ni), np.infty*np.ones(nf)))
    lb = np.tile(lb0, n)
    ub = np.tile(ub0, n)

    # nominal constraints
    lbA_eq0 = np.zeros(nc_eq)
    ubA_eq0 = np.zeros(nc_eq)
    lbA_eq = np.tile(lbA_eq0, n)
    ubA_eq = np.tile(ubA_eq0, n)

    lbA_ineq0 = np.concatenate(([0], -VEL_LIM*np.ones(ni), -np.infty * np.ones(nf)))
    ubA_ineq0 = np.concatenate(([np.pi], VEL_LIM*np.ones(ni), np.zeros(nf)))
    lbA_ineq = np.tile(lbA_ineq0, n)
    ubA_ineq = np.tile(ubA_ineq0, n)

    lbA = np.concatenate((lbA_eq, lbA_ineq))
    ubA = np.concatenate((ubA_eq, ubA_ineq))

    # timescaling = trajectories.CubicTimeScaling(0.5*DURATION)
    # traj1 = trajectories.PointToPoint(pe, pe + [2, 0, 0], timescaling, 0.5*DURATION)
    # traj2 = trajectories.PointToPoint(pe + [2, 0, 0], pe, timescaling, 0.5*DURATION)
    # trajectory = trajectories.Chain([traj1, traj2])

    timescaling = trajectories.QuinticTimeScaling(DURATION)
    # trajectory = trajectories.PointToPoint(pe, pe + np.array([2, 0, 0]), timescaling, DURATION)
    trajectory = trajectories.Circle(np.array(pe)[:2], 0.25, timescaling, DURATION)

    # trajectory = trajectories.Point(pe + np.array([2, 0, 0]))

    start_renderer = plotting.PointRenderer(pe[:2], color='k')
    goal_renderer = plotting.PointRenderer(pe[:2] + np.array([2, 0]), color='b')
    rendering_model = models.ThreeInputModel(l1=L1, l2=L2)
    robot_renderer = plotting.ThreeInputRenderer(rendering_model, X_q[:3])
    tray_renderer = TrayRenderer(RADIUS, e_p_t, e_p_1, e_p_2, pe)
    trajectory_renderer = plotting.TrajectoryRenderer(trajectory, ts)
    video = plotting.Video(name='tray_balance_mm.mp4', fps=1./SIM_DT)
    plotter = plotting.RealtimePlotter([trajectory_renderer, robot_renderer, tray_renderer, start_renderer, goal_renderer], video=None)
    plotter.start(grid=True)

    def force_balance_equations(X_ee, a_ee, f):
        θ, dθ = X_ee[2], X_ee[5]
        w_R_e = jnp.array([[jnp.cos(θ), -jnp.sin(θ)],
                           [jnp.sin(θ),  jnp.cos(θ)]])
        D = jnp.block([[w_R_e,       w_R_e],
                      [perp(t_p_1), perp(t_p_2)]])
        M = jnp.block([[MASS*jnp.eye(2), jnp.dot(jnp.dot(skew1(1), w_R_e), e_p_t).reshape(2, 1)],
                      [0, 0,             MOMENT_INERTIA]])
        rhs = jnp.array([0, -MASS*GRAVITY, 0]) + jnp.append(MASS*dθ**2*jnp.dot(w_R_e, e_p_t), 0)
        return jnp.dot(M, a_ee) - jnp.dot(D, f) - rhs

    def objective_unrolled(X_q_0, X_ee_d, var):
        ''' Unroll the objective over n timesteps. '''
        obj = 0
        X_q = X_q_0

        for i in range(n):
            v = var[i*nv:(i+1)*nv]
            u = v[:ni]
            X_q = model.step_unconstrained(X_q, u, MPC_DT)  #jnp.dot(A, x) + jnp.dot(B, u)
            X_ee = model.ee_state(X_q)
            e = X_ee_d[i*ns_ee:(i+1)*ns_ee] - X_ee
            obj = obj + e @ Q @ e + v @ R @ v

        return obj

    def constraints_unrolled(X_q_0, X_ee_d, var):
        ''' Unroll the equality (force balance) constraints over n timesteps. '''
        eq_con = jnp.zeros(n * nc_eq)
        ineq_con = jnp.zeros(n * nc_ineq)
        X_q = X_q_0

        for i in range(n):
            vari = var[i*nv:(i+1)*nv]
            u = vari[:ni]
            f = vari[ni:]

            X_ee = model.ee_state(X_q)
            a_ee = model.ee_acceleration(X_q, u)
            eq_coni = force_balance_equations(X_ee, a_ee, f)
            eq_con = jax.ops.index_update(eq_con, jax.ops.index[i*nc_eq:(i+1)*nc_eq], eq_coni)

            # step the model before inequality constraints because there are
            # constraints on the k+1 state
            X_q = model.step_unconstrained(X_q, u, MPC_DT)

            ineq_coni = jnp.concatenate((jnp.array([X_q[2]]), X_q[ni:], E @ vari))
            ineq_con = jax.ops.index_update(ineq_con, jax.ops.index[i*nc_ineq:(i+1)*nc_ineq], ineq_coni)

        return jnp.concatenate((eq_con, ineq_con))

    # Construct the SQP controller
    obj_fun = jax.jit(objective_unrolled)
    obj_jac = jax.jit(jax.jacfwd(objective_unrolled, argnums=2))
    obj_hess = jax.jit(hessian(objective_unrolled, argnums=2))
    objective = sqp.Objective(obj_fun, obj_jac, obj_hess)

    con_fun = jax.jit(constraints_unrolled)
    con_jac = jax.jit(jax.jacfwd(constraints_unrolled, argnums=2))
    constraints = sqp.Constraints(con_fun, con_jac, lbA, ubA)

    bounds = sqp.Bounds(lb, ub)

    controller = sqp.SQP(nv*n, nc*n, objective, constraints, bounds)

    for i in range(N - 1):
        t = ts[i+1]
        t_sample = np.minimum(t + MPC_DT*np.arange(n), DURATION)
        pd, vd, _ = trajectory.sample(t_sample, flatten=True)
        X_ee_d = np.zeros(ns_ee*n)
        for j in range(n):
            # X_ee_d[j*ns_ee:j*ns_ee+3] = pd[j*3:(j+1)*3]
            # X_ee_d[j*ns_ee+3:(j+1)*ns_ee] = vd[j*3:(j+1)*3]
            X_ee_d[j*ns_ee:j*ns_ee+2] = pd[j*2:(j+1)*2]
            X_ee_d[j*ns_ee+3:(j+1)*ns_ee-1] = vd[j*2:(j+1)*2]

        var = controller.solve(X_q, X_ee_d)
        u = var[:ni]
        f = var[ni:nv]

        # integrate the system
        X_q = model.step_unconstrained(X_q, u, SIM_DT)
        pe = model.ee_position(X_q)
        ve = model.ee_velocity(X_q)

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
        fs[i+1, :] = f

        tray_renderer.set_state(pt)
        robot_renderer.set_state(X_q[:3])
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

    plt.figure()
    plt.plot(ts, fs[:, 0], label='$f_{t,1}$', color='r', linestyle='--')
    plt.plot(ts, fs[:, 1], label='$f_{n,1}$', color='r')
    plt.plot(ts, fs[:, 2], label='$f_{t,2}$', color='b', linestyle='--')
    plt.plot(ts, fs[:, 3], label='$f_{n,2}$', color='b')
    plt.grid()
    plt.legend()
    plt.xlabel('Time (s)')
    plt.ylabel('Force (N)')
    plt.title('Contact forces')


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
