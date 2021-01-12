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
from mm_model import FourInputModel

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
INERTIA = 0.5*MASS*RADIUS**2
MU = 1.0

DT = 0.01         # simulation timestep (s)
MPC_DT = 0.1       # lookahead timestep of the controller
PLOT_PERIOD = 10  # update plot every PLOT_PERIOD timesteps
CTRL_PERIOD = 10  # generate new control signal every CTRL_PERIOD timesteps
DURATION = 5.0     # duration of trajectory (s)

n = 5      # num horizon
ns_ee = 6  # num EE states
ns_q = 8   # num joint states
ni = 4     # num inputs
nc_eq = 0
nc_ineq = 3  # nf force + ni velocity + 1 constraint on q2
nv = ni      # num opt vars
nc = nc_eq + nc_ineq  # num constraints


def skew1(x):
    return np.array([[0, -x], [x, 0]])


def perp(v):
    return skew1(1).dot(v)


def hessian(f, argnums=0):
    return jax.jacfwd(jax.jacrev(f, argnums=argnums), argnums=argnums)


def main():
    N = int(DURATION / DT) + 1

    # tray params
    w = 0.4
    h = 0

    p_te_e = np.array([0, h])
    p_c1e_e = np.array([-w, 0])
    p_c2e_e = np.array([w, 0])

    p_c1t_e = p_c1e_e - p_te_e
    p_c2t_e = p_c2e_e - p_te_e

    model = FourInputModel(L1, L2, VEL_LIM, ACC_LIM)

    # MPC weights
    Q = np.diag([1, 1, 1, 0, 0, 0])
    R = 0.01 * np.eye(ni)

    ts = DT * np.arange(N)
    us = np.zeros((N, ni))
    pes = np.zeros((N, 3))
    pds = np.zeros((N, 3))
    ves = np.zeros((N, 3))
    pts = np.zeros((N, 3))

    # state of joints
    X_q = np.array([0, 0, 0.25*np.pi, -0.25*np.pi, 0, 0, 0, 0])

    pe = model.ee_position(X_q)
    ve = model.ee_velocity(X_q)
    pes[0, :] = pe
    ves[0, :] = ve
    pds[0, :] = pe

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
    tray_renderer = TrayRenderer(RADIUS, p_te_e, p_c1e_e, p_c2e_e, pe)
    trajectory_renderer = plotting.TrajectoryRenderer(trajectory, ts)
    video = plotting.Video(name='tray_balance_mm.mp4', fps=1./(DT*PLOT_PERIOD))
    plotter = plotting.RealtimePlotter([trajectory_renderer, robot_renderer, tray_renderer, start_renderer, goal_renderer], video=None)
    plotter.start(grid=True)

    # def force_balance_equations(X_ee, a_ee, f):
    #     θ, dθ = X_ee[2], X_ee[5]
    #     w_R_e = jnp.array([[jnp.cos(θ), -jnp.sin(θ)],
    #                        [jnp.sin(θ),  jnp.cos(θ)]])
    #     D = jnp.block([[w_R_e,       w_R_e],
    #                   [perp(t_p_1), perp(t_p_2)]])
    #     M = jnp.block([[MASS*jnp.eye(2), jnp.dot(jnp.dot(skew1(1), w_R_e), e_p_t).reshape(2, 1)],
    #                   [0, 0,             MOMENT_INERTIA]])
    #     rhs = jnp.array([0, -MASS*GRAVITY, 0]) + jnp.append(MASS*dθ**2*jnp.dot(w_R_e, e_p_t), 0)
    #     return jnp.dot(M, a_ee) - jnp.dot(D, f) - rhs

    def objective_unrolled(X_q_0, X_ee_d, var):
        ''' Unroll the objective over n timesteps. '''
        obj = 0
        X_q = X_q_0

        for i in range(n):
            u = var[i*nv:(i+1)*nv]
            X_q = model.step_unconstrained(X_q, u, MPC_DT)
            X_ee = model.ee_state(X_q)
            e = X_ee_d[i*ns_ee:(i+1)*ns_ee] - X_ee
            obj = obj + e @ Q @ e + u @ R @ u

        return obj

    # def constraints_unrolled(X_q_0, X_ee_d, var):
    #     ''' Unroll the equality (force balance) constraints over n timesteps. '''
    #     eq_con = jnp.zeros(n * nc_eq)
    #     ineq_con = jnp.zeros(n * nc_ineq)
    #     X_q = X_q_0
    #
    #     for i in range(n):
    #         vari = var[i*nv:(i+1)*nv]
    #         u = vari[:ni]
    #         f = vari[ni:]
    #
    #         X_ee = model.ee_state(X_q)
    #         a_ee = model.ee_acceleration(X_q, u)
    #         eq_coni = force_balance_equations(X_ee, a_ee, f)
    #         eq_con = jax.ops.index_update(eq_con, jax.ops.index[i*nc_eq:(i+1)*nc_eq], eq_coni)
    #
    #         # step the model before inequality constraints because there are
    #         # constraints on the k+1 state
    #         X_q = model.step_unconstrained(X_q, u, MPC_DT)
    #
    #         ineq_coni = jnp.concatenate((jnp.array([X_q[2]]), X_q[ni:], E @ vari))
    #         ineq_con = jax.ops.index_update(ineq_con, jax.ops.index[i*nc_ineq:(i+1)*nc_ineq], ineq_coni)
    #
    #     return jnp.concatenate((eq_con, ineq_con))

    def ineq_constraints(X_ee, a_ee):
        θ_ew, dθ_ew = X_ee[2], X_ee[5]
        a_ew_w, ddθ_ew = a_ee[:2], a_ee[2]
        R_ew = jnp.array([[ jnp.cos(θ_ew),  jnp.sin(θ_ew)],
                          [-jnp.sin(θ_ew), jnp.cos(θ_ew)]])
        S1 = skew1(1)
        g = jnp.array([0, GRAVITY])

        α1, α2 = MASS * R_ew @ (a_ew_w+g) + (ddθ_ew*S1-dθ_ew**2) @ p_te_e
        α3 = INERTIA * ddθ_ew

        ineq_con = jnp.array([
            MU*jnp.abs(α2) - jnp.abs(α1),  # TODO first abs may be unneeded
            α2,
            h**2*α1**2 + w*(w-2*MU*h)*α2**2 - α3**2])
        return ineq_con

    def ineq_constraints_unrolled(X_q_0, X_ee_d, var):
        # var is now just the lifted joint acceleration inputs
        X_q = X_q_0
        ineq_con = jnp.zeros(n * nc_ineq)

        for i in range(n):
            u = var[i*nv:(i+1)*nv]
            X_q = model.step_unconstrained(X_q, u, MPC_DT)

            X_ee = model.ee_state(X_q)
            a_ee = model.ee_acceleration(X_q, u)

            ineq_coni = ineq_constraints(X_ee, a_ee)
            ineq_con = jax.ops.index_update(ineq_con, jax.ops.index[i*nc_ineq:(i+1)*nc_ineq], ineq_coni)
        return ineq_con

    # Construct the SQP controller
    obj_fun = jax.jit(objective_unrolled)
    obj_jac = jax.jit(jax.jacfwd(objective_unrolled, argnums=2))
    obj_hess = jax.jit(hessian(objective_unrolled, argnums=2))
    objective = sqp.Objective(obj_fun, obj_jac, obj_hess)

    lbA = np.zeros(n * nc)
    ubA = np.infty * np.ones(n * nc)
    con_fun = jax.jit(ineq_constraints_unrolled)
    con_jac = jax.jit(jax.jacfwd(ineq_constraints_unrolled, argnums=2))
    constraints = sqp.Constraints(con_fun, con_jac, lbA, ubA)

    lb = -ACC_LIM * np.ones(n * nv)
    ub =  ACC_LIM * np.ones(n * nv)
    bounds = sqp.Bounds(lb, ub)

    controller = sqp.SQP(nv*n, nc*n, objective, constraints, bounds)

    for i in range(N - 1):
        t = ts[i+1]

        if i % CTRL_PERIOD == 0:
            t_sample = np.minimum(t + MPC_DT*np.arange(n), DURATION)
            pd, vd, _ = trajectory.sample(t_sample, flatten=True)
            X_ee_d = np.zeros(ns_ee*n)
            for j in range(n):
                X_ee_d[j*ns_ee:j*ns_ee+2] = pd[j*2:(j+1)*2]
                X_ee_d[j*ns_ee+3:(j+1)*ns_ee-1] = vd[j*2:(j+1)*2]

            start = time.time()
            var = controller.solve(X_q, X_ee_d)
            print(time.time() - start)
            u = var[:ni]

        # integrate the system
        X_q = model.step_unconstrained(X_q, u, DT)
        pe = model.ee_position(X_q)
        ve = model.ee_velocity(X_q)

        # tray position is a constant offset from EE frame
        # TODO not sure if this is correct
        θ_ew = pe[2]
        R_we = util.rotation_matrix(-θ_ew)
        pt = pe + np.append(R_we @ p_te_e, 0)

        # record
        us[i, :] = u
        pes[i+1, :] = pe
        ves[i+1, :] = ve
        pts[i+1, :] = pt
        pds[i+1, :] = pd[:3]

        if i % PLOT_PERIOD == 0:
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

    plt.show()


if __name__ == '__main__':
    main()
