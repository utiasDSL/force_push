#!/usr/bin/env python
"""Baseline tray balancing formulation."""
import jax.numpy as jnp
import jax
import numpy as np
import matplotlib.pyplot as plt
import pymunk
import pymunk.matplotlib_util
from _tkinter import TclError

from mm2d import plotter as plotting
from mm2d import trajectory as trajectories
from mm2d import util
from mm2d.control import sqp

from mm_model import FourInputModel
from pymunk_sim import PymunkSimulationTrayBalance, PymunkRenderer

import IPython


# robot parameters
L1 = 1
L2 = 1
VEL_LIM = 1
ACC_LIM = 10

# tray parameters
GRAVITY = 9.81
RADIUS = 0.5
MASS = 0.5
# INERTIA = 0.25*MASS*RADIUS**2
TRAY_MU = 0.75
TRAY_W = 0.1
TRAY_H = 0.05
INERTIA = MASS * (3*RADIUS**2 + (2*TRAY_H)**2) / 12.0

OBJ_W = 0.1
OBJ_H = 0.1
OBJ_MU = TRAY_MU
OBJ_MASS = 0.5
OBJ_INERTIA = OBJ_MASS * (OBJ_W**2 + OBJ_H**2) / 12.0

# simulation parameters
SIM_DT = 0.001     # simulation timestep (s)
MPC_DT = 0.1       # lookahead timestep of the controller
MPC_STEPS = 12     # number of timesteps to lookahead
SQP_ITER = 5       # number of iterations for the SQP solved by the controller
PLOT_PERIOD = 100  # update plot every PLOT_PERIOD timesteps
CTRL_PERIOD = 100  # generate new control signal every CTRL_PERIOD timesteps
RECORD_PERIOD = 10
DURATION = 5.0     # duration of trajectory (s)

ns_ee = 6  # num EE states
ns_q = 8   # num joint states
ni = 4     # num inputs
nc_eq = 0
nc_ineq = 7  # num inequality constraints
nv = ni      # num opt vars
nc = nc_eq + nc_ineq  # num constraints

# MPC weights
Q = np.diag([0, 0, 0, 0, 0.01, 0.01, 0.01, 0.01])
W = np.diag([1, 1, 0.1, 0, 0, 0])
R = 0.01 * np.eye(ni)

# lifted weight matrices
Ibar = np.eye(MPC_STEPS)
Qbar = np.kron(Ibar, Q)
Wbar = np.kron(Ibar, W)
Rbar = np.kron(Ibar, R)


def skew1(x):
    return np.array([[0, -x], [x, 0]])


def hessian(f, argnums=0):
    return jax.jacfwd(jax.jacrev(f, argnums=argnums), argnums=argnums)


def main():
    if TRAY_W < TRAY_MU * TRAY_H:
        print('warning: w < μh')

    N = int(DURATION / SIM_DT) + 1
    N_record = int(DURATION / (SIM_DT * RECORD_PERIOD))

    p_te_e = np.array([0, 0.05 + TRAY_H])
    p_oe_e = p_te_e + np.array([0, TRAY_H + OBJ_H])

    model = FourInputModel(l1=L1, l2=L2, vel_lim=VEL_LIM, acc_lim=ACC_LIM)

    ts = SIM_DT * np.arange(N_record)
    us = np.zeros((N_record, ni))
    P_ew_ws = np.zeros((N_record, 3))
    P_ew_wds = np.zeros((N_record, 3))
    V_ew_ws = np.zeros((N_record, 3))
    P_tw_ws = np.zeros((N_record, 3))
    p_te_es = np.zeros((N_record, 2))
    ineq_cons = np.zeros((N_record, nc_ineq))

    p_te_es[0, :] = p_te_e

    # state of joints
    q0 = np.array([0, 0, 0.25*np.pi, -0.25*np.pi])
    dq0 = np.zeros(ni)
    X_q = np.concatenate((q0, dq0))

    P_ew_w = model.ee_position(X_q)
    V_ew_w = model.ee_velocity(X_q)
    P_ew_ws[0, :] = P_ew_w
    V_ew_ws[0, :] = V_ew_w

    # physics simulation
    sim = PymunkSimulationTrayBalance(SIM_DT, gravity=-GRAVITY)
    sim.add_robot(model, q0, TRAY_W, TRAY_MU)

    # tray
    tray_body = pymunk.Body(mass=MASS, moment=INERTIA)
    tray_body.position = tuple(P_ew_w[:2] + p_te_e)
    tray_corners = [(-RADIUS, TRAY_H), (-RADIUS, -TRAY_H), (RADIUS, -TRAY_H),
                    (RADIUS, TRAY_H)]
    tray = pymunk.Poly(tray_body, tray_corners, radius=0)
    tray.facecolor = (0.25, 0.5, 1, 1)
    tray.friction = TRAY_MU
    tray.collision_type = 1
    sim.space.add(tray.body, tray)

    # object on top of the tray
    obj_body = pymunk.Body(mass=OBJ_MASS, moment=OBJ_INERTIA)
    obj_body.position = tuple(P_ew_w[:2] + p_oe_e)
    obj_corners = [(-OBJ_W, OBJ_H), (-OBJ_W, -OBJ_H), (OBJ_W, -OBJ_H),
                   (OBJ_W, OBJ_H)]
    obj = pymunk.Poly(obj_body, obj_corners, radius=0)
    obj.facecolor = (0.5, 0.5, 0.5, 1)
    obj.friction = OBJ_MU / TRAY_MU  # so that mu with tray = OBJ_MU
    obj.collision_type = 1
    # sim.space.add(obj.body, obj)

    # reference trajectory
    trajectory = trajectories.Point(P_ew_w[:2] + np.array([1, -1]))
    P_ew_wds[:, :2], _, _ = trajectory.sample(ts)
    # P_ew_wds[:, 2] = -np.pi / 2

    # rendering
    goal_renderer = plotting.PointRenderer(P_ew_wds[-1, :2], color='r')
    sim_renderer = PymunkRenderer(sim.space, sim.markers)
    renderers = [goal_renderer, sim_renderer]
    video = plotting.Video(name='tray_balance_mm.mp4', fps=1./(SIM_DT*PLOT_PERIOD))
    plotter = plotting.RealtimePlotter(renderers, video=video)
    plotter.start()  # TODO for some reason setting grid=True messes up the base rendering

    def objective_unrolled(X_q_0, X_ee_d, var):
        """Unroll the objective over n timesteps."""
        obj = 0
        X_q = X_q_0

        for i in range(MPC_STEPS):
            u = var[i*nv:(i+1)*nv]
            X_q = model.step_unconstrained(X_q, u, MPC_DT)
            X_ee = model.ee_state(X_q)
            e = X_ee_d[i*ns_ee:(i+1)*ns_ee] - X_ee  # TODO this may also break down with angle wrapping
            obj = obj + 0.5 * (e @ W @ e + X_q @ Q @ X_q + u @ R @ u)

        return obj

    def error_unrolled(X_q_0, X_ee_d, var):
        """Unroll the pose error over the time horizon."""
        X_q = X_q_0
        e = jnp.zeros(MPC_STEPS * ns_ee)

        # TODO can we make this more efficient?
        for i in range(MPC_STEPS):
            u = var[i*nv:(i+1)*nv]
            X_q = model.step_unconstrained(X_q, u, MPC_DT)
            X_ee = model.ee_state(X_q)
            ei = X_ee_d[i*ns_ee:(i+1)*ns_ee] - X_ee
            e = jax.ops.index_update(e, jax.ops.index[i*ns_ee:(i+1)*ns_ee], ei)
        return e

    def ineq_constraints(X_ee, a_ee, jnp=jnp):
        """Calculate inequality constraints for a single timestep."""
        θ_ew, dθ_ew = X_ee[2], X_ee[5]
        a_ew_w, ddθ_ew = a_ee[:2], a_ee[2]
        R_ew = jnp.array([[ jnp.cos(θ_ew), jnp.sin(θ_ew)],
                          [-jnp.sin(θ_ew), jnp.cos(θ_ew)]])
        S1 = skew1(1)
        g = jnp.array([0, GRAVITY])

        α1, α2 = MASS * R_ew @ (a_ew_w+g) + MASS * (ddθ_ew*S1 - dθ_ew**2*jnp.eye(2)) @ p_te_e
        # α1, α2 = OBJ_MASS * R_ew @ (a_ew_w + g) + OBJ_MASS * (ddθ_ew*S1 - dθ_ew**2*jnp.eye(2)) @ p_oe_e
        α3 = INERTIA * ddθ_ew

        # NOTE: this is written to be >= 0
        # h1 = TRAY_MU*α2 - jnp.abs(α1)
        h1a = TRAY_MU*α2 + α1
        h1b = TRAY_MU*α2 - α1
        h2 = α2

        w1 = TRAY_W
        w2 = TRAY_W
        # h3 = α3 + w1 * α2 - TRAY_H * jnp.abs(α1)
        h3a = α3 + w1 * α2 + TRAY_H * α1
        h3b = α3 + w1 * α2 - TRAY_H * α1

        # h4 = -α3 + w2 * α2 - TRAY_H * jnp.abs(α1)
        h4a = -α3 + w2 * α2 + TRAY_H * α1
        h4b = -α3 + w2 * α2 - TRAY_H * α1
        # h3 = 1
        # h4 = 1

        # return jnp.array([h1, h2, h3, h4])
        return jnp.array([h1a, h1b, h2, h3a, h3b, h4a, h4b])

    def ineq_constraints_unrolled(X_q_0, X_ee_d, var):
        """Unroll the inequality constraints over the time horizon."""
        # var is now just the lifted joint acceleration inputs
        X_q = X_q_0
        ineq_con = jnp.zeros(MPC_STEPS * nc_ineq)

        for i in range(MPC_STEPS):
            u = var[i*nv:(i+1)*nv]

            X_ee = model.ee_state(X_q)
            a_ee = model.ee_acceleration(X_q, u)

            ineq_coni = ineq_constraints(X_ee, a_ee)
            ineq_con = jax.ops.index_update(ineq_con, jax.ops.index[i*nc_ineq:(i+1)*nc_ineq], ineq_coni)
            X_q = model.step_unconstrained(X_q, u, MPC_DT)
        return ineq_con

    err_jac = jax.jit(jax.jacfwd(error_unrolled, argnums=2))

    def obj_hess_jac(X_q_0, X_ee_d, var):
        """Calculate objective Hessian and Jacobian."""
        # TODO: not currently accounting for penalty on state x
        e = error_unrolled(X_q_0, X_ee_d, var)
        dedu = err_jac(X_q_0, X_ee_d, var)
        u = var

        # Jacobian
        g = e.T @ Wbar @ dedu + u.T @ Rbar

        # (Approximate) Hessian
        H = dedu.T @ Wbar @ dedu + Rbar

        return H, g

    # Construct the SQP controller
    lbA = np.zeros(MPC_STEPS * nc)
    ubA = np.infty * np.ones(MPC_STEPS * nc)
    con_fun = jax.jit(ineq_constraints_unrolled)
    con_jac = jax.jit(jax.jacfwd(ineq_constraints_unrolled, argnums=2))
    con_sparsity_mask = np.kron(np.tril(np.ones((MPC_STEPS, MPC_STEPS))), np.ones((nc, nv)))
    constraints = sqp.Constraints(con_fun, con_jac, lbA, ubA, nz_idx=np.nonzero(con_sparsity_mask))

    lb = -ACC_LIM * np.ones(MPC_STEPS * nv)
    ub =  ACC_LIM * np.ones(MPC_STEPS * nv)
    bounds = sqp.Bounds(lb, ub)

    controller = sqp.SQP(nv*MPC_STEPS, nc*MPC_STEPS, jax.jit(obj_hess_jac), constraints,
                         bounds, num_wsr=300, num_iter=SQP_ITER, verbose=False,
                         solver="qpoases")

    # controller = sqp.SQP(nv*MPC_STEPS, nc*MPC_STEPS, jax.jit(obj_hess_jac), constraints,
    #                      bounds, num_iter=SQP_ITER, verbose=False, solver="osqp")

    for i in range(N - 1):
        t = i * SIM_DT

        if i % CTRL_PERIOD == 0:
            t_sample = np.minimum(t + MPC_DT*np.arange(MPC_STEPS), DURATION)
            pd, vd, _ = trajectory.sample(t_sample, flatten=False)
            z = np.zeros((MPC_STEPS, 1))
            X_ee_d = np.hstack((pd, z, vd, z)).flatten()
            # -0.5*np.pi*np.ones((MPC_STEPS, 1))

            var = controller.solve(X_q, X_ee_d)
            u = var[:ni]  # joint acceleration input
            sim.command_acceleration(u)

        # integrate the system
        X_q = np.concatenate(sim.step())

        if i % RECORD_PERIOD == 0:
            idx = i // RECORD_PERIOD
            P_ew_w = model.ee_position(X_q)
            V_ew_w = model.ee_velocity(X_q)

            # NOTE: calculating these quantities is fairly expensive
            X_ee = model.ee_state(X_q)
            a_ee = model.ee_acceleration(X_q, u)
            ineq_cons[idx+1, :] = ineq_constraints(X_ee, a_ee, jnp=np)

            # tray position is (ideally) a constant offset from EE frame
            θ_ew = P_ew_w[2]
            R_we = util.rotation_matrix(θ_ew)
            P_tw_w = np.array([tray.body.position.x, tray.body.position.y, tray.body.angle])

            # record
            us[idx, :] = u
            P_ew_ws[idx+1, :] = P_ew_w
            V_ew_ws[idx+1, :] = V_ew_w
            P_tw_ws[idx+1, :] = P_tw_w
            p_te_es[idx+1, :] = R_we.T @ (P_tw_w[:2] - P_ew_w[:2])

        if i % PLOT_PERIOD == 0:
            # break early if plot window is closed
            try:
                plotter.update()
            except TclError:
                break

        if np.linalg.norm(P_ew_wds[0, :2] - P_ew_w[:2]) < 0.01:
            print("Position within 1 cm. Stopping.")
            break
    plotter.done()

    controller.benchmark.print_stats()

    # v_te_es = (p_te_es[1:] - p_te_es[:-1]) / SIM_DT
    # v_te_es_smooth = np.zeros_like(v_te_es)
    # v_te_es_smooth[:, 0] = np.convolve(v_te_es[:, 0], np.ones(100) / 100, 'same')
    # v_te_es_smooth[:, 1] = np.convolve(v_te_es[:, 1], np.ones(100) / 100, 'same')

    idx = i // RECORD_PERIOD

    plt.figure()
    plt.plot(ts[1:idx], P_ew_wds[1:idx, 0], label='$x_d$', color='b', linestyle='--')
    plt.plot(ts[1:idx], P_ew_wds[1:idx, 1], label='$y_d$', color='r', linestyle='--')
    plt.plot(ts[1:idx], P_ew_ws[1:idx, 0],  label='$x$', color='b')
    plt.plot(ts[1:idx], P_ew_ws[1:idx, 1],  label='$y$', color='r')
    # plt.plot(ts[:i], P_tw_ws[:i, 0],  label='$t_x$')
    # plt.plot(ts[:i], P_tw_ws[:i, 1],  label='$t_y$')
    plt.grid()
    plt.legend()
    plt.xlabel('Time (s)')
    plt.ylabel('Position')
    plt.title('End effector position')

    plt.figure()
    plt.plot(ts, p_te_es[:, 0], label='$x$', color='b')
    plt.plot(ts, p_te_es[:, 1], label='$y$', color='r')
    plt.grid()
    plt.legend()
    plt.xlabel('Time (s)')
    plt.ylabel('$p^{te}_e$')
    plt.title('$p^{te}_e$')

    plt.figure()
    plt.plot(ts, p_te_e[0] - p_te_es[:, 0], label='$x$', color='b')
    plt.plot(ts, p_te_e[1] - p_te_es[:, 1], label='$y$', color='r')
    plt.grid()
    plt.legend()
    plt.xlabel('Time (s)')
    plt.ylabel('$p^{te}_e$ error')
    plt.title('$p^{te}_e$ error')

    # plt.figure()
    # plt.plot(ts[:N-1], v_te_es[:, 0], label='$v_x$', color='b')
    # plt.plot(ts[:N-1], v_te_es[:, 1], label='$v_y$', color='r')
    # plt.plot(ts[:N-1], v_te_es_smooth[:, 0], label='$v_x$ (smooth)')
    # plt.plot(ts[:N-1], v_te_es_smooth[:, 1], label='$v_y$ (smooth)')
    # plt.grid()
    # plt.legend()
    # plt.xlabel('Time (s)')
    # plt.ylabel('$v^{te}_e$')
    # plt.title('$v^{te}_e$')

    plt.figure()
    for j in range(nc_ineq):
        plt.plot(ts[:idx], ineq_cons[:idx, j], label=f'$h_{j+1}$')
    # plt.plot(ts[:N], ineq_cons[:, 1], label='$h_2$')
    # plt.plot(ts[:N], ineq_cons[:, 2], label='$h_3$')
    # plt.plot(ts[:N], ineq_cons[:, 3], label='$h_4$')
    plt.plot([0, ts[idx]], [0, 0], color='k')
    plt.grid()
    plt.legend()
    plt.xlabel('Time (s)')
    plt.ylabel('Value')
    plt.title('Inequality constraints')

    plt.figure()
    plt.plot(ts, us[:, 0], label='u1')
    plt.plot(ts, us[:, 1], label='u2')
    plt.plot(ts, us[:, 2], label='u3')
    plt.plot(ts, us[:, 3], label='u4')
    plt.grid()
    plt.legend()
    plt.xlabel('Time (s)')
    plt.ylabel('Commanded joint acceleration')
    plt.title('Acceleration commands')

    plt.show()


if __name__ == '__main__':
    main()
