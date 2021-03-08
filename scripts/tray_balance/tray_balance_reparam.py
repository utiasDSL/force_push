#!/usr/bin/env python
"""This script investigates reparameterizing the tray state. Instead of pose
and velocity, we experiment with the position and velocity of the contact
points. This ultimately did not improve the theory and made the math and
implementation more complicated."""
import time
import jax.numpy as jnp
import jax
import numpy as np
import matplotlib.pyplot as plt
import pymunk
import pymunk.matplotlib_util

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
ACC_LIM = 50

# tray parameters
GRAVITY = 9.81
RADIUS = 0.5
MASS = 0.5
# INERTIA = 0.25*MASS*RADIUS**2
INERTIA = MASS * (3*RADIUS**2 + 0.1**2) / 12.0
TRAY_MU = 0.75
TRAY_W = 0.2
TRAY_H = 0.3
TRAY_G_MAT = np.diag([MASS, MASS, INERTIA])

OBJ_W = 0.1
OBJ_H = 0.1
OBJ_MU = TRAY_MU
OBJ_MASS = 0.5
OBJ_INERTIA = OBJ_MASS * (OBJ_W**2 + OBJ_H**2) / 12.0

# simulation parameters
SIM_DT = 0.001         # simulation timestep (s)
MPC_DT = 0.1      # lookahead timestep of the controller
MPC_STEPS = 8
PLOT_PERIOD = 100  # update plot every PLOT_PERIOD timesteps
CTRL_PERIOD = 100  # generate new control signal every CTRL_PERIOD timesteps
DURATION = 5.0     # duration of trajectory (s)

ns_ee = 6  # num EE states
ns_q = 8   # num joint states
ni = 4     # num inputs
nc_eq = 0
nc_ineq = 3  # nf force + ni velocity + 1 constraint on q2
nv = ni      # num opt vars
nc = nc_eq + nc_ineq  # num constraints

# MPC weights
Q = np.diag([0, 0, 0, 0, 0.01, 0.01, 0.01, 0.01])
W = np.diag([1, 1, 0.1, 0, 0, 0])
R = 0.01 * np.eye(ni)


def skew1(x):
    return np.array([[0, -x], [x, 0]])


def hessian(f, argnums=0):
    return jax.jacfwd(jax.jacrev(f, argnums=argnums), argnums=argnums)


def main():
    # if TRAY_W < TRAY_MU * TRAY_H:
    #     print('must have w >= μh')
    #     return

    N = int(DURATION / SIM_DT) + 1

    p_te_e = np.array([0, 0.05 + TRAY_H])
    p_oe_e = p_te_e + np.array([0, TRAY_H + OBJ_H])

    r_c1e_e = np.array([-TRAY_W, 0.05])
    r_c2e_e = np.array([TRAY_W, 0.05])

    r_tc1_t = np.array([TRAY_W, TRAY_H])
    r_tc2_t = np.array([-TRAY_W, TRAY_H])
    r_t2t1_t = -r_tc2_t + r_tc1_t

    model = FourInputModel(l1=L1, l2=L2, vel_lim=VEL_LIM, acc_lim=ACC_LIM)

    ts = SIM_DT * np.arange(N)
    us = np.zeros((N, ni))
    P_ew_ws = np.zeros((N, 3))
    P_ew_wds = np.zeros((N, 3))
    V_ew_ws = np.zeros((N, 3))
    P_tw_ws = np.zeros((N, 3))
    p_te_es = np.zeros((N, 2))

    ineq_cons = np.zeros((N, 3))

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
    sim = PymunkSimulationTrayBalance(SIM_DT, gravity=-GRAVITY, iterations=30)
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

    obj_body = pymunk.Body(mass=OBJ_MASS, moment=OBJ_INERTIA)
    obj_body.position = tuple(P_ew_w[:2] + p_oe_e)
    obj_corners = [(-OBJ_W, OBJ_H), (-OBJ_W, -OBJ_H), (OBJ_W, -OBJ_H),
                   (OBJ_W, OBJ_H)]
    obj = pymunk.Poly(obj_body, obj_corners, radius=0)
    obj.facecolor = (0.5, 0.5, 0.5, 1)
    obj.friction = OBJ_MU / TRAY_MU  # so that mu with tray = OBJ_MU
    obj.collision_type = 1
    # sim.space.add(obj.body, obj)

    # sim.space.add_collision_handler(1, 1).post_solve = tray_cb

    # reference trajectory
    # timescaling = trajectories.QuinticTimeScaling(DURATION)
    # trajectory = trajectories.Circle(P_ew_w[:2], 0.25, timescaling, DURATION)
    trajectory = trajectories.Point(P_ew_w[:2] + np.array([1, -1]))
    P_ew_wds[:, :2], _, _ = trajectory.sample(ts)
    # P_ew_wds[:, 2] = -np.pi / 2

    # rendering
    goal_renderer = plotting.PointRenderer(P_ew_wds[-1, :2], color='r')
    sim_renderer = PymunkRenderer(sim.space, sim.markers)
    # trajectory_renderer = plotting.TrajectoryRenderer(trajectory, ts)
    renderers = [goal_renderer, sim_renderer]
    video = plotting.Video(name='tray_balance.mp4', fps=1./(SIM_DT*PLOT_PERIOD))
    plotter = plotting.RealtimePlotter(renderers, video=video)
    plotter.start()  # TODO for some reason setting grid=True messes up the base rendering

    # pymunk rendering
    # plt.ion()
    # fig = plt.figure()
    # ax = plt.gca()
    # plt.grid()
    #
    # ax.set_xlabel('x (m)')
    # ax.set_ylabel('y (m)')
    # ax.set_xlim([-1, 6])
    # ax.set_ylim([-1, 2])
    #
    # ax.set_aspect('equal')
    #
    # options = pymunk.matplotlib_util.DrawOptions(ax)
    # options.flags = pymunk.SpaceDebugDrawOptions.DRAW_SHAPES

    def jax_rotation_matrix(θ):
        return jnp.array([[jnp.cos(θ), -jnp.sin(θ)],
                          [jnp.sin(θ),  jnp.cos(θ)]])

    def calc_tray_pose(q):
        """Calculate tray pose in terms of x1, y1, y2."""
        x1, y1, y2 = q
        Δy = y2 - y1
        Δx = jnp.sqrt((2*TRAY_W)**2 - Δy**2)
        θ_tw = jnp.arctan2(Δy, Δx)
        R_wt = jax_rotation_matrix(θ_tw)
        r_c1w_w = jnp.array([x1, y1])
        r_tw_w = r_c1w_w + R_wt @ r_tc1_t
        return jnp.array([r_tw_w[0], r_tw_w[1], θ_tw])

    # def calc_tray_state(P_tw_w, V_tw_w):
    #     """Calculate tray parameters q given the pose."""
    #     # TODO ideally we'd just keep everything in terms of q, but the old
    #     # approach parameterizes the tray by its pose.
    #     # TODO inconsistent notation
    #     r_tw_w, θ_tw = P_tw_w[:2], P_tw_w[2]
    #     R_wt = jnp.array([[jnp.cos(θ_tw), -jnp.sin(θ_tw)],
    #                       [jnp.sin(θ_tw),  jnp.cos(θ_tw)]])
    #     r_t1w_w = r_tw_w - R_wt @ r_tc1_t
    #     r_t2w_w = r_tw_w - R_wt @ r_tc2_t
    #     q = jnp.array([r_t1w_w[0], r_t1w_w[1], r_t2w_w[1]])
    #
    #     return q, dq

    calc_tray_jacobian = jax.jit(jax.jacfwd(calc_tray_pose))

    def calc_tray_potential(q):
        """Calculate tray potential energy."""
        x, y, θ = calc_tray_pose(q)
        return MASS*GRAVITY*y

    calc_g = jax.jit(jax.jacfwd(calc_tray_potential))

    def calc_tray_mass_matrix(q):
        """Calculate tray mass matrix."""
        J = calc_tray_jacobian(q)
        M = J.T @ TRAY_G_MAT @ J
        return M

    def calc_ddR_wt(q, dq, ddq):
        """Calculate the second time derivative of tray's rotation matrix."""
        Δy = q[2] - q[1]
        dΔy = dq[2] - dq[1]
        ddΔy = ddq[2] - ddq[1]
        Δx = jnp.sqrt((2*TRAY_W)**2 - Δy**2)

        θ_tw = jnp.arctan2(Δy, Δx)
        dθ_tw = dΔy / Δx  # TODO possible division by zero
        ddθ_tw = (ddΔy * Δx**2 + dΔy**2 * Δy) / Δx**3

        S1 = skew1(1)
        ddR_wt = (ddθ_tw * S1 - dθ_tw**2) @ jax_rotation_matrix(θ_tw)
        return ddR_wt

    calc_dMdq = jax.jit(jax.jacfwd(calc_tray_mass_matrix))

    def calc_tray_h(q, dq):
        """Calculate h term in dynamics of the form M(q)*ddq + h(q,dq) = f."""
        dMdq = calc_dMdq(q)
        Γ = dMdq - 0.5*dMdq.T
        g = calc_g(q)
        h = dq.dot(Γ).dot(dq) + g
        return h

    def objective_unrolled(X_q_0, X_ee_d, var):
        """Unroll the objective over n timesteps."""
        obj = 0
        X_q = X_q_0

        for i in range(MPC_STEPS):
            u = var[i*nv:(i+1)*nv]
            X_q = model.step_unconstrained(X_q, u, MPC_DT)
            X_ee = model.ee_state(X_q)
            e = X_ee_d[i*ns_ee:(i+1)*ns_ee] - X_ee
            obj = obj + 0.5 * (e @ W @ e + X_q @ Q @ X_q + u @ R @ u)

        return obj

    def ineq_constraints(X_ee, a_ee, jnp=jnp):
        r_ew_w = X_ee[:2]
        v_ew_w = X_ee[3:5]
        θ_ew, dθ_ew = X_ee[2], X_ee[5]
        a_ew_w, ddθ_ew = a_ee[:2], a_ee[2]
        R_ew = jnp.array([[ jnp.cos(θ_ew), jnp.sin(θ_ew)],
                          [-jnp.sin(θ_ew), jnp.cos(θ_ew)]])
        S1 = skew1(1)
        g = jnp.array([0, GRAVITY])

        # calculate tray state from EE state
        # r_c1w_w = r_ew_w + R_ew.T @ r_c1e_e
        # r_c2w_w = r_ew_w + R_ew.T @ r_c2e_e
        # v_c1w_w = v_ew_w + dθ_ew * S1 @ R_ew.T @ r_c1e_e
        # v_c2w_w = v_ew_w + dθ_ew * S1 @ R_ew.T @ r_c2e_e
        # qt = jnp.array([r_c1w_w[0], r_c1w_w[1], r_c2w_w[1]])
        # dqt = jnp.array([v_c1w_w[0], v_c1w_w[1], v_c2w_w[1]])

        # calculate tray dynamics
        # Mt = calc_tray_mass_matrix(qt)
        # ht = calc_tray_h(qt, dqt)
        # Qt = jnp.linalg.solve(Mt, ht)  # TODO bad name: this isn't the generalized forces

        # ddR_we = (ddθ_ew*S1 - dθ_ew**2*jnp.eye(2)) @ R_ew.T
        # ddR_wt = calc_ddR_wt(qt, dqt, Qt)

        # TODO: this will need some further justification
        # a_t1w_w = Qt[:2]
        # a_t2w_w = a_t1w_w + ddR_wt @ r_t2t1_t

        # ddy1 = np.array([0, 1]) @ R_ew @ (a_ew_w + ddR_we @ r_c1e_e)
        # ddy2 = np.array([0, 1]) @ R_ew @ (a_ew_w + ddR_we @ r_c2e_e)

        α1, α2 = MASS * R_ew @ (a_ew_w+g) + MASS * (ddθ_ew*S1 - dθ_ew**2*jnp.eye(2)) @ p_te_e

        # TODO: how to deal with friction under this formulation?
        # h1 = TRAY_MU*jnp.abs(α2) - jnp.abs(α1)
        # h2 = ddy1 + a_t1w_w[1]
        # h3 = ddy2 + a_t2w_w[1]

        # α3 = INERTIA * ddθ_ew
        #
        # h1 = 1
        # h2 = 1
        h1 = TRAY_MU*jnp.abs(α2) - jnp.abs(α1)
        h2 = α2
        h3 = 1

        return jnp.array([h1, h2, h3])

    def ineq_constraints_unrolled(X_q_0, X_ee_d, var):
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

    # Construct the SQP controller
    obj_fun = jax.jit(objective_unrolled)
    obj_jac = jax.jit(jax.jacfwd(objective_unrolled, argnums=2))
    obj_hess = jax.jit(hessian(objective_unrolled, argnums=2))
    objective = sqp.Objective(obj_fun, obj_jac, obj_hess)

    lbA = np.zeros(MPC_STEPS * nc)
    ubA = np.infty * np.ones(MPC_STEPS * nc)
    con_fun = jax.jit(ineq_constraints_unrolled)
    con_jac = jax.jit(jax.jacfwd(ineq_constraints_unrolled, argnums=2))
    constraints = sqp.Constraints(con_fun, con_jac, lbA, ubA)

    lb = -ACC_LIM * np.ones(MPC_STEPS * nv)
    ub =  ACC_LIM * np.ones(MPC_STEPS * nv)
    bounds = sqp.Bounds(lb, ub)

    controller = sqp.SQP(nv*MPC_STEPS, nc*MPC_STEPS, objective, constraints, bounds, num_iter=3, verbose=False)

    for i in range(N - 1):
        t = ts[i+1]

        if i % CTRL_PERIOD == 0:
            t_sample = np.minimum(t + MPC_DT*np.arange(MPC_STEPS), DURATION)
            pd, vd, _ = trajectory.sample(t_sample, flatten=False)
            z = np.zeros((MPC_STEPS, 1))
            X_ee_d = np.hstack((pd, z, vd, z)).flatten()
            # -0.5*np.pi*np.ones((MPC_STEPS, 1))

            # start = time.time()
            var = controller.solve(X_q, X_ee_d)
            # print(time.time() - start)
            u = var[:ni]  # joint acceleration input
            sim.command_acceleration(u)

        # integrate the system
        X_q = np.concatenate(sim.step())
        P_ew_w = model.ee_position(X_q)
        V_ew_w = model.ee_velocity(X_q)

        X_ee = model.ee_state(X_q)
        a_ee = model.ee_acceleration(X_q, u)
        ineq_cons[i+1, :] = ineq_constraints(X_ee, a_ee, jnp=np)
        # if (ineq_con < 0).any():
        #     IPython.embed()

        # tray position is (ideally) a constant offset from EE frame
        θ_ew = P_ew_w[2]
        R_we = util.rotation_matrix(θ_ew)
        P_tw_w = np.array([tray.body.position.x, tray.body.position.y, tray.body.angle])

        # record
        us[i, :] = u
        P_ew_ws[i+1, :] = P_ew_w
        V_ew_ws[i+1, :] = V_ew_w
        P_tw_ws[i+1, :] = P_tw_w
        p_te_es[i+1, :] = R_we.T @ (P_tw_w[:2] - P_ew_w[:2])

        if i % PLOT_PERIOD == 0:
            # ax.cla()
            # ax.set_xlim([-1, 6])
            # ax.set_ylim([-1, 2])
            #
            # sim.space.debug_draw(options)
            # fig.canvas.draw()
            # fig.canvas.flush_events()

            plotter.update()

        if np.linalg.norm(P_ew_w - P_ew_wds[0, :]) < 1e-2:
            break

    plotter.done()

    v_te_es = (p_te_es[1:] - p_te_es[:-1]) / SIM_DT
    v_te_es_smooth = np.zeros_like(v_te_es)
    v_te_es_smooth[:, 0] = np.convolve(v_te_es[:, 0], np.ones(100) / 100, 'same')
    v_te_es_smooth[:, 1] = np.convolve(v_te_es[:, 1], np.ones(100) / 100, 'same')

    np.savez("data", P_tw_ws=P_tw_ws, P_ew_ws=P_ew_ws, ts=ts, P_ew_wd=P_ew_wds[0, :])

    plt.figure()
    plt.plot(ts[:N], P_ew_wds[:, 0], label='$x_d$', color='b', linestyle='--')
    plt.plot(ts[:N], P_ew_wds[:, 1], label='$y_d$', color='r', linestyle='--')
    plt.plot(ts[:N], P_ew_ws[:, 0],  label='$x$', color='b')
    plt.plot(ts[:N], P_ew_ws[:, 1],  label='$y$', color='r')
    plt.plot(ts[:N], P_tw_ws[:, 0],  label='$t_x$')
    plt.plot(ts[:N], P_tw_ws[:, 1],  label='$t_y$')
    plt.grid()
    plt.legend()
    plt.xlabel('Time (s)')
    plt.ylabel('Position')
    plt.title('End effector position')

    plt.figure()
    plt.plot(ts[:N], p_te_es[:, 0], label='$x$', color='b')
    plt.plot(ts[:N], p_te_es[:, 1], label='$y$', color='r')
    plt.grid()
    plt.legend()
    plt.xlabel('Time (s)')
    plt.ylabel('$p^{te}_e$')
    plt.title('$p^{te}_e$')

    plt.figure()
    plt.plot(ts[:N], p_te_e[0] - p_te_es[:, 0], label='$x$', color='b')
    plt.plot(ts[:N], p_te_e[1] - p_te_es[:, 1], label='$y$', color='r')
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
    plt.plot(ts[:N], ineq_cons[:, 0], label='$h_1$')
    plt.plot(ts[:N], ineq_cons[:, 1], label='$h_2$')
    plt.plot(ts[:N], ineq_cons[:, 2], label='$h_3$')
    plt.plot([0, ts[-1]], [0, 0], color='k')
    plt.grid()
    plt.legend()
    plt.xlabel('Time (s)')
    plt.ylabel('Value')
    plt.title('Inequality constraints')

    plt.show()


if __name__ == '__main__':
    main()
