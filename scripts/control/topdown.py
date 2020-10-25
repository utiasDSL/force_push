#!/usr/bin/env python
import numpy as np
from numpy.linalg import norm
import matplotlib.pyplot as plt

from mm2d.model import TopDownHolonomicModel
from mm2d import obstacle, plotter
from mm2d import controller as control
from mm2d import trajectory as trajectories
from mm2d.util import rms

import IPython


# robot parameters
L1 = 1
L2 = 1
VEL_LIM = 1
ACC_LIM = 1

DT = 0.05        # timestep (s)
DURATION = 30.0  # duration of trajectory (s)

NUM_WSR = 100     # number of working set recalculations


def unit(a):
    return a / np.linalg.norm(a)


def main():
    N = int(DURATION / DT) + 1

    model = TopDownHolonomicModel(L1, L2, VEL_LIM, acc_lim=ACC_LIM, output_idx=[0, 1])

    W = 0.1 * np.eye(model.ni)
    K = 2*np.eye(model.no)
    C = 1*np.eye(model.no)
    Cinv = np.eye(model.no)
    controller = control.ConstrainedDiffIKController(model, W, K, DT, VEL_LIM, ACC_LIM)

    ts = np.array([i * DT for i in range(N)])
    qs = np.zeros((N, model.ni))
    dqs = np.zeros((N, model.ni))
    us = np.zeros((N, model.ni))
    ps = np.zeros((N, model.no))
    vs = np.zeros((N, model.no))
    pds = np.zeros((N, model.no))

    # initial state
    q = np.array([0, 0, 0, 0])
    p = model.forward(q)
    dq = np.zeros(model.ni)
    f = np.zeros(2)

    # reference trajectory
    timescaling = trajectories.QuinticTimeScaling(DURATION)
    trajectory = trajectories.PointToPoint(p, p + [1, 0], timescaling, DURATION)

    # obstacle
    pc = np.array([-2., 1.])
    obs = obstacle.Circle(0.5, 1000)

    pg = np.array([5., 0])

    qs[0, :] = q
    ps[0, :] = p
    pds[0, :] = p

    circle_renderer = plotter.CircleRenderer(obs, pc)
    robot_renderer = plotter.TopDownHolonomicRenderer(model, q)
    trajectory_renderer = plotter.TrajectoryRenderer(trajectory, ts)
    plot = plotter.RealtimePlotter([robot_renderer, trajectory_renderer, circle_renderer])
    plot.start(limits=[-5, 10, -5, 10], grid=True)

    for i in range(N - 1):
        t = ts[i]

        # controller
        # pd, vd, ad = trajectory.sample(t, flatten=True)
        # pd = pc

        # experimental controller for aligning and pushing object to a goal
        # point - generates a desired set point pd; the admittance portion
        # doesn't really seem helpful at this point (since we actually *want*
        # to hit/interact with the environment)
        vd = np.zeros(2)
        # a = pg - pc
        # b = pc - p
        # e_norm = np.linalg.norm(pg - pc)
        # r = 0.3   # radius
        # pd = pc - r * unit(pg - pc)
        # closest = pc - (a.dot(b)) * a / a.dot(a)
        # closest_dist = np.linalg.norm(closest - p)
        # r = 0.1  # radius
        # if closest_dist < r:
        #     d = np.sqrt(r**2 - closest_dist**2)
        #     pd = pc - d * unit(pg - closest)
        # else:
        #     pd = p + r * unit(closest - p)
        b = 0.1
        cos_alpha = np.cos(np.pi * 0.25)
        p1 = pc - 0.25 * unit(pg - pc)
        # p2 = pc + obs.r * unit(pg - pc)
        d = np.linalg.norm(p - pc)
        J = model.jacobian(q)

        A = 2*DT*(p - pc).T.dot(J)
        A = A.reshape((model.ni, 1))

        pd = p1

        cos_angle = unit(p - pc).dot(unit(p1 - pc))
        lbA = np.array([obs.r + b - d])

        if cos_angle >= cos_alpha:
            lbA = np.zeros_like(lbA)
            A = np.zeros_like(A)

        # K1 = np.eye(2)
        # K2 = 0.5*np.eye(2)
        #
        # a = 0.5*obs.r
        # cardioid_center = pc - a*unit(pg - pc)
        # r_cardioid = 2*a*(1 - unit(p - pc).dot(unit(p1 - pc)))
        # grad_r_cardioid = -2*a*norm(p1-pc)*( norm(p-pc)*(p1-pc) + 2*(p1-pc).dot(p-pc)*(p-pc) )
        #
        # d_cardioid = norm(p - cardioid_center) - r_cardioid
        # grad_d_cardioid = 2*(p - cardioid_center) - grad_r_cardioid
        #
        # grad_Ua = K1.dot(p - p1)
        # eta = 0.1
        # b = -0.1
        # if d_cardioid <= b:
        #     grad_Ur = eta*(1./b - 1./d_cardioid) / d_cardioid**2 * grad_d_cardioid
        # else:
        #     grad_Ur = np.zeros_like(p)
        # print(d_cardioid)
        # grad_U = grad_Ua + grad_Ur

        # vd = K1.dot(p1 - p)
        # if np.linalg.norm(p2 - p) < obs.r:
        #     vd = K1.dot(p1 - p) - K2.dot(p2 - p)
        # else:
        #     vd = K1.dot(p1 - p)
        # pnext = p + DT * v
        # if np.linalg.norm(pnext - pc) <= obs.r + 0.1:
        # vd = K1.dot(p1 - p) - K2.dot(unit(p2 - p))
        # vd = -unit(grad_U) * 0.3
        # vd = -grad_U

        u = controller.solve(q, dq, pd, vd, A, lbA)
        if np.linalg.norm(pg - pc) < 0.1:
            print('done')
            break

        # step the model
        q, dq = model.step(q, u, DT, dq_last=dq)
        p = model.forward(q)
        v = model.jacobian(q).dot(dq)

        # obstacle interaction
        f, movement = obs.force(pc, p)
        pc += movement

        # record
        us[i, :] = u
        dqs[i+1, :] = dq
        qs[i+1, :] = q
        ps[i+1, :] = p
        pds[i+1, :] = pd[:model.no]
        vs[i+1, :] = v

        # render
        robot_renderer.set_state(q)
        circle_renderer.set_state(pc)
        plot.update()
    plot.done()

    xe = pds[1:, 0] - ps[1:, 0]
    ye = pds[1:, 1] - ps[1:, 1]
    print('RMSE(x) = {}'.format(rms(xe)))
    print('RMSE(y) = {}'.format(rms(ye)))

    plt.figure()
    plt.plot(ts, pds[:, 0], label='$x_d$', color='b', linestyle='--')
    plt.plot(ts, pds[:, 1], label='$y_d$', color='r', linestyle='--')
    plt.plot(ts, ps[:, 0],  label='$x$', color='b')
    plt.plot(ts, ps[:, 1],  label='$y$', color='r')
    plt.grid()
    plt.legend()
    plt.xlabel('Time (s)')
    plt.ylabel('Position')
    plt.title('End effector position')

    plt.figure()
    plt.plot(ts, dqs[:, 0], label='$\\dot{q}_x$')
    plt.plot(ts, dqs[:, 1], label='$\\dot{q}_1$')
    plt.plot(ts, dqs[:, 2], label='$\\dot{q}_2$')
    plt.grid()
    plt.legend()
    plt.title('Actual joint velocity')
    plt.xlabel('Time (s)')
    plt.ylabel('Velocity')

    plt.figure()
    plt.plot(ts, us[:, 0], label='$u_x$')
    plt.plot(ts, us[:, 1], label='$u_1$')
    plt.plot(ts, us[:, 2], label='$u_2$')
    plt.grid()
    plt.legend()
    plt.title('Commanded joint velocity')
    plt.xlabel('Time (s)')
    plt.ylabel('Velocity')

    plt.figure()
    plt.plot(ts, qs[:, 0], label='$q_x$')
    plt.plot(ts, qs[:, 1], label='$q_1$')
    plt.plot(ts, qs[:, 2], label='$q_2$')
    plt.grid()
    plt.legend()
    plt.title('Joint positions')
    plt.xlabel('Time (s)')
    plt.ylabel('Joint positions')

    plt.show()


if __name__ == '__main__':
    main()
