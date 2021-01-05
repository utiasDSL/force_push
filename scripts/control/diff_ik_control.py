#!/usr/bin/env python
import numpy as np
import matplotlib.pyplot as plt

from mm2d import models, control, plotter
from mm2d import trajectory as trajectories
from mm2d.util import rms

import IPython


# robot parameters
L1 = 1
L2 = 1
VEL_LIM = 1
ACC_LIM = 1

DT = 0.1        # timestep (s)
DURATION = 10.0  # duration of trajectory (s)


def main():
    N = int(DURATION / DT) + 1

    model = models.ThreeInputKinematicModel(VEL_LIM, ACC_LIM, l1=L1, l2=L2,
                                            output_idx=[0, 1])

    W = 0.1 * np.eye(model.ni)
    K = np.eye(model.no)
    controller = control.DiffIKController(model, W, K, DT, VEL_LIM, ACC_LIM)

    ts = DT * np.arange(N)
    qs = np.zeros((N, model.ni))
    dqs = np.zeros((N, model.ni))
    us = np.zeros((N, model.ni))
    ps = np.zeros((N, model.no))
    vs = np.zeros((N, model.no))
    pds = np.zeros((N, model.no))

    q0 = np.array([0, np.pi/4.0, -np.pi/4.0])
    p0 = model.forward(q0)

    # reference trajectory
    # trajectory = Line(p0, v0=np.zeros(2), a=np.array([0.01, 0]))
    # timescaling = trajectories.QuinticTimeScaling(DURATION)
    timescaling = trajectories.TrapezoidalTimeScalingV(0.15, DURATION)
    # timescaling = trajectories.TrapezoidalTimeScalingA(0.1, DURATION)

    # trajectory = trajectories.PointToPoint(p0, p0 + [1, 0], timescaling, DURATION)
    trajectory = trajectories.Sine(p0, 2, 0.5, 1, timescaling, DURATION)
    # trajectory2 = PointToPoint(p0 + [1, 0], p0 + [2, 0], timescaling, 0.5*DURATION)
    # trajectory = Chain([trajectory1, trajectory2])

    # points = np.array([p0, p0 + [1, 1], p0 + [2, -1], p0 + [3, 0]])
    # trajectory = CubicBezier(points, timescaling, DURATION)
    # trajectory = trajectories.Circle(p0, 0.5, timescaling, DURATION)
    # points = np.array([p0, p0 + [1, 0], p0 + [1, -1], p0 + [0, -1], p0])
    # trajectory = Polygon(points, v=0.4)

    # pref, vref, aref = trajectory.sample(ts)
    # plt.figure()
    # plt.plot(ts, pref[:, 0], label='$p$')
    # plt.plot(ts, vref[:, 0], label='$v$')
    # plt.plot(ts, aref[:, 0], label='$a$')
    # plt.grid()
    # plt.legend()
    # plt.title('EE reference trajectory')
    # plt.xlabel('Time (s)')
    # plt.ylabel('Reference signal')
    # plt.show()

    q = q0
    p = p0
    dq = np.zeros(model.ni)
    qs[0, :] = q0
    ps[0, :] = p0
    pds[0, :] = p0

    robot_renderer = plotter.ThreeInputRenderer(model, q0)
    trajectory_renderer = plotter.TrajectoryRenderer(trajectory, ts)
    plot = plotter.RealtimePlotter([robot_renderer, trajectory_renderer])
    plot.start(grid=True)

    for i in range(N - 1):
        t = ts[i]

        # controller
        pd, vd, ad = trajectory.sample(t, flatten=True)
        u = controller.solve(q, dq, pd, vd)

        # step the model
        q, dq = model.step(q, u, DT, dq_last=dq)
        p = model.forward(q)
        v = model.jacobian(q).dot(dq)

        # record
        us[i, :] = u
        dqs[i+1, :] = dq
        qs[i+1, :] = q
        ps[i+1, :] = p
        pds[i+1, :] = pd[:model.no]
        vs[i+1, :] = v

        # render
        robot_renderer.set_state(q)
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
    plt.plot(ts, vs[:, 0],  label='$v_x$', color='b')
    plt.plot(ts, vs[:, 1],  label='$v_y$', color='r')
    plt.grid()
    plt.legend()
    plt.xlabel('Time (s)')
    plt.ylabel('Velocity (m/s)')
    plt.title('End effector velocity')

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
