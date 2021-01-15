import numpy as np
import pymunk
import pymunk.matplotlib_util
import matplotlib.pyplot as plt

from mm2d.simulations import PymunkSimulationVelocity, PymunkSimulationTorque
from mm2d import models, control, plotter
from mm2d import trajectory as trajectories
from mm2d.util import rms

import IPython


# sim parameters
DT = 0.001         # simulation timestep (s)
PLOT_PERIOD = 100  # update plot every PLOT_PERIOD timesteps
CTRL_PERIOD = 100  # generate new control signal every CTRL_PERIOD timesteps

DURATION = 5.0  # duration of trajectory (s)


def main():
    N = int(DURATION / DT) + 1

    model = models.ThreeInputModel(output_idx=[0, 1])

    ts = DT * np.arange(N)
    q0 = np.array([0, np.pi/4.0, -np.pi/4.0])
    p0 = model.forward(q0)

    sim = PymunkSimulationTorque(DT, iterations=10)
    sim.add_robot(model, q0)

    box_body = pymunk.Body()
    box_body.position = (p0[0], p0[1] + 0.1)
    box_corners = [(-0.2, 0.05), (-0.2, -0.05), (0.2, -0.05), (0.2, 0.05)]
    box = pymunk.Poly(box_body, box_corners, radius=0.01)
    box.mass = 0.5
    box.friction = 0.75
    # sim.space.add(box.body, box)

    n_balls = 20
    ball_r = 0.1
    ball_x = np.random.random(n_balls) * 8 - 2
    ball_y = np.random.random(n_balls) * 3 + 2
    for i in range(n_balls):
        body = pymunk.Body()
        body.position = (ball_x[i], ball_y[i])
        ball = pymunk.Circle(body, ball_r, (0, 0))
        ball.mass = 0.1
        ball.color = (255, 0, 0, 255)
        ball.friction = 0.5
        sim.space.add(body, ball)

    W = 0.01 * np.eye(model.ni)
    K = np.eye(model.no)
    controller = control.DiffIKController(model, W, K, DT, model.vel_lim,
                                          model.acc_lim)

    # timescaling = trajectories.QuinticTimeScaling(DURATION)
    # trajectory = trajectories.Sine(p0, 2, 0.5, 1, timescaling, DURATION)
    trajectory = trajectories.Point(p0)

    ps = np.zeros((N, model.no))
    pds = np.zeros((N, model.no))

    robot_renderer = plotter.ThreeInputRenderer(model, q0)
    box_renderer = plotter.PolygonRenderer(np.array(box.body.position),
                                           box.body.angle,
                                           np.array(box_corners))
    # trajectory_renderer = plotter.TrajectoryRenderer(trajectory, ts)
    plot = plotter.RealtimePlotter([robot_renderer, box_renderer])
    plot.start(grid=True)

    plt.ion()
    fig = plt.figure()
    ax = plt.gca()

    # video = plotter.Video(name='balls.mp4', fps=1./(PLOT_PERIOD * DT))
    # video.setup(fig)

    ax.set_xlabel('x (m)')
    ax.set_ylabel('y (m)')
    ax.set_xlim([-1, 6])
    ax.set_ylim([-1, 2])

    ax.set_aspect('equal')

    options = pymunk.matplotlib_util.DrawOptions(ax)
    options.flags = pymunk.SpaceDebugDrawOptions.DRAW_SHAPES

    sim.space.debug_draw(options)
    fig.canvas.draw()
    fig.canvas.flush_events()

    # video.writer.grab_frame()

    q = q0
    dq = np.zeros(3)
    u = np.zeros(3)
    pd, vd, ad = trajectory.sample(0, flatten=True)

    ps[0, :] = p0
    pds[0, :] = pd[:model.no]

    kp = 5
    kd = 3
    ki = 0.1
    ddqd = np.zeros(3)
    dqd = np.zeros(3)
    qd = q0

    e_sum = 0

    for i in range(N - 1):
        t = ts[i]

        # controller
        if i % CTRL_PERIOD == 0:
            pd, vd, ad = trajectory.sample(t, flatten=True)
            u = controller.solve(q, dq, pd, vd)
            # sim.command_velocity(u)

            # torque control law
            e = qd - q
            e_sum += CTRL_PERIOD * DT * e
            α = ddqd + kp * e + kd * (dqd - dq) + ki * e_sum
            tau = model.calc_torque(q, dq, α)
            sim.command_torque(tau)

        # step the sim
        q, dq = sim.step()

        p = model.forward(q)
        ps[i+1, :] = p
        pds[i+1, :] = pd[:model.no]

        if i % PLOT_PERIOD == 0:
            box_renderer.set_state(np.array(box.body.position), box.body.angle)
            robot_renderer.set_state(q)

            ax.cla()
            ax.set_xlim([-1, 6])
            ax.set_ylim([-1, 2])

            sim.space.debug_draw(options)
            fig.canvas.draw()
            fig.canvas.flush_events()
            # video.writer.grab_frame()

            plot.update()
    plot.done()
    # video.writer.finish()

    xe = pds[1:, 0] - ps[1:, 0]
    ye = pds[1:, 1] - ps[1:, 1]
    print('RMSE(x) = {}'.format(rms(xe)))
    print('RMSE(y) = {}'.format(rms(ye)))


if __name__ == '__main__':
    main()
