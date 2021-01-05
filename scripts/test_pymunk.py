import numpy as np

from mm2d.simulations import PymunkSimulation
from mm2d import models, control, plotter
from mm2d import trajectory as trajectories
from mm2d.util import rms

import IPython


# robot parameters
Lx = 0
Ly = 0
L1 = 1
L2 = 1
VEL_LIM = 1
ACC_LIM = 1

DT = 0.001         # simulation timestep (s)
PLOT_PERIOD = 100  # update plot every PLOT_PERIOD timesteps
CTRL_PERIOD = 100  # generate new control signal every CTRL_PERIOD timesteps

DURATION = 10.0  # duration of trajectory (s)


def main():
    N = int(DURATION / DT) + 1

    model = models.ThreeInputKinematicModel(VEL_LIM, ACC_LIM, l1=L1, l2=L2,
                                            output_idx=[0, 1])

    ts = DT * np.arange(N)
    q0 = np.array([0, np.pi/4.0, -np.pi/4.0])
    p0 = model.forward(q0)

    sim = PymunkSimulation(DT)
    sim.add_robot(model, q0)

    W = 0.01 * np.eye(model.ni)
    K = np.eye(model.no)
    controller = control.DiffIKController(model, W, K, DT, VEL_LIM, ACC_LIM)

    timescaling = trajectories.QuinticTimeScaling(DURATION)
    trajectory = trajectories.Sine(p0, 2, 0.5, 1, timescaling, DURATION)

    ps = np.zeros((N, model.no))
    pds = np.zeros((N, model.no))

    robot_renderer = plotter.ThreeInputRenderer(model, q0)
    trajectory_renderer = plotter.TrajectoryRenderer(trajectory, ts)
    plot = plotter.RealtimePlotter([robot_renderer, trajectory_renderer])
    plot.start(grid=True)

    q = q0
    dq = np.zeros(3)
    u = np.zeros(3)
    pd, vd, ad = trajectory.sample(0, flatten=True)

    ps[0, :] = p0
    pds[0, :] = pd[:model.no]

    for i in range(N - 1):
        t = ts[i]

        # controller
        if i % CTRL_PERIOD == 0:
            pd, vd, ad = trajectory.sample(t, flatten=True)
            u = controller.solve(q, dq, pd, vd)
            sim.command_velocity(u)

        # step the sim
        q, dq = sim.step()

        p = model.forward(q)
        ps[i+1, :] = p
        pds[i+1, :] = pd[:model.no]

        if i % PLOT_PERIOD == 0:
            robot_renderer.set_state(q)
            plot.update()
    plot.done()

    xe = pds[1:, 0] - ps[1:, 0]
    ye = pds[1:, 1] - ps[1:, 1]
    print('RMSE(x) = {}'.format(rms(xe)))
    print('RMSE(y) = {}'.format(rms(ye)))


if __name__ == '__main__':
    main()
