import numpy as np
import pymunk

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

DT = 0.01         # simulation timestep (s)
PLOT_PERIOD = 10  # update plot every PLOT_PERIOD timesteps
CTRL_PERIOD = 10  # generate new control signal every CTRL_PERIOD timesteps

DURATION = 10.0  # duration of trajectory (s)


def main():
    N = int(DURATION / DT) + 1

    model = models.ThreeInputKinematicModel(VEL_LIM, ACC_LIM, l1=L1, l2=L2,
                                            output_idx=[0, 1])

    ts = DT * np.arange(N)
    q0 = np.array([0, np.pi/4.0, -np.pi/4.0])

    sim = PymunkSimulation()
    sim.add_robot(model, q0)

    robot_renderer = plotter.ThreeInputRenderer(model, q0)
    plot = plotter.RealtimePlotter([robot_renderer])
    plot.start(grid=True)

    sim.command_velocity([0.1, 0.1, 0.1])

    for i in range(N - 1):
        t = ts[i]

        q, dq = sim.step(DT)

        if i % PLOT_PERIOD == 0:
            robot_renderer.set_state(q)
            plot.update()

    plot.done()


if __name__ == '__main__':
    main()
