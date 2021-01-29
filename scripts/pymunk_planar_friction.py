import pymunk
import pymunk.matplotlib_util
import numpy as np
import matplotlib.pyplot as plt


DT = 0.01
PLOT_PERIOD = 10
DURATION = 20.0  # duration of trajectory (s)

M = 1
G = 10
MU = 1


def main():
    N = int(DURATION / DT) + 1

    space = pymunk.Space()
    space.gravity = (0, 0)

    box_body = pymunk.Body()
    box_body.position = (1, 0.4)
    box_corners = [(-0.5, 0.5), (-0.5, -0.5), (0.5, -0.5), (0.5, 0.5)]
    box = pymunk.Poly(box_body, box_corners, radius=0.01)
    box.mass = M
    box.friction = 0.75
    space.add(box.body, box)

    # linear friction
    pivot = pymunk.PivotJoint(space.static_body, box.body, box.body.position)
    pivot.max_force = 100  # μmg
    pivot.max_bias = 0  # disable correction so that body doesn't bounce back
    space.add(pivot)

    # angular friction
    gear = pymunk.GearJoint(space.static_body, box.body, 0, 1)
    gear.max_force = 10
    gear.max_bias = 0
    space.add(gear)

    control_body = pymunk.Body(body_type=pymunk.Body.KINEMATIC)
    control_body.position = (0, 0)
    control_shape = pymunk.Circle(control_body, 0.1, (0, 0))
    control_shape.friction = 1
    space.add(control_shape.body, control_shape)


    plt.ion()
    fig = plt.figure()
    ax = plt.gca()

    ax.set_xlabel('x (m)')
    ax.set_ylabel('y (m)')
    ax.set_xlim([-1, 5])
    ax.set_ylim([-3, 3])
    ax.grid()

    ax.set_aspect('equal')

    options = pymunk.matplotlib_util.DrawOptions(ax)
    options.flags = pymunk.SpaceDebugDrawOptions.DRAW_SHAPES

    space.debug_draw(options)
    fig.canvas.draw()
    fig.canvas.flush_events()

    for i in range(N - 1):
        t = i * DT

        control_body.velocity = (0.25, 0)
        # if t < 8:
        #     control_body.velocity = (0.1, 0)
        # else:
        #     control_body.velocity = (0, 0)

        # step the sim
        space.step(DT)

        if i % PLOT_PERIOD == 0:
            ax.cla()
            ax.set_xlim([-1, 5])
            ax.set_ylim([-3, 3])
            ax.grid()

            space.debug_draw(options)
            fig.canvas.draw()
            fig.canvas.flush_events()


if __name__ == '__main__':
    main()
