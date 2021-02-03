import pymunk
import pymunk.matplotlib_util
import numpy as np
import matplotlib.pyplot as plt
import IPython


DT = 0.01
PLOT_PERIOD = 10
DURATION = 20.0  # duration of trajectory (s)

M = 1
G = 10
MU = 1


class CollisionWatcher:
    """Tracks information about collisions."""
    def __init__(self):
        self.n = np.zeros(2)
        self.f = np.zeros(2)
        self.nf = np.zeros(2)
        self.first_contact = False

    def update(self, arbiter, space, data):
        self.first_contact = True
        self.n = np.array(arbiter.contact_point_set.normal)
        self.f = -np.array(arbiter.total_impulse / DT)  # negative compared to normal
        # self.nf = 0.9 * unit(self.f) + 0.1 * self.nf
        if np.linalg.norm(self.f) > 0:
            self.nf = unit(self.f)


def unit(x):
    norm = np.linalg.norm(x)
    if norm > 0:
        return x / norm
    return np.zeros(2)


def main():
    N = int(DURATION / DT) + 1

    space = pymunk.Space()
    space.gravity = (0, 0)

    box_body = pymunk.Body()
    box_body.position = (1, 0.3)
    box_r = 0.5
    box_corners = [(-box_r, box_r), (-box_r, -box_r), (box_r, -box_r), (box_r, box_r)]
    box = pymunk.Poly(box_body, box_corners, radius=0.01)
    # box = pymunk.Circle(box_body, 0.5)
    box.mass = M
    box.friction = 0.5
    box.collision_type = 1
    space.add(box.body, box)

    # linear friction
    pivot = pymunk.PivotJoint(space.static_body, box.body, box.body.position)
    pivot.max_force = MU*M*G  # μmg
    pivot.max_bias = 0  # disable correction so that body doesn't bounce back
    space.add(pivot)

    # angular friction
    gear = pymunk.GearJoint(space.static_body, box.body, 0, 1)
    gear.max_force = 2
    gear.max_bias = 0
    space.add(gear)

    control_body = pymunk.Body(body_type=pymunk.Body.KINEMATIC)
    control_body.position = (0, 0)
    control_shape = pymunk.Circle(control_body, 0.1, (0, 0))
    control_shape.friction = 1
    control_shape.collision_type = 1
    space.add(control_shape.body, control_shape)

    watcher = CollisionWatcher()
    space.add_collision_handler(1, 1).post_solve = watcher.update

    plt.ion()
    fig = plt.figure()
    ax = plt.gca()

    ax.set_xlabel('x (m)')
    ax.set_ylabel('y (m)')
    ax.set_xlim([-5, 5])
    ax.set_ylim([-3, 3])
    ax.grid()

    ax.set_aspect('equal')

    options = pymunk.matplotlib_util.DrawOptions(ax)
    options.flags = pymunk.SpaceDebugDrawOptions.DRAW_SHAPES

    space.debug_draw(options)
    fig.canvas.draw()
    fig.canvas.flush_events()

    control_body.velocity = (0.25, 0)

    for i in range(N - 1):
        t = i * DT

        # step the sim
        space.step(DT)

        if i % PLOT_PERIOD == 0:
            ax.cla()
            ax.set_xlim([-5, 5])
            ax.set_ylim([-3, 3])
            ax.grid()

            space.debug_draw(options)
            fig.canvas.draw()
            fig.canvas.flush_events()


if __name__ == '__main__':
    main()
