import pymunk
import pymunk.matplotlib_util
import numpy as np
import matplotlib.pyplot as plt
import IPython


DT = 0.01
PLOT_PERIOD = 10
DURATION = 60.0  # duration of trajectory (s)

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


def plot_line(ax, a, b, color='k'):
    patch = plt.Line2D([a[0], b[0]], [a[1], b[1]], color=color, linewidth=1)
    ax.add_line(patch)


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

    pd = np.array([3, 0])
    # pd = np.array([-3, 0])
    # pd = np.array([2, -2])

    ax.plot(pd[0], pd[1], 'o', color='r')

    v_max = 0.2
    kp = 0.5

    for i in range(N - 1):
        t = i * DT

        # step the sim
        space.step(DT)

        p = np.array(control_body.position)
        Δp = pd - p
        # Δp_unit = unit(Δp)

        c = np.array(box.body.position)
        # Δp = pd - c

        # if watcher.first_contact:
        #     # φ = np.arccos(watcher.nf @ Δp_unit)
        #     φ = 1 - watcher.nf @ Δp_unit
        #
        #     # k is still a tuning parameter -- depends to some extent on object
        #     # curvature and friction
        #     # k = 1 / (1 + φ**2)
        #     k = 1
        #     kφ = k * φ  # / np.linalg.norm(Δp)
        #     print(φ)
        #     R = np.array([[np.cos(kφ), -np.sin(kφ)],
        #                   [np.sin(kφ), np.cos(kφ)]])
        #     v_unit = R @ watcher.nf
        #     v = min(v_max, np.linalg.norm(Δp)) * v_unit
        # else:
        #     v = v_max * np.array([1, 0])

        if watcher.first_contact:
            v_traj = kp * Δp
            v_mag = max(min(watcher.nf @ v_traj, v_max), 0)

            # reflect over nf
            α = 0.5
            d = (1+α) * watcher.nf * (watcher.nf @ Δp) - α*Δp

            # project onto d
            v = (v_traj @ unit(d)) * unit(d)

            # v_unit = unit(d)

            # v = v_mag * v_unit
        else:
            v = v_max * np.array([1, 0])

        control_body.velocity = (v[0], v[1])

        if i % PLOT_PERIOD == 0:
            ax.cla()
            ax.set_xlim([-5, 5])
            ax.set_ylim([-3, 3])
            ax.grid()

            ax.plot(pd[0], pd[1], 'o', color='r')

            plot_line(ax, p, p + unit(v), color='r')
            plot_line(ax, p, p + watcher.nf, color='b')

            space.debug_draw(options)
            fig.canvas.draw()
            fig.canvas.flush_events()


if __name__ == '__main__':
    main()
