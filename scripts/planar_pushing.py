import pymunk
import pymunk.matplotlib_util
import numpy as np
import matplotlib.pyplot as plt
import IPython


DT = 0.01
PLOT_PERIOD = 10
DURATION = 100.0  # duration of trajectory (s)

M = 1
G = 10
MU_GROUND = 1
MU_CONTACT = 0.2


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

    def get_force(self):
        f = self.f.copy()
        self.f = np.zeros(2)
        return f


def plot_line(ax, a, b, color="k"):
    patch = plt.Line2D([a[0], b[0]], [a[1], b[1]], color=color, linewidth=1)
    ax.add_line(patch)


def unit(x):
    norm = np.linalg.norm(x)
    if norm > 0:
        return x / norm
    return np.zeros(2)


def rot2d(θ):
    return np.array([[np.cos(θ), -np.sin(θ)], [np.sin(θ), np.cos(θ)]])


def main():
    N = int(DURATION / DT) + 1

    space = pymunk.Space()
    space.gravity = (0, 0)

    box_body = pymunk.Body()
    box_body.position = (1, -0.3)
    box_r = 0.5
    box_corners = [(-box_r, box_r), (-box_r, -box_r), (box_r, -box_r), (box_r, box_r)]
    box = pymunk.Poly(box_body, box_corners, radius=0.01)
    # box = pymunk.Circle(box_body, 0.5)

    box.mass = M
    box.friction = MU_CONTACT
    box.collision_type = 1
    space.add(box.body, box)

    # linear friction
    pivot = pymunk.PivotJoint(space.static_body, box.body, box.body.position)
    pivot.max_force = MU_GROUND * M * G  # μmg
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

    ax.set_xlabel("x (m)")
    ax.set_ylabel("y (m)")
    ax.set_xlim([-5, 10])
    ax.set_ylim([-3, 3])
    ax.grid()

    ax.set_aspect("equal")

    options = pymunk.matplotlib_util.DrawOptions(ax)
    options.flags = pymunk.SpaceDebugDrawOptions.DRAW_SHAPES

    space.debug_draw(options)
    fig.canvas.draw()
    fig.canvas.flush_events()

    pd = np.array([-3, -2])
    # pd = np.array([-3, 0])
    # pd = np.array([2, -2])

    ax.plot(pd[0], pd[1], "o", color="r")

    v_max = 0.2
    kθ = 1
    k = 1
    ki = 0.01
    θ_max = 0.25 * np.pi
    print_count = 0

    θd = 0
    θ = 0
    Δθ_int = 0
    φ = 0

    box_positions = []

    for i in range(N - 1):
        t = i * DT

        # step the sim
        space.step(DT)

        p = np.array(control_body.position)
        Δ = unit([1, -p[1]])
        # Δ = unit([1, 0])

        if watcher.first_contact:
            # compute heading relative to desired direction
            θd = np.arccos(watcher.nf @ Δ)
            if watcher.nf[1] < 0:
                θd = -θd

            φd = np.arccos(watcher.nf @ [1, 0])
            if watcher.nf[1] < 0:
               φd = -φd

            dφdt = 0.5 * (φd - φ)
            φ += DT * dφdt
            φ = φd

            # limit the angle5
            # θ_max = np.arctan(MU_CONTACT)
            θd = max(min(θd, θ_max), -θ_max)

            # this is an attempt to smooth out the control law
            Δθ = θd - θ
            Δθ_int += DT * Δθ
            dθdt = k * Δθ + ki * Δθ_int
            θ += DT * dθdt

            θ = θd

            # TODO why is the initial angle not zero?
            # print_count += 1
            # if print_count <= 2:
            #     print(θ)

            direction = rot2d(1 * θ + φ) @ [1, 0]
            # direction = rot2d(kθ * θ) @ watcher.nf
            # direction = rot2d((1+kθ) * θ) @ Δ
            # direction = rot2d(1. * θ) @ [Δ[0], -Δ[1]]

            v_cmd = v_max * direction
        else:
            # if haven't yet touched the box, just go straight (toward it)
            v_cmd = v_max * np.array([1, 0])

        control_body.velocity = (v_cmd[0], v_cmd[1])

        box_positions.append(box.body.position)

        # if i % PLOT_PERIOD == 0:
        #     ax.cla()
        #     ax.set_xlim([-5, 10])
        #     ax.set_ylim([-3, 3])
        #     ax.grid()
        #
        #     # ax.plot(pd[0], pd[1], "o", color="r")
        #
        #     plot_line(ax, p, p + unit(v_cmd), color="r")
        #     plot_line(ax, p, p + watcher.nf, color="b")
        #     plot_line(ax, p, p + unit(Δ), color="g")
        #
        #     space.debug_draw(options)
        #     fig.canvas.draw()
        #     fig.canvas.flush_events()

    box_positions = np.array(box_positions)
    plt.ioff()
    plt.figure()
    ts = DT * np.arange(N-1)
    plt.plot(ts, box_positions[:, 0], label="x")
    plt.plot(ts, box_positions[:, 1], label="y")
    plt.legend()
    plt.grid()
    plt.show()


if __name__ == "__main__":
    main()
