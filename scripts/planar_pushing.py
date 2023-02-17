import pymunk
import pymunk.matplotlib_util
import numpy as np
import matplotlib.pyplot as plt
import IPython


DT = 0.01
PLOT_PERIOD = 100
DURATION = 1000.0  # duration of trajectory (s)

M = 1
G = 10
MU_GROUND = 1
MU_CONTACT = 0.5


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
    """Normalize a vector."""
    norm = np.linalg.norm(x)
    if norm > 0:
        return x / norm
    return x


def rot2d(θ):
    """2D rotation matrix."""
    return np.array([[np.cos(θ), -np.sin(θ)], [np.sin(θ), np.cos(θ)]])


def signed_angle(a, b):
    """See <https://stackoverflow.com/a/2150111/5145874>"""
    return np.arctan2(b[1], b[0]) - np.arctan2(a[1], a[0])


def pursuit(p, lookahead):
    """Pure pursuit along the x-axis."""
    if np.abs(p[1]) >= lookahead:
        return np.array([0, -np.sign(p[1]) * lookahead])
    x = lookahead**2 - p[1]**2
    return np.array([x, -p[1]])


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
    ax.set_xlim([-5, 15])
    ax.set_ylim([-3, 3])
    ax.grid()

    ax.set_aspect("equal")

    options = pymunk.matplotlib_util.DrawOptions(ax)
    options.flags = pymunk.SpaceDebugDrawOptions.DRAW_SHAPES

    space.debug_draw(options)
    fig.canvas.draw()
    fig.canvas.flush_events()

    # pd = np.array([-3, -2])
    # pd = np.array([-3, 0])
    # pd = np.array([2, -2])
    # ax.plot(pd[0], pd[1], "o", color="r")

    # TODO we need the angle to be a certain size (dependent on μ) in order to
    # escape the friction cone

    v_max = 0.2
    kθ = 0.1  #np.arctan(MU_CONTACT)
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
        Δ = pursuit(p, 5)
        # Δ = unit([1, 0])

        if watcher.first_contact:
            # compute heading relative to desired direction
            θd = signed_angle(Δ, watcher.nf)
            φd = signed_angle([1, 0], watcher.nf)

            dφdt = 0.5 * (φd - φ)
            φ += DT * dφdt
            φ = φd

            # this is an attempt to smooth out the control law
            Δθ = θd - θ
            Δθ_int += DT * Δθ
            dθdt = k * Δθ + ki * Δθ_int
            θ += DT * dθdt

            θ = θd

            # we don't ever want to go backward
            # angle = kθ * θ + φ
            angle = kθ * θ + φ
            angle = max(min(angle, np.pi / 2), -np.pi / 2)

            direction = rot2d(angle) @ [1, 0]
            v_cmd = v_max * direction
        else:
            # if haven't yet touched the box, just go straight (toward it)
            v_cmd = v_max * np.array([1, 0])

        control_body.velocity = (v_cmd[0], v_cmd[1])

        box_positions.append(box.body.position)

        # if i % PLOT_PERIOD == 0:
        #     ax.cla()
        #     ax.set_xlim([-5, 15])
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
    ts = DT * np.arange(N - 1)
    plt.plot(ts, box_positions[:, 0], label="x")
    plt.plot(ts, box_positions[:, 1], label="y")
    plt.legend()
    plt.grid()
    plt.show()


if __name__ == "__main__":
    main()
