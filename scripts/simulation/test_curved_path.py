"""Simulation of pushing based on quasistatic model and my control law."""
import numpy as np
import matplotlib.pyplot as plt
import time
import tqdm
import force_push as fp

import IPython


def plot_simulation(xs):
    plt.plot(xs[:, 0], xs[:, 1])
    plt.xlabel("x [m]")
    plt.ylabel("y [m]")
    plt.grid()


def main():
    direction = np.array([1, 0])

    # radius = 2.0
    # path = fp.CirclePath(radius)

    radius = 0
    vertices = np.array([[0, 0], [5, 0]])
    path = fp.SegmentPath(vertices, final_direction=[0, 1])

    speed = 0.5

    hx = 0.5
    hy = 0.5
    f_max = 1
    τ_max = f_max * fp.rectangle_r_tau(2 * hx, 2 * hy)
    μ = 0.2

    # control gains
    kθ = 0.3
    ky = 0.1

    # x = (x, y, θ, s, f_x, f_y)
    x0 = np.array([hx, -radius, 0, 0, 1, 0])

    motion = fp.QPPusherSliderMotion(f_max, τ_max, μ)
    slider = fp.QuadSlider(hx, hy, cof=[0, 0])
    controller = fp.Controller(speed, kθ, ky, path, ki_θ=0, ki_y=0, lookahead=2)

    duration = 100
    timestep = 0.001

    success, ts, xs, us = fp.simulate_pushing2(
        motion, slider, controller, x0, duration, timestep
    )
    if not success:
        print("pushing failed!")

    plot_simulation(xs)
    for i in range(0, len(xs), 1000):
        r = xs[i, :2]
        φ = xs[i, 2]
        s = xs[i, 3]
        C_wo = fp.rot2d(φ)
        ee = C_wo @ slider.contact_point(s) + r

        c = path.compute_closest_point(ee)

        plt.plot([c[0], ee[0]], [c[1], ee[1]], color="k", alpha=0.1)
    plt.show()
    # fp.playback_simulation(xs, us, slider, path, sleep=0, step=50)


if __name__ == "__main__":
    main()
