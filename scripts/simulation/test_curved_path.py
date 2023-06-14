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
    radius = 2.0
    path = fp.CirclePath(radius)
    speed = 0.5

    hx = 0.5
    hy = 0.5
    f_max = 1
    τ_max = f_max * fp.rectangle_r_tau(2 * hx, 2 * hy)
    μ = 0.2

    # control gains
    kθ = 0.25
    ky = 0.1

    # x = (x, y, θ, s, f_x, f_y)
    x0 = np.array([hx, -radius, 0, 0, 1, 0])

    motion = fp.QPPusherSliderMotion(f_max, τ_max, μ)
    slider = fp.QuadSlider(hx, hy, cof=[0, 0])

    duration = 100
    timestep = 0.001

    success, ts, xs, us = fp.simulate_pushing2(
        motion, slider, path, speed, kθ, ky, x0, duration, timestep
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
