"""Simulation of pushing based on quasistatic model and my control law."""
import numpy as np
import matplotlib.pyplot as plt
import time
import tqdm
import mmpush

import IPython


def plot_simulation(xs):
    plt.plot(xs[:, 0], xs[:, 1])
    plt.xlabel("x [m]")
    plt.ylabel("y [m]")
    plt.grid()


def test_circle_slider():
    direction = np.array([1, 0])
    path = mmpush.StraightPath(direction)

    radius = 0.5
    speed = 0.5
    f_max = 1
    τ_max = 0.1 * f_max * mmpush.circle_r_tau(radius)
    μ = 0.0

    # control gains
    kθ = 0.1
    ky = 0.01

    # x = (x, y, θ, s, f_x, f_y)
    x0 = np.array([0.0, 0.4, 0, 0, 1, 0])

    if np.isclose(μ, 0):
        motion = mmpush.QPPusherSliderMotionZeroFriction(f_max, τ_max)
    else:
        motion = mmpush.QPPusherSliderMotion(f_max, τ_max, μ)
    slider = mmpush.CircleSlider(radius)

    duration = 2 * 120
    timestep = 0.005

    successes, ts, xs, us = mmpush.simulate_pushing(
        motion, slider, path, speed, kθ, ky, x0, duration, timestep
    )

    plt.figure()
    plot_simulation(xs)
    plt.show()
    return

    playback_simulation(xs, us, slider, path, sleep=0.000)


def test_quad_slider():
    direction = np.array([1, 0])
    path = mmpush.StraightPath(direction)
    speed = 0.5

    f_max = 5
    τ_max = 2.5
    μ = 0.2

    # control gains
    kθ = 0.1
    ky = 0.1

    # x = (x, y, θ, s, f_x, f_y)
    x0 = np.array([0.0, 0.4, 0, 0, 1, 0])

    motion = mmpush.QPPusherSliderMotion(f_max, τ_max, μ)
    slider = mmpush.QuadSlider(0.5, 0.5, cof=[0, 0])

    duration = 10
    timestep = 0.01

    successes, ts, xs, us = mmpush.simulate_pushing(
        motion, slider, path, speed, kθ, ky, x0, duration, timestep
    )
    mmpush.playback_simulation(xs, us, slider, path, sleep=0.001)


def main():
    test_circle_slider()


if __name__ == "__main__":
    main()
