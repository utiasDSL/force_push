"""Simulation of pushing based on quasistatic model and my control law."""
import numpy as np
import matplotlib.pyplot as plt
import time
from mmpush import *

import IPython


def simulate_many(
    slider, f_max, τ_maxes, path, speed, kθ, ky, duration, timestep, y0s, θ0s, sθs, μs
):
    """Simulate many pushes, one for each value of x0."""
    x0s = []

    all_ts = []
    all_xs = []
    successes = []
    all_μs = []

    for τ_max in τ_maxes:
        for μ in μs:
            motion = QPPusherSliderMotion(f_max, τ_max, μ)
            for y0 in y0s:
                for θ0 in θ0s:
                    for s0 in sθs:
                        x0 = np.array([0.0, y0, θ0, s0, 1, 0])
                        x0s.append(x0)
                        success, ts, xs, us = simulate_pushing(
                            motion, slider, path, speed, kθ, ky, x0, duration, timestep
                        )
                        successes.append(success)
                        all_ts.append(ts)
                        all_xs.append(xs)
                        all_μs.append(μ)
    return successes, all_ts, all_xs, all_μs


def plot_simulation(xs):
    plt.plot(xs[:, 0], xs[:, 1])
    plt.xlabel("x [m]")
    plt.ylabel("y [m]")
    plt.grid()


def test_circle_slider():
    direction = np.array([1, 0])
    path = StraightPath(direction)

    radius = 0.5
    speed = 0.5
    f_max = 1
    τ_max = 0.01  #0.1 * f_max * circle_r_tau(radius)
    μ = 0.0

    # control gains
    kθ = 0.1
    ky = 0.01

    # x = (x, y, θ, s, f_x, f_y)
    x0 = np.array([0.0, 0.4, 0, 0, 1, 0])

    # motion = QPPusherSliderMotionZeroFriction(f_max, τ_max)
    motion = PusherSliderMotion(f_max, τ_max, μ)
    slider = CircleSlider(radius)

    duration = 2 * 120
    timestep = 0.005

    successes, ts, xs, us = simulate_pushing(
        motion, slider, path, speed, kθ, ky, x0, duration, timestep
    )

    plt.figure()
    plot_simulation(xs)
    plt.show()
    return

    playback_simulation(xs, us, slider, path, sleep=0.000)


# def test_quad_slider():
#     direction = np.array([1, 0])
#     path = StraightPath(direction)
#     speed = 0.5
#
#     f_max = 5
#     τ_max = 2.5
#     μ = 0.2
#
#     # control gains
#     kθ = 0.1
#     ky = 0.1
#
#     # x = (x, y, θ, s, f_x, f_y)
#     x0 = np.array([0.0, 0.4, 0, 0, 1, 0])
#
#     motion = QPPusherSliderMotion(f_max, τ_max, μ)
#     slider = QuadSlider(0.5, 0.5, cof=[0, 0])
#
#     duration = 10
#     timestep = 0.01
#
#     successes, ts, xs, us = simulate_pushing(
#         motion, slider, path, speed, kθ, ky, x0, duration, timestep
#     )
#     playback_simulation(xs, us, slider, path, sleep=0.001)


def main():
    test_circle_slider()
    return

    direction = np.array([1, 0])
    path = StraightPath(direction)

    f_max = 1
    speed = 0.5

    # control gains
    kθ = 0.1
    ky = 0.01
    # kθ = 0.1
    # ky = 0.01

    # state is x = (x, y, θ, s, f_x, f_y)

    duration = 120  # two minutes
    timestep = 0.01

    y0s = [-0.4, 0, 0.4]
    θ0s = [-np.pi/8, 0, np.pi/8]
    s0s = [-0.4, 0, 0.4]
    μ0s = [0, 0.5, 1.0]

    # print("Simulating square slider...")

    # τ_max_uniform = f_max * rectangle_r_tau(1, 1)
    # τ_max_min = 0.1 * τ_max_uniform
    # τ_max_max = f_max * np.sqrt(2) / 2
    # τ_maxes = [τ_max_min, τ_max_uniform, τ_max_max]
    # slider = QuadSlider(0.5, 0.5)
    #
    # plt.figure()
    # successes, ts, xs, μs = simulate_many(
    #     slider, f_max, τ_maxes, path, speed, kθ, ky, duration, timestep, y0s, θ0s, s0s, μ0s
    # )
    # n = len(ts)
    # for i in range(n):
    #     plt.plot(xs[i][:, 0], xs[i][:, 1], color="b", alpha=0.1)
    #     if not successes[i]:
    #         print(f"square failed with x0 = {xs[i][0, :]}, μ = {μs[i]}")
    # plt.xlabel("x [m]")
    # plt.ylabel("y [m]")
    # plt.title("Square slider")
    # plt.grid()

    print("Simulating circle slider...")

    radius = 0.5
    τ_max_uniform = f_max * circle_r_tau(radius)
    τ_max_min = 0.1 * τ_max_uniform
    τ_max_max = f_max * radius
    τ_maxes = [τ_max_min, τ_max_uniform, τ_max_max]
    slider = CircleSlider(radius)

    # μ0s = [1.0]
    # successes, ts, xs_μ1, μs = simulate_many(
    #     slider, f_max, τ_maxes, path, speed, kθ, ky, duration, timestep, y0s, θ0s, s0s, μ0s
    # )

    μ0s = [0]
    τ_maxes = [τ_max_min]
    successes, ts, xs_μ0, μs = simulate_many(
        slider, f_max, τ_maxes, path, speed, kθ, ky, duration, timestep, y0s, θ0s, s0s, μ0s
    )

    n = len(ts)

    # plt.figure()
    # for i in range(n):
    #     plt.plot(xs_μ1[i][:, 0], xs_μ1[i][:, 1], color="r", alpha=0.1)
    #     if not successes[i]:
    #         print(f"circle failed with x0 = {xs[i][0, :]}, μ = {μs[i]}")
    for i in range(n):
        plt.plot(xs_μ0[i][:, 0], xs_μ0[i][:, 1], color="b", alpha=0.2)
        if not successes[i]:
            print(f"circle failed with x0 = {xs[i][0, :]}, μ = {μs[i]}")

    plt.xlabel("x [m]")
    plt.ylabel("y [m]")
    plt.title("Circle slider")
    plt.grid()

    plt.show()


if __name__ == "__main__":
    main()
