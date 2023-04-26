"""Simulation of pushing based on quasistatic model and my control law."""
import argparse
import pickle
import time

import numpy as np
import matplotlib.pyplot as plt
import tqdm

from mmpush import *

import IPython


def simulate_many(
    slider, f_max, τ_maxes, path, speed, kθ, ky, duration, timestep, y0s, θ0s, s0s, μs
):
    """Simulate many pushes, one for each value of x0 and other parameters."""
    x0s = []

    all_ts = []
    all_xs = []
    successes = []
    all_μs = []

    num_sims = len(τ_maxes) * len(y0s) * len(θ0s) * len(s0s) * len(μs)

    with tqdm.tqdm(total=num_sims) as progress:
        for τ_max in τ_maxes:
            for μ in μs:
                motion = QPPusherSliderMotion(f_max, τ_max, μ)
                for y0 in y0s:
                    for θ0 in θ0s:
                        for s0 in s0s:
                            x0 = np.array([0.0, y0, θ0, s0, 1, 0])
                            x0s.append(x0)
                            success, ts, xs, us = simulate_pushing(
                                motion,
                                slider,
                                path,
                                speed,
                                kθ,
                                ky,
                                x0,
                                duration,
                                timestep,
                            )
                            successes.append(success)
                            all_ts.append(ts)
                            all_xs.append(xs)
                            all_μs.append(μ)
                            progress.update(1)
    return successes, all_ts, all_xs, all_μs


def generate_data():
    direction = np.array([1, 0])
    path = StraightPath(direction)

    # state is x = (x, y, θ, s, f_x, f_y)
    duration = 120  # two minutes
    timestep = 0.005
    f_max = 1
    speed = 0.5

    # control gains
    kθ = 0.1
    ky = 0.01

    y0s = [-0.4, 0, 0.4]
    θ0s = [-np.pi / 8, 0, np.pi / 8]
    s0s = [-0.4, 0, 0.4]
    μ0s = [0, 0.5, 1.0]

    print("Simulating square slider...")

    hx, hy = 0.5, 0.5
    τ_max_uniform = f_max * rectangle_r_tau(2 * hx, 2 * hy)
    τ_max_min = 0.1 * τ_max_uniform
    τ_max_max = f_max * np.linalg.norm([hx, hy])
    τ_maxes = [τ_max_min, τ_max_uniform, τ_max_max]
    slider = QuadSlider(hx, hy)

    successes, ts, xs_square, μs = simulate_many(
        slider,
        f_max,
        τ_maxes,
        path,
        speed,
        kθ,
        ky,
        duration,
        timestep,
        y0s,
        θ0s,
        s0s,
        μ0s,
    )
    for i, success in enumerate(successes):
        if not success:
            raise ValueError(
                f"square failed with x0 = {xs_square[i][0, :]}, μ = {μs[i]}"
            )

    print("Simulating circle slider...")

    radius = 0.5
    τ_max_uniform = f_max * circle_r_tau(radius)
    τ_max_min = 0.1 * τ_max_uniform
    τ_max_max = f_max * radius
    τ_maxes = [τ_max_min, τ_max_uniform, τ_max_max]
    slider = CircleSlider(radius)

    successes, ts, xs_circle, μs = simulate_many(
        slider,
        f_max,
        τ_maxes,
        path,
        speed,
        kθ,
        ky,
        duration,
        timestep,
        y0s,
        θ0s,
        s0s,
        μ0s,
    )
    for i, success in enumerate(successes):
        if not success:
            raise ValueError(
                f"circle failed with x0 = {xs_circle[i][0, :]}, μ = {μs[i]}"
            )

    return {
        "duration": duration,
        "timestep": timestep,
        "f_max": f_max,
        "speed": speed,
        "kθ": kθ,
        "ky": ky,
        "y0s": y0s,
        "θ0s": θ0s,
        "s0s": s0s,
        "μ0s": μ0s,
        "square": xs_square,
        "circle": xs_circle,
    }


def hide_x_ticks(ax):
    ax.set_xticklabels([])
    # ax.set_xticks([])


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--save", help="Save processed data to this file.")
    parser.add_argument("--load", help="Load processed data from this file.")
    args = parser.parse_args()

    if args.load is not None:
        with open(args.load, "rb") as f:
            data = pickle.load(f)
        print(f"Loaded processed data from {args.load}")
    else:
        data = generate_data()
        if args.save is not None:
            with open(args.save, "wb") as f:
                pickle.dump(data, f)
            print(f"Saved processed data to {args.save}")

    xs_square = data["square"]

    plt.figure()
    plt.subplot(211)
    ax = plt.gca()
    for i in range(len(xs_square)):
        plt.plot(xs_square[i][:, 0], xs_square[i][:, 1], color="b", alpha=0.1)
    plt.ylabel("y [m]")
    hide_x_ticks(ax)
    plt.title("Square")
    plt.grid()

    xs_circle = data["circle"]

    plt.subplot(212)
    for i in range(len(xs_circle)):
        plt.plot(xs_circle[i][:, 0], xs_circle[i][:, 1], color="r", alpha=0.1)
    plt.xlabel("x [m]")
    plt.ylabel("y [m]")
    plt.title("Circle")
    plt.grid()

    plt.show()


if __name__ == "__main__":
    main()
