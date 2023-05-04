"""Simulation of pushing based on quasistatic model and my control law."""
import argparse
import pickle
import time

import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import seaborn
import tqdm

from mmpush import *

import IPython


# FIGURE_PATH = "simulate_many.pdf"
FIGURE_PATH = "/home/adam/phd/papers/pushing/heins-icra23/tex/figures/simulate_many.pdf"


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
                if np.isclose(μ, 0):
                    motion = QPPusherSliderMotionZeroFriction(f_max, τ_max)
                else:
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
                            if not success:
                                raise ValueError(
                                    f"{type(slider)} failed with x0 = {x0}, μ = {μ}"
                                )

                            successes.append(success)
                            all_ts.append(ts)
                            all_xs.append(xs)
                            all_μs.append(μ)
                            progress.update(1)
    return successes, all_ts, all_xs, all_μs


def generate_data(square=True, circle=True):
    direction = np.array([1, 0])
    path = StraightPath(direction)

    # state is x = (x, y, θ, s, f_x, f_y)
    duration = 600  # 10 minutes
    # timestep = 0.005
    timestep = 0.01
    f_max = 1
    speed = 0.1

    # control gains
    kθ = 0.1
    ky = 0.01

    y0s = [-0.4, 0, 0.4]
    θ0s = [-np.pi / 8, 0, np.pi / 8]
    s0s = [-0.4, 0, 0.4]
    μ0s = [0, 0.5, 1.0]

    data = {
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
    }

    if square:
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
        data["square"] = xs_square

    if circle:
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
        data["circle"] = xs_circle

    return data


def hide_x_ticks(ax):
    ax.set_xticklabels([])
    ax.tick_params(axis="x", colors=(0, 0, 0, 0))


def plot_data(data):
    mpl.use("pgf")
    plt.rcParams.update(
        {
            "pgf.texsystem": "pdflatex",
            "font.size": 6,
            "font.family": "serif",
            # "font.serif": "Palatino",
            "font.sans-serif": "DejaVu Sans",
            "font.weight": "normal",
            "text.usetex": True,
            "legend.fontsize": 6,
            "axes.titlesize": 6,
            "axes.labelsize": 6,
            "figure.labelsize": 6,
            "xtick.labelsize": 6,
            "pgf.preamble": "\n".join(
                [
                    r"\usepackage[utf8]{inputenc}",
                    r"\usepackage[T1]{fontenc}",
                    r"\usepackage{siunitx}",
                    r"\usepackage{bm}",
                ]
            ),
        }
    )

    palette = seaborn.color_palette("deep")

    fig = plt.figure(figsize=(3.25, 1.75))

    if "square" in data:
        xs_square = data["square"]
        ax1 = plt.subplot(2, 1, 1)
        plt.axhline(0, color="k", linestyle="--", linewidth=0.75, zorder=-1)
        for i in range(len(xs_square)):
            plt.plot(xs_square[i][:, 0], xs_square[i][:, 1], color=palette[0], alpha=0.1)
        # plt.ylabel("$y$ [m]")
        plt.yticks([-3, 0, 3])
        hide_x_ticks(ax1)
        plt.grid(color=(0.75, 0.75, 0.75), alpha=0.5, linewidth=0.5)

        # legend with only a title
        plt.legend([], [], title="Square", labelspacing=0, loc="upper right")

    if "circle" in data:
        xs_circle = data["circle"]
        ax2 = plt.subplot(2, 1, 2)
        plt.axhline(0, color="k", linestyle="--", linewidth=0.75, zorder=-1)
        for i in range(len(xs_circle)):
            plt.plot(xs_circle[i][:, 0], xs_circle[i][:, 1], color=palette[3], alpha=0.1)
        plt.xlabel("$x$ [m]")

        # fake y label
        plt.ylabel("$y$ [m]", alpha=0)
        plt.yticks([-3, 0, 3])
        plt.grid(color=(0.75, 0.75, 0.75), alpha=0.5, linewidth=0.5)
        plt.legend([], [], title="Circular", labelspacing=0, loc="upper right")

    fig.supylabel("$y$ [m]")

    fig.tight_layout(pad=0.1)
    fig.savefig(FIGURE_PATH)
    print(f"Saved figure to {FIGURE_PATH}")


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

    plot_data(data)


if __name__ == "__main__":
    main()
