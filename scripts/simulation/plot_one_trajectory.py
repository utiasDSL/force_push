"""Simulation and plot a single trajectory."""
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import seaborn
from mmpush import *

import IPython


# FIGURE_PATH = "simulate_few.pdf"
FIGURE_PATH = "/home/adam/phd/papers/pushing/heins-icra23/tex/figures/simulate_few.pdf"


def generate_data(slider, motion):
    direction = np.array([1, 0])
    path = StraightPath(direction)

    duration = 210
    # timestep = 0.005
    timestep = 0.01
    f_max = 1
    # speed = 0.5
    speed = 0.1

    # control gains
    kθ = 0.1
    ky = 0.01

    # x = (x, y, θ, s, f_x, f_y)
    x0 = np.array([0.0, -0.4, -np.pi / 8, -0.4, 1, 0])

    success, ts, xs, us = simulate_pushing(
        motion, slider, path, speed, kθ, ky, x0, duration, timestep
    )

    # check pushing was actually successful
    if not success:
        raise ValueError("slider failed!")

    return {"slider": slider, "motion": motion, "ts": ts, "xs": xs, "us": us}


def plot_data(data, ax):
    slider = data["slider"]
    motion = data["motion"]
    xs = data["xs"]
    us = data["us"]
    n = xs.shape[0]
    step = n // 15

    circle = type(slider) is CircleSlider

    palette = seaborn.color_palette("pastel")

    ax.set_aspect("equal")
    plt.axhline(0, color="k", linestyle="--", linewidth=0.75, zorder=-1)
    plt.xlim([-1, 21])
    plt.xticks([0, 5, 10, 15, 20])
    plt.ylim([-1.2, 2])
    plt.yticks([0, 1])
    plt.legend([], [], title=f"$\mu_c={int(motion.μ)}$", labelspacing=0, loc="lower right")

    for i in range(1, n - 1, step):
        x = xs[i, :]
        u = us[i, :]

        if x[0] > 20:
            break

        φ = x[2]
        s = x[3]
        C_wo = rot2d(φ)
        f = x[4:]  # world frame
        r_co_o = slider.contact_point(s)
        r_ow_w = x[:2]
        r_cw_w = r_ow_w + C_wo @ r_co_o
        vp = C_wo @ u

        if circle:
            patch = plt.Circle(r_ow_w, radius=slider.r, fill=True, ec=(0.8, 0.447, 0.498), fc=(1, 0.753, 0.796))
            ax.add_line(make_line(r_ow_w, r_ow_w + rot2d(φ) @ [slider.r, 0], color=(0.8, 0.447, 0.498)))
        else:
            patch = plt.Rectangle(
                r_ow_w - [slider.hx, slider.hy],
                width=2 * slider.hx,
                height=2 * slider.hy,
                angle=np.rad2deg(φ),
                rotation_point="center",
                ec=(0.392, 0.62, 0.71),
                fc=(0.678, 0.847, 0.902),
                fill=True,
            )
        ax.add_patch(patch)

        ax.add_line(make_line(r_cw_w, r_cw_w + 0.5 * unit(vp), color="k"))  #(0, 0.8, 0)))
        ax.plot(r_cw_w[0], r_cw_w[1], ".", color="k") #(0, 0.8, 0))

    plt.grid(color=(0.75, 0.75, 0.75), alpha=0.5, linewidth=0.5)


def hide_x_ticks(ax):
    ax.set_xticklabels([])
    ax.tick_params(axis="x", colors=(0, 0, 0, 0))


def main():
    f_max = 1

    print("Simulating square slider...")

    hx, hy = 0.5, 0.5
    τ_max = f_max * rectangle_r_tau(2 * hx, 2 * hy)
    slider = QuadSlider(hx, hy)
    motion = QPPusherSliderMotion(f_max, τ_max, μ=0)
    square_data0 = generate_data(slider, motion)

    motion = QPPusherSliderMotion(f_max, τ_max, μ=1.0)
    square_data1 = generate_data(slider, motion)

    print("Simulating circle slider...")

    r = 0.5
    τ_max = f_max * circle_r_tau(r)
    slider = CircleSlider(r)
    motion = QPPusherSliderMotion(f_max, τ_max, μ=0)
    circle_data0 = generate_data(slider, motion)

    motion = QPPusherSliderMotion(f_max, τ_max, μ=1.0)
    circle_data1 = generate_data(slider, motion)

    print("Plotting...")

    mpl.use("pgf")
    plt.rcParams.update(
        {
            "pgf.texsystem": "pdflatex",
            "font.size": 6,
            "font.family": "serif",
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

    fig = plt.figure(figsize=(3.25, 2))
    for i, data in enumerate([square_data0, square_data1, circle_data0, circle_data1]):
        ax = plt.subplot(4, 1, i + 1)
        plot_data(data, ax)
        if i < 3:
            hide_x_ticks(ax)
    ax.set_xlabel("$x$ [m]")
    fig.supylabel("$y$ [m]")

    fig.tight_layout(pad=0.1)
    fig.savefig(FIGURE_PATH)
    print(f"Saved figure to {FIGURE_PATH}")


if __name__ == "__main__":
    main()
