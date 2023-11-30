"""Utilities for making nice plots."""
import matplotlib.pyplot as plt


def rcparams(fontsize=6):
    return {
        "pgf.texsystem": "pdflatex",
        "font.size": fontsize,
        "font.family": "serif",
        "font.sans-serif": "DejaVu Sans",
        "font.weight": "normal",
        "text.usetex": True,
        "legend.fontsize": fontsize,
        "axes.titlesize": fontsize,
        "axes.labelsize": fontsize,
        "figure.labelsize": fontsize,
        "xtick.labelsize": fontsize,
        "pgf.preamble": "\n".join(
            [
                r"\usepackage[utf8]{inputenc}",
                r"\usepackage[T1]{fontenc}",
                r"\usepackage{siunitx}",
                r"\usepackage{bm}",
            ]
        ),
    }


def hide_x_ticks(ax):
    ax.set_xticklabels([])
    ax.tick_params(axis="x", colors=(0, 0, 0, 0))


def hide_y_ticks(ax):
    ax.set_yticklabels([])
    ax.tick_params(axis="y", colors=(0, 0, 0, 0))


def make_line(a, b, color="k", **kwargs):
    return plt.Line2D(
        [a[0], b[0]],
        [a[1], b[1]],
        color=color,
        linewidth=1,
        solid_capstyle="round",
        **kwargs
    )


def update_line(line, a, b):
    line.set_xdata([a[0], b[0]])
    line.set_ydata([a[1], b[1]])
