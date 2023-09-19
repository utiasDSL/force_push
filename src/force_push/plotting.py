import numpy as np
import matplotlib.pyplot as plt
import seaborn


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


def plot_simulation_results(data):
    all_r_sw_ws = data["slider_positions"]
    all_forces = data["forces"]
    ts = data["times"]
    r_dw_ws = data["path_positions"]

    Is = np.array([p[0] for p in data["parameters"]])
    μs = np.array([p[1] for p in data["parameters"]])

    n = len(all_r_sw_ws)

    # plotting
    palette = seaborn.color_palette("deep")

    plt.figure()
    for i in range(0, n):
        if μs[i] > 0.7:
            color = palette[1]
        elif μs[i] < 0.3:
            color = palette[2]
        else:
            color = palette[0]

        # if μs[i] < 0.7:
        #     continue
        #
        # if Is[i][2, 2] < 0.1:
        #     color = palette[1]
        #     continue
        # elif Is[i][2, 2] > 0.4:
        #     color = palette[2]
        # else:
        #     color = palette[0]
        #     continue

        plt.plot(all_r_sw_ws[i][:, 0], all_r_sw_ws[i][:, 1], color=color, alpha=0.2)
    plt.plot(r_dw_ws[:, 0], r_dw_ws[:, 1], "--", color="k")
    ax = plt.gca()
    ax.set_aspect("equal")
    plt.grid()

    plt.figure()
    for i in range(n):
        plt.plot(ts[i], all_forces[i][:, 0], color="r", alpha=0.5)
        plt.plot(ts[i], all_forces[i][:, 1], color="b", alpha=0.5)
    plt.grid()
    plt.title("Forces vs. time")
    plt.xlabel("Time [s]")
    plt.ylabel("Force [N]")

    plt.show()
