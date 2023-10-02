from collections import namedtuple
import tqdm

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


ExtremePointInfo = namedtuple("ExtremePointInfo", ["index", "distance", "point"])


def compute_simulation_extreme_points(data):
    """Compute the extreme points of a set of runs.

    The extreme points are the finishing point that is farthest from the
    "ideal" end of the path (i.e., the distance a perfect system would achieve
    with the given speed) and the point of max distance from the path.
    """
    # compute the final point if the slider moved exactly along the path
    path = data["path"]
    dist = data["duration"] * data["push_speed"]
    ideal_final_pos = path.point_at_distance(dist)

    all_r_sw_ws = data["slider_positions"]
    num_sims = len(all_r_sw_ws)

    # compute distance between actual final points and the ideal one
    # we also compute the completion fraction for each run, which is the
    # fraction of the path completed by an ideal system if it ended the same
    # distance away from the end as the real system did
    max_final_info = ExtremePointInfo(index=0, distance=0, point=[0, 0])
    completion_fractions = []
    for i in range(num_sims):
        final_slider_pos = all_r_sw_ws[i][-1, :2]
        final_dist = np.linalg.norm(ideal_final_pos - final_slider_pos)
        completion_fractions.append((dist - final_dist) / dist)
        if final_dist > max_final_info.distance:
            max_final_info = ExtremePointInfo(i, final_dist, final_slider_pos)
    completion_fractions = np.array(completion_fractions)

    # compute point of maximum deviation from the path (max closest distance)
    max_deviation_info = ExtremePointInfo(index=0, distance=0, point=[0, 0])
    with tqdm.tqdm(total=num_sims) as progress:
        for i in range(num_sims):
            for k in range(all_r_sw_ws[i].shape[0]):
                r_sw_w = all_r_sw_ws[i][k, :2]
                closest = path.compute_closest_point(r_sw_w)
                dist = np.linalg.norm(closest - r_sw_w)
                if dist > max_deviation_info.distance:
                    max_deviation_info = ExtremePointInfo(i, dist, r_sw_w)
            progress.update(1)

    return {
        "ideal_final_pos": ideal_final_pos,
        "max_final_info": max_final_info,
        "max_deviation_info": max_deviation_info,
        "completion_fractions": completion_fractions,
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
        color = palette[0]
        # if μs[i] > 0.7:
        #     color = palette[1]
        # elif μs[i] < 0.3:
        #     color = palette[2]
        # else:
        #     color = palette[0]

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

    # plt.figure()
    # for i in range(n):
    #     plt.plot(ts[i], all_forces[i][:, 0], color="r", alpha=0.5)
    #     plt.plot(ts[i], all_forces[i][:, 1], color="b", alpha=0.5)
    # plt.grid()
    # plt.title("Forces vs. time")
    # plt.xlabel("Time [s]")
    # plt.ylabel("Force [N]")
