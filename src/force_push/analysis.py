from collections import namedtuple
import glob
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
import seaborn
import tqdm


ExtremePointInfo = namedtuple("ExtremePointInfo", ["index", "distance", "point"])


def compute_simulation_extreme_points(data):
    """Compute the extreme points of a set of runs.

    The extreme points are the finishing point that is farthest from the
    "ideal" end of the path (i.e., the distance a perfect system would achieve
    with the given speed) and the point of max distance from the path.
    """
    # compute the final point if the slider moved exactly along the path
    path = data["path"]
    v = data["push_speed"]

    all_r_sw_ws = data["slider_positions"]
    all_in_contacts = data["in_contacts"]
    all_times = data["times"]
    num_sims = len(all_r_sw_ws)

    # compute distance between actual final points and the ideal one
    # we also compute the completion fraction for each run, which is the
    # fraction of the path completed by an ideal system if it ended the same
    # distance away from the end as the real system did
    max_final_info = ExtremePointInfo(index=0, distance=0, point=[0, 0])
    completion_fractions = []
    for i in range(num_sims):

        first_contact_idx = np.argmax(all_in_contacts[i])
        t0 = all_times[i][first_contact_idx]
        tf = all_times[i][-1]
        duration = tf - t0
        ideal_dist = v * duration

        # final position if path was tracked exactly the whole time
        ideal_final_pos = path.point_at_distance(ideal_dist)

        # actual final position
        # slider starts such that the closest point on the path is the origin
        final_slider_pos = all_r_sw_ws[i][-1, :2]

        # distance between ideal and actual final positions
        final_dist_from_ideal = np.linalg.norm(ideal_final_pos - final_slider_pos)

        completion_fractions.append((ideal_dist - final_dist_from_ideal) / ideal_dist)

        # also record the run that has the farthest distance from the ideal
        # final position
        if final_dist_from_ideal > max_final_info.distance:
            max_final_info = ExtremePointInfo(i, final_dist_from_ideal, final_slider_pos)
    completion_fractions = np.array(completion_fractions)

    # compute point of maximum deviation from the path (max closest distance)
    # we want the max overall as well as the max deviation for each run
    max_overall_deviation = ExtremePointInfo(index=0, distance=0, point=[0, 0])
    max_deviations = []
    with tqdm.tqdm(total=num_sims) as progress:
        for i in range(num_sims):

            # get the maximum deviation for a single run
            max_deviation = ExtremePointInfo(index=0, distance=0, point=[0, 0])
            for k in range(all_r_sw_ws[i].shape[0]):
                r_sw_w = all_r_sw_ws[i][k, :2]
                info = path.compute_closest_point_info(r_sw_w)

                if info.deviation > max_deviation.distance:
                    max_deviation = ExtremePointInfo(k, info.deviation, r_sw_w)

                # compare to overall maximum deviation
                if info.deviation > max_overall_deviation.distance:
                    max_overall_deviation = ExtremePointInfo(i, info.deviation, r_sw_w)
            max_deviations.append(max_deviation)
            progress.update(1)

    return {
        "ideal_final_pos": ideal_final_pos,
        "max_final_info": max_final_info,
        "max_deviation_info": max_overall_deviation,
        "completion_fractions": completion_fractions,
        "max_deviations": max_deviations,
    }


def plot_simulation_results(data, plot_forces=False):
    all_r_sw_ws = data["slider_positions"]
    all_forces = data["forces"]
    all_times = data["times"]
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

    if plot_forces:
        plt.figure()
        for i in range(n):
            plt.plot(all_times[i], all_forces[i][:, 0], color="r", alpha=0.5)
            plt.plot(all_times[i], all_forces[i][:, 1], color="b", alpha=0.5)
        plt.grid()
        plt.title("Forces vs. time")
        plt.xlabel("Time [s]")
        plt.ylabel("Force [N]")


def parse_bag_dir(directory):
    """Parse params pickle path and bag path from a data directory.

    Returns (param_path, bag_path), as strings."""
    dir_path = Path(directory)

    param_files = glob.glob(dir_path.as_posix() + "/*.pkl")
    if len(param_files) == 0:
        raise FileNotFoundError(
            "Error: could not find a pickle in the specified directory."
        )
    if len(param_files) > 1:
        raise FileNotFoundError(
            "Error: multiple pickles in the specified directory. Please specify the name using the `--config_name` option."
        )
    param_path = param_files[0]

    bag_files = glob.glob(dir_path.as_posix() + "/*.bag")
    if len(bag_files) == 0:
        raise FileNotFoundError(
            "Error: could not find a bag file in the specified directory."
        )
    if len(bag_files) > 1:
        raise FileNotFoundError(
            "Error: multiple bag files in the specified directory. Please specify the name using the `--bag_name` option."
        )
    bag_path = bag_files[0]
    return param_path, bag_path
