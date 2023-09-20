#!/usr/bin/env python3
import argparse
import pickle
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import tqdm
import pyb_utils

import force_push as fp

import IPython


def main():
    np.set_printoptions(precision=6, suppress=True)

    parser = argparse.ArgumentParser()
    parser.add_argument("file", help="npz file to load data from.")
    parser.add_argument("--save", action="store_true", help="Save the results.")
    args = parser.parse_args()

    input_path = Path(args.file)
    with open(input_path, "rb") as f:
        data = pickle.load(f)
    print(f"Loaded processed data from {args.file}")

    # print(data["slider_stiffness"])

    # compute the final point if the slider moved exactly along the path
    path = data["path"]
    dist = data["duration"] * data["push_speed"]
    ideal_final_pos = path.point_at_distance(dist)

    # compute distance between actual final points and the ideal one
    all_r_sw_ws = data["slider_positions"]
    num_sims = len(all_r_sw_ws)
    max_final_dist = (0, 0, [0, 0])  # index, distance, point
    for i in range(num_sims):
        final_slider_pos = all_r_sw_ws[i][-1, :2]
        final_dist = np.linalg.norm(ideal_final_pos - final_slider_pos)
        if final_dist > max_final_dist[1]:
            max_final_dist = (i, final_dist, final_slider_pos)

    # compute point of maximum deviation from the path (max closest distance)
    max_closest_point = (0, 0, [0, 0])  # index, distance, point
    with tqdm.tqdm(total=num_sims) as progress:
        for i in range(num_sims):
            for k in range(all_r_sw_ws[i].shape[0]):
                r_sw_w = all_r_sw_ws[i][k, :2]
                closest = path.compute_closest_point(r_sw_w)
                dist = np.linalg.norm(closest - r_sw_w)
                if dist > max_closest_point[1]:
                    max_closest_point = (i, dist, r_sw_w)
            progress.update(1)

    # TODO we can save, but we need to be able to load it later
    if args.save:
        results = {
            "ideal_final_pos": ideal_final_pos,
            "max_final_dist": max_final_dist,
            "max_closest_point": max_closest_point,
        }
        output_path = Path(input_path.stem + "_results.pkl")
        with open(output_path, "wb") as f:
            pickle.dump(results, f)

    fp.plot_simulation_results(data)
    plt.plot([max_final_dist[2][0]], [max_final_dist[2][1]], "x", color="k")
    plt.plot([max_closest_point[2][0]], [max_closest_point[2][1]], "o", color="k")
    plt.show()


if __name__ == "__main__":
    main()
