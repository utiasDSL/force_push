#!/usr/bin/env python3
import argparse
import pickle
from pathlib import Path

import numpy as np
import pyb_utils

import force_push as fp

import IPython


def main():
    np.set_printoptions(precision=6, suppress=True)

    parser = argparse.ArgumentParser()
    parser.add_argument("file", help="npz file to load data from.")
    args = parser.parse_args()

    with open(args.file, "rb") as f:
        data = pickle.load(f)
    print(f"Loaded processed data from {args.file}")

    print(data["stiffness"])

    # compute the final point if the slider moved exactly along the path
    path = data["path"]
    dist = data["duration"] * data["push_speed"]
    ideal_final_pos = path.point_at_distance(dist)
    print(ideal_final_pos)

    # compute distance between actual final points and the ideal one
    all_r_sw_ws = data["slider_positions"]
    num_sims = len(all_r_sw_ws)
    for i in range(num_sims):
        final_slider_pos = all_r_sw_ws[i][-1, :2]
        print(np.linalg.norm(ideal_final_pos - final_slider_pos))

    fp.plot_simulation_results(data)


if __name__ == "__main__":
    main()
