#!/usr/bin/env python3
"""Calibrate the arm using data collected with `collect_arm_calibration_data.py`.

We formulate a nonlinear optimization problem over the product of two SE(3) manifolds.
"""
import argparse
import datetime

import numpy as np
import pymanopt
import yaml
import jax
import jaxlie
from spatialmath.base import trnorm, r2q

import IPython


GRAVITY = np.array([0, 0, -9.81])
OBJECT_MASS = 2.225

# duct tape roll: 0.481 kg
# 5lb weight: 2.225 kg

NUM_DATA = 1000


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "data_file_name", help="NPZ file containing the calibration data."
    )
    parser.add_argument(
        "--save",
        help="YAML file to output with optimized transforms.",
    )
    args = parser.parse_args()

    # load calibration data
    data = np.load(args.data_file_name)
    f_fs = data["f_fs"]
    C_bfs = data["C_bfs"]
    n = f_fs.shape[0]
    indices = n // NUM_DATA * np.arange(NUM_DATA)

    f_d = OBJECT_MASS * GRAVITY

    manifold = pymanopt.manifolds.SpecialOrthogonalGroup(3)

    # jax-version of cost for autodiff
    def jcost(C):
        cost = 0
        for i in indices:
            f_b = C_bfs[i, :, :] @ f_fs[i, :]
            e = C @ f_b - f_d
            cost = cost + 0.5 * e @ e
        return cost

    # gradient of the cost
    jgrad = jax.grad(jcost)

    @pymanopt.function.numpy(manifold)
    def cost(C):
        return jcost(C)

    @pymanopt.function.numpy(manifold)
    def gradient(C):
        return jgrad(C)

    # initial guess
    C0 = np.eye(3)

    # setup and solve the optimization problem
    problem = pymanopt.Problem(manifold, cost, euclidean_gradient=gradient)
    line_searcher = pymanopt.optimizers.line_search.BackTrackingLineSearcher()
    optimizer = pymanopt.optimizers.SteepestDescent(line_searcher=line_searcher)
    result = optimizer.run(problem, initial_point=C0)

    # save as a quaternion
    C_opt = result.point
    C_normed = trnorm(C_opt)
    Q = r2q(C_normed, order="xyzs")
    print(f"C = {C_opt}")

    out_data = {"x": float(Q[0]), "y": float(Q[1]), "z": float(Q[2]), "w": float(Q[3])}
    print(yaml.dump(out_data))

    f_bs = np.array([C_bfs[i, :, :] @ f_fs[i, :] for i in range(n)])
    f_bs2 = np.array([C_normed @ C_bfs[i, :, :] @ f_fs[i, :] for i in range(n)])

    IPython.embed()

    # save parameters to a file for use in control
    if args.save is not None:
        timestamp = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        output_file_name = args.save + "_" + timestamp + ".yaml"
        with open(output_file_name, "w") as f:
            yaml.dump(out_data, f)
        print(f"Saved transform data to {output_file_name}.")


if __name__ == "__main__":
    main()
