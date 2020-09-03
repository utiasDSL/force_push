#!/usr/bin/env python

import numpy as np
import matplotlib.pyplot as plt
import IPython

from mm2d.model import ThreeInputModel
from mm2d.util import right_pseudoinverse


# model parameters
# link lengths
L1 = 1
L2 = 1

# input bounds
LB = -1
UB = 1


def svd(J):
    U, S, V = np.linalg.svd(J, full_matrices=True)
    V = V.T  # to be consistent with usual math notation
    S = np.diag(S)
    S = np.hstack((S, np.zeros((2, 1))))
    return U, S, V


def main():
    np.set_printoptions(precision=3, suppress=True)

    model = ThreeInputModel(L1, L2, LB, UB, output_idx=[0, 1])

    v = np.array([0.5, 0.5])
    q = np.array([0, 0.25*np.pi, -0.5*np.pi])

    J = model.jacobian(q)

    # SVD: J = U @ S @ V
    U, S, V = svd(J)

    v2 = U.T @ v

    Jp = right_pseudoinverse(J)
    Sp = right_pseudoinverse(S)
    u2 = Sp @ v2
    u = V @ u2

    IPython.embed()


if __name__ == '__main__':
    main()
