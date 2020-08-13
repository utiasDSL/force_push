#!/usr/bin/env python

import numpy as np
import scipy.optimize as opt

from mm2d.model import ThreeInputModel

import IPython

# model parameters
# link lengths
L1 = 1
L2 = 1

# input bounds
LB = -1
UB = 1


def main():
    model = ThreeInputModel(L1, L2, LB, UB, output_idx=[0, 1])

    def manipulability(qa):
        # Only for the arm
        q = np.array([0, qa[0], qa[1]])
        J = model.jacobian(q)
        Ja = J[:, 1:]
        m2 = np.linalg.det(Ja @ Ja.T)
        return -m2


    def worst_case_v(J):
        W = np.linalg.inv(J.dot(J.T))
        max_val = 0
        opt_v = np.zeros(2)
        for i in range(1000):
            v = np.random.rand(2)
            v = v / np.linalg.norm(v)
            C = v.T @ W @ v

            if C > max_val:
                max_val = C
                opt_v = v
        return opt_v, max_val


    # qa0 = np.zeros(2)
    # res = opt.minimize(manipulability, qa0, method='Nelder-Mead')

    # optimal solution is q2 = +-pi/2
    q = np.array([0, 0, np.pi/2])
    J = model.jacobian(q)

    opt_v, max_val = worst_case_v(J)

    IPython.embed()


if __name__ == '__main__':
    main()
