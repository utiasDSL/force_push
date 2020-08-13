#!/usr/bin/env python

# Closed-loop inverse kinematics algorithm: finds q such that f(q) = pd by
# solving a differential equation.

import numpy as np
import matplotlib.pyplot as plt
import IPython

from mm2d.model import ThreeInputModel


# model parameters
# link lengths
L1 = 1
L2 = 1

# input bounds
LB = -1
UB = 1


def pseudoinverse(J):
    JJT = J.dot(J.T)
    return J.T.dot(np.linalg.inv(JJT))


def clik(model, K, dt, vd, pd, q0):
    q = q0
    e = pd - model.forward(q)

    while e @ e > 0.0001:
        J = model.jacobian(q)
        p = model.forward(q)
        e = pd - p
        Jpi = pseudoinverse(J)
        dq = Jpi @ (vd + K @ e)
        q = q + dt * dq

    return q


def main():
    model = ThreeInputModel(L1, L2, LB, UB, output_idx=[0, 1])

    vd = np.array([0, 0])
    pd = np.array([1, 1])
    K = np.eye(2)
    dt = 0.1
    q0 = np.zeros(3)

    q = clik(model, K, dt, vd, pd, q0)

    print(q)
    print(model.forward(q))


if __name__ == '__main__':
    main()
