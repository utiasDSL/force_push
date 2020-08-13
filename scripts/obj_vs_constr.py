#!/usr/bin/env python

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


def obj_map(J, Q, R, dt):
    return np.linalg.inv(R*dt**(-2) + J.T @ Q @ J) @ J.T @ Q


def block_triangular_jacobian(model, q, v):
    J = model.jacobian(q)

    J11 = J[:1, :1]
    J12 = J[:1, 1:]
    J22 = J[1:, 1:]

    u = pseudoinverse(J).dot(v)

    v23 = v[1:]
    u23 = pseudoinverse(J22).dot(v23)
    u1 = (v[0] - J12.dot(u23)) / float(J11)

    u123 = np.array([u1[0], u23[0], u23[1]])

    return u, u123


def obj_vs_constr(model, q, v):
    J = model.jacobian(q)

    Q = np.eye(2) * 1e8
    R = 1 * np.eye(3)
    dt = 0.01

    u_constr = pseudoinverse(J).dot(v)
    u_obj = obj_map(J, Q, R, dt).dot(v)

    return u_constr, u_obj


def cost(model, q, v):
    J = model.jacobian(q)
    return v.T.dot(np.linalg.inv(J.dot(J.T))).dot(v)


def main():
    model = ThreeInputModel(L1, L2, LB, UB, output_idx=[0, 1])

    v = np.array([0.5, 0.5])
    q = np.array([0, 0.25*np.pi, -0.5*np.pi])

    u, u123 = block_triangular_jacobian(model, q, v)

    # v = np.array([0.5, 0.5])
    # q = np.array([0, 0.7*np.pi, np.pi/2])
    # C = cost(model, q, v)

    IPython.embed()


if __name__ == '__main__':
    main()
