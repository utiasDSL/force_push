#!/usr/bin/env python

import numpy as np
np.set_printoptions(precision=3, suppress=True)
import sympy
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


def cost(J, v):
    return v.T.dot(np.linalg.inv(J.dot(J.T))).dot(v)


def rot(a):
    c = np.cos(a)
    s = np.sin(a)
    return np.array([[c, -s], [s, c]])


def li(A):
    ''' Check linear independence of rows of A. '''
    return sympy.Matrix(A).T.rref()


def decouple(J, v, P, inspect=False):
    PJ = P @ J
    Pv = P @ v

    # selection matrix with elementwise multiplication
    S = np.array([[0, 1, 1], [1, 0, 0]])

    # constraint matrix
    C = S * PJ

    A = np.vstack((PJ, C))
    b = np.append(Pv, [0, 0])

    if inspect:
        IPython.embed()

    u_con = np.linalg.solve(A, b)
    C_con = cost(A, b)

    return u_con, C_con


def main():
    np.set_printoptions(precision=3, suppress=True)

    model = ThreeInputModel(L1, L2, LB, UB, output_idx=[0, 1])

    v = np.array([0.5, 0.5])
    q = np.array([0, 0.25*np.pi, -0.5*np.pi])

    J = model.jacobian(q)

    u_free = pseudoinverse(J) @ v
    C_free = cost(J, v)

    # best_a = 0
    # _, best_C = decouple(J, v, rot(best_a))
    #
    # for _ in range(100):
    #     a = (np.random.rand() * 2 - 1) * np.pi
    #     P = rot(a)
    #     u_con, C_con = decouple(J, v, P)
    #
    #     if C_con < best_C:
    #         best_a = a
    #         best_C = C_con

    # found by the above, this solution appears to give nearly the same results
    # as the unconstrained case
    a = 0.395297484101207
    P = rot(a)
    u_con, C_con = decouple(J, v, P, inspect=True)

    IPython.embed()


def main2():
    v = np.array([1, 2])
    J = np.array([[1, 2, 3, 4], [5, 6, 7, 8]])
    C = np.array([[0, 2, 3, 4], [5, 0, 0, 0]])

    A = np.vstack((J, C))
    b = np.append(v, [0, 0])

    A0 = A[:3, :]
    a = A[3, :]

    # we know a is a linear combination of rows of A0: a = A0^T @ a0
    # use pseudoinverse of A0
    a0 = np.linalg.inv(A0 @ A0.T) @ A0 @ a

    # A = P @ A0 -- since A is rank deficient, we know it can be decomposed
    # into the product of two thinner matrices
    P = np.vstack((np.eye(3), a0[None, :]))

    bp = np.linalg.inv(P.T @ P) @ P.T @ b

    # now the constraint is A0 @ u = bp

    IPython.embed()


def main3():
    J = np.array([[1, 2, 3, 4], [5, 6, 7, 8]])
    Q1, R1 = np.linalg.qr(J)
    _, R2 = np.linalg.qr(J.T)
    L = R2.T

    IPython.embed()

if __name__ == '__main__':
    main3()
