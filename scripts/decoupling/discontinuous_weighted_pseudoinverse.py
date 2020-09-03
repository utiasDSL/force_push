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


def weighted_ps(D, J):
    A = np.diag(D)
    return pseudoinverse(A @ J) @ A


def main():
    model = ThreeInputModel(L1, L2, LB, UB, output_idx=[0, 1])

    v = np.array([0.5, 0.5])
    q = np.array([0, 0.25*np.pi, -0.5*np.pi])

    J = model.jacobian(q)
    Jps = pseudoinverse(J)

    IPython.embed()


if __name__ == '__main__':
    main()
