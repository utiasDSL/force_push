#!/usr/bin/env python
import jax.numpy as np
import jax

import IPython


# robot parameters
L1 = 1
L2 = 1
VEL_LIM = 1
ACC_LIM = 1


def forward(q):
    return np.array([q[0] + L1*np.cos(q[1]) + L2*np.cos(q[1]+q[2]),
                     L1*np.sin(q[1]) + L2*np.sin(q[1]+q[2])])


def jacobian(q):
    return np.array([
        [1, -L1*np.sin(q[1])-L2*np.sin(q[1]+q[2]), -L2*np.sin(q[1]+q[2])],
        [0,  L1*np.cos(q[1])+L2*np.cos(q[1]+q[2]),  L2*np.cos(q[1]+q[2])]])


def main():
    # model = models.ThreeInputModel(L1, L2, VEL_LIM, acc_lim=ACC_LIM, output_idx=[0, 1])
    q1 = np.zeros(3)
    q2 = np.array([1, 0.25*np.pi, -0.5*np.pi])
    auto_jacobian = jax.jacobian(forward)
    IPython.embed()


if __name__ == '__main__':
    main()
