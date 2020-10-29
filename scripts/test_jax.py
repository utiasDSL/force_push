#!/usr/bin/env python
import jax.numpy as np
import jax

import IPython


# robot parameters
L1 = 1
L2 = 1


def unroll(q0, u):
    n = u.shape[0] + 1
    q = np.zeros(n)
    q = jax.ops.index_update(q, jax.ops.index[0], q0)
    for i in range(n - 1):
        q = jax.ops.index_update(q, jax.ops.index[i+1], q[i] + u[1])
    return q


def unroll_forward(q0, dt, u):
    ni = q0.shape[0]
    n = 3
    # q = np.zeros(ni * (n+1))
    p = np.zeros(2 * (n+1))
    # q = jax.ops.index_update(q, jax.ops.index[:ni], q0)
    q = q0
    p = jax.ops.index_update(p, jax.ops.index[:2], forward(q))
    for i in range(n):
        q = q + dt*u[i*ni:(i+1)*ni]
        p = jax.ops.index_update(p, jax.ops.index[(i+1)*2:(i+2)*2], forward(q))
        # q = jax.ops.index_update(q, jax.ops.index[(i+1)*ni:(i+2)*ni], q[i*ni:(i+1)*ni] + dt*u[i*ni:(i+1)*ni])
    return p


def forward(q):
    return np.array([q[0] + L1*np.cos(q[1]) + L2*np.cos(q[1]+q[2]),
                     L1*np.sin(q[1]) + L2*np.sin(q[1]+q[2])])


def jacobian(q):
    return np.array([
        [1, -L1*np.sin(q[1])-L2*np.sin(q[1]+q[2]), -L2*np.sin(q[1]+q[2])],
        [0,  L1*np.cos(q[1])+L2*np.cos(q[1]+q[2]),  L2*np.cos(q[1]+q[2])]])


# def main():
    # model = models.ThreeInputModel(L1, L2, VEL_LIM, acc_lim=ACC_LIM, output_idx=[0, 1])
# q1 = np.zeros(3)
# q2 = np.array([1, 0.25*np.pi, -0.5*np.pi])
# jax_jacobian = jax.jacfwd(forward)
# jit_jacobian = jax.jit(jax_jacobian)
n = 3
u = np.ones(3 * n)
q0 = np.zeros(3)
p = unroll_forward(q0, 1, u)
J = jax.jit(jax.jacfwd(unroll_forward, argnums=2))
IPython.embed()


if __name__ == '__main__':
    main()
