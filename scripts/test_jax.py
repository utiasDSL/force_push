#!/usr/bin/env python
import numpy as np
import jax.numpy as jnp
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
    n = u.shape[0]
    p = jnp.zeros((n, 2))
    q = q0
    for i in range(n):
        p = jax.ops.index_update(p, jax.ops.index[i, :], forward(q))
        q = q + dt * u[i, :]
    return p

def unroll_forward_lax(q0, dt, u):
    # TODO actually about twice as slow as above on GPU
    def scan_func(qi, ui):
        pi = forward(qi)
        qi = qi + dt * ui
        return qi, pi
    _, p = jax.lax.scan(scan_func, q0, u)
    return p


def forward(q, np=jnp):
    return np.array([q[0] + L1*np.cos(q[1]) + L2*np.cos(q[1]+q[2]),
                     L1*np.sin(q[1]) + L2*np.sin(q[1]+q[2])])


def jacobian(q, np=jnp):
    return np.array([
        [1, -L1*np.sin(q[1])-L2*np.sin(q[1]+q[2]), -L2*np.sin(q[1]+q[2])],
        [0,  L1*np.cos(q[1])+L2*np.cos(q[1]+q[2]),  L2*np.cos(q[1]+q[2])]])


# def main():
# model = models.ThreeInputModel(L1, L2, VEL_LIM, acc_lim=ACC_LIM, output_idx=[0, 1])
# q1 = np.zeros(3)
# q2 = np.array([1, 0.25*np.pi, -0.5*np.pi])
jax_jacobian = jax.jit(jax.jacfwd(forward))
# jit_jacobian = jax.jit(jax_jacobian)

dt = 0.1
n = 3
u = np.ones((n, 3))
q0 = np.zeros(3)
p = unroll_forward(q0, 1, u)
J1 = jax.jit(jax.jacfwd(unroll_forward, argnums=2))
J2 = jax.jit(jax.jacfwd(unroll_forward_lax, argnums=2))

IPython.embed()


# if __name__ == '__main__':
#     main()
