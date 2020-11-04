import numpy as np
import jax
import jax.numpy as jnp
from mm2d.util import bound_array

import IPython


class ThreeInputModel:
    ''' Three-input 2D mobile manipulator. Consists of mobile base (1 input)
        and 2-link arm (2 inputs). State is q = [x_b, q_1, q_2]; inputs u = dq. '''
    def __init__(self, l1, l2, vel_lim, acc_lim):
        self.ni = 3  # number of joints (inputs/DOFs)
        self.no = 3  # possible outputs are: x, y, theta

        self.l1 = l1
        self.l2 = l2

        self.vel_lim = vel_lim
        self.acc_lim = acc_lim

        Z = np.zeros((3, 3))
        self.A = np.block([[Z, np.eye(3)], [Z, Z]])
        self.B = np.block([[Z], [np.eye(3)]])

        self.jacobian = jax.jit(jax.jacrev(self.ee_position))
        self.dJdq = jax.jit(jax.jacfwd(self.jacobian))

    def ee_position(self, X):
        q = X[:3]
        p = jnp.array([q[0] + self.l1*jnp.cos(q[1]) + self.l2*jnp.cos(q[1]+q[2]),
                             self.l1*jnp.sin(q[1]) + self.l2*jnp.sin(q[1]+q[2]),
                      q[1] + q[2]])
        return p

    def ee_velocity(self, X):
        q, dq = X[:3], X[3:]
        return self.jacobian(q) @ dq

    def ee_acceleration(self, X, u):
        q, dq = X[:3], X[3:]
        return self.jacobian(q) @ u + dq @ self.dJdq(q) @ dq

    def ee_state(self, X):
        return jnp.concatenate((self.ee_position(X), self.ee_velocity(X)))

    def step_unconstrained(self, Q, u, dt):
        ''' Step forward one timestep without applying state or input
            constraints. '''
        dQ = self.A @ Q + self.B @ u
        Q = Q + dt * dQ
        return Q

    # def step(self, X, u, dt):
    #     ''' Step forward one timestep. State constraints are applied to the
    #         output state but not the supplied input state. '''
    #     # velocity limits
    #     dq = bound_array(u, -self.vel_lim, self.vel_lim)
    #
    #     # acceleration limits
    #     if dq_last is not None:
    #         dq = bound_array(dq, -self.acc_lim * dt + dq_last, self.acc_lim * dt + dq_last)
    #
    #     q = q + dt * dq
    #     return q, dq


def main():
    model = ThreeInputModel(1, 1, 1, 1)
    q1 = np.array([0., 0., 0.])
    q2 = np.array([1., 0.25*np.pi, -0.5*np.pi])
    IPython.embed()


if __name__ == '__main__':
    main()
