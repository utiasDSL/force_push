import jax
import jax.numpy as jnp
import numpy as np
from functools import partial
from mm2d.util import bound_array


class TopDownHolonomicModelAD:
    ''' Holonomic top-down model. Five inputs: base x, y, and yaw velocity, and
        two arm joint velocities. This implementation uses automatic
        differentiation to compute the Jacobian. '''
    def __init__(self, l1, l2, vel_lim, acc_lim):
        self.ni = 5  # number of joints (inputs/DOFs)
        self.no = 2

        self.l1 = l1
        self.l2 = l2

        self.vel_lim = vel_lim
        self.acc_lim = acc_lim

        # Jacobian is calculated automatically using JAX auto-differentiation.
        self.jacobian = jax.jit(jax.jacobian(partial(self.forward, np=jnp)))

    def forward(self, q, np=np):
        ''' Forward kinematic transform for the end effector. '''
        xb, yb, θb, θ1, θ2 = q
        p = np.array([xb + self.l1*np.cos(θb+θ1) + self.l2*np.cos(θb+θ1+θ2),
                      yb + self.l1*np.sin(θb+θ1) + self.l2*np.sin(θb+θ1+θ2)])
        return p

    def step(self, q, u, dt, dq_last=None):
        ''' Step forward one timestep. '''
        # velocity limits
        dq = bound_array(u, -self.vel_lim, self.vel_lim)

        # acceleration limits
        if dq_last is not None:
            dq = bound_array(dq, -self.acc_lim * dt + dq_last, self.acc_lim * dt + dq_last)

        # if not (u == dq).all():
        #     print('limits hit')

        q = q + dt * dq
        return q, dq
