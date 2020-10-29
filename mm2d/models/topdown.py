import jax
import jax.numpy as jnp
import numpy as np
from mm2d.util import bound_array


class TopDownHolonomicModel:
    ''' Holonomic top-down model. Four inputs: base x and y velocity, and two
        arm joint velocities. '''
    def __init__(self, l1, l2, vel_lim, acc_lim, output_idx=[0,1,2]):
        self.ni = 5  # number of joints (inputs/DOFs)

        # control which outputs are used
        # possible outputs are: x, y, theta
        self.no = len(output_idx)
        self.output_idx = output_idx

        self.l1 = l1
        self.l2 = l2

        self.vel_lim = vel_lim
        self.acc_lim = acc_lim

    def forward(self, q):
        ''' Forward kinematic transform for the end effector. '''
        xb, yb, θb, θ1, θ2 = q
        p = np.array([xb + self.l1*np.cos(θb+θ1) + self.l2*np.cos(θb+θ1+θ2),
                      yb + self.l1*np.sin(θb+θ1) + self.l2*np.sin(θb+θ1+θ2),
                      θb + θ1 + θ2])
        return p[self.output_idx]

    def forward_f(self, q):
        pb = q[:2]
        θb = q[2]
        rx = 0.5
        pf = pb + np.array([rx*np.cos(θb), -rx*np.sin(θb)])
        return pf

    def forward_m(self, q):
        pf = self.forward_f(q)
        pe = self.forward(q)
        pm = 0.5*(pf + pe)
        return pm

    def jacobian(self, q):
        ''' End effector Jacobian. '''
        _, _, θb, θ1, θ2 = q
        dp1dθb = -self.l1*np.sin(θb+θ1)-self.l2*np.sin(θb+θ1+θ2)
        dp1dθ1 = dp1dθb
        dp2dθb = self.l1*np.cos(θb+θ1)+self.l2*np.cos(θb+θ1+θ2)
        dp2dθ1 = dp2dθb
        J = np.array([
            [1, 0, dp1dθb, dp1dθ1, -self.l2*np.sin(θb+θ1+θ2)],
            [0, 1, dp2dθb, dp2dθ1,  self.l2*np.cos(θb+θ1+θ2)],
            [0, 0, 1, 1, 1]])
        return J[self.output_idx, :]

    def jacobian_f(self, q):
        θb = q[2]
        rx = 0.5
        Jf = np.array([[1, 0, -rx*np.sin(θb), 0, 0],
                       [0, 1, -rx*np.cos(θb), 0, 0]])
        return Jf

    def jacobian_m(self, q):
        Jf = self.jacobian_f(q)
        Je = self.jacobian(q)
        Jm = 0.5*(Jf + Je)
        return Jm

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
