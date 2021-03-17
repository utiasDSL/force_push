import numpy as np
import jax
import jax.numpy as jnp
from scipy.linalg import expm
from mm2d.util import bound_array

import IPython


# geometry
L1 = 1
L2 = 1
Bw = 1.0
Bh = 0.25

# mass
Mb = 10
M1 = 1
M2 = 1

# limits
VEL_LIM = 1
ACC_LIM = 1
TAU_LIM = 100


def zoh(A, B, dt):
    """Compute discretized system matrices assuming zero-order hold on imput."""
    ra, ca = A.shape
    rb, cb = B.shape

    assert ra == ca  # A is square
    assert ra == rb  # B has same number of rows as A

    ch = ca + cb
    rh = ch

    H = np.block([[A, B], [np.zeros((rh - ra, ch))]])
    Hd = expm(dt * H)
    Ad = Hd[:ra, :ca]
    Bd = Hd[:rb, ca:ca+cb]

    return Ad, Bd


class FourInputModel:
    """Four-input 2D velocity-controlled mobile manipulator."""
    def __init__(self, l1=L1, l2=L2, vel_lim=VEL_LIM, acc_lim=ACC_LIM):
        self.ni = 4  # number of joints (inputs/DOFs)
        self.no = 3  # possible outputs are: x, y, theta

        self.lx = 0
        self.ly = 0
        self.l1 = l1
        self.l2 = l2
        self.bw = Bw
        self.bh = Bh

        # link masses
        self.mb = Mb
        self.m1 = M1
        self.m2 = M2

        # link inertias
        self.I1 = self.m1 * l1**2 / 12
        self.I2 = self.m2 * l2**2 / 12

        self.vel_lim = vel_lim
        self.acc_lim = acc_lim

        Z = np.zeros((self.ni, self.ni))
        self.A = np.block([[Z, np.eye(self.ni)], [Z, Z]])
        self.B = np.block([[Z], [np.eye(self.ni)]])

        # compute discretized matrices assuming zero-order hold
        self.Ad, self.Bd = zoh(self.A, self.B, 0.1)

        self.jacobian = jax.jit(jax.jacrev(self.ee_position))
        self.dJdq = jax.jit(jax.jacfwd(self.jacobian))

    def base_corners(self, q):
        """Calculate the corners of the base of the robot."""
        x0 = q[0]
        y0 = 0
        r = self.bw * 0.5
        h = self.bh

        x = np.array([x0 - r, x0 - r, x0 + r, x0 + r])
        y = np.array([y0, y0 - h, y0 - h, y0])

        return x, y

    def arm_points(self, q):
        """Calculate points on the arm."""
        x0 = q[0] + self.lx
        x1 = x0 + self.l1*np.cos(q[1])
        x2 = x1 + self.l2*np.cos(q[1]+q[2])

        y0 = self.ly
        y1 = y0 + self.l1*np.sin(q[1])
        y2 = y1 + self.l2*np.sin(q[1]+q[2])

        x = np.array([x0, x1, x2])
        y = np.array([y0, y1, y2])

        return x, y

    def ee_position(self, X):
        q = X[:self.ni]
        p = jnp.array([q[0] + self.l1*jnp.cos(q[1]) + self.l2*jnp.cos(q[1]+q[2]),
                             self.l1*jnp.sin(q[1]) + self.l2*jnp.sin(q[1]+q[2]),
                      q[1] + q[2] + q[3]])
        return p

    def ee_velocity(self, X):
        q, dq = X[:self.ni], X[self.ni:]
        return self.jacobian(q) @ dq

    def ee_acceleration(self, X, u):
        q, dq = X[:self.ni], X[self.ni:]
        return self.jacobian(q) @ u + dq @ self.dJdq(q) @ dq

    def ee_state(self, X):
        return jnp.concatenate((self.ee_position(X), self.ee_velocity(X)))

    def step_unconstrained(self, Q, u, dt, method="zoh"):
        """Step forward one timestep without applying state or input constraints.

        Available integration methods are: euler, rk4, zoh
        """
        if method == "euler":
            # euler integration
            dQdt = self.A @ Q + self.B @ u
            Q_new = Q + dt * dQdt
        elif method == "rk4":
            # fourth-order explicit Runge-Kutta integration method
            dQdt = self.A @ Q + self.B @ u
            k1 = dQdt
            k2 = dQdt + 0.5 * dt * k1
            k3 = dQdt + 0.5 * dt * k2
            k4 = dQdt + k3
            Q_new = Q + dt * (k1 + 2 * k2 + 2 * k3 + k4) / 6.0
        elif method == "zoh":
            # exact solution assuming zero-order hold on input (i.e. that it is
            # constant over the time step)
            Q_new = self.Ad @ Q + self.Bd @ u
        else:
            raise Exception(f"invalid method: {method}")
        return Q_new


def main():
    model = FourInputModel(1, 1, 1, 1)
    q1 = np.array([0., 0., 0.])
    q2 = np.array([1., 0.25*np.pi, -0.5*np.pi])
    IPython.embed()


if __name__ == '__main__':
    main()
