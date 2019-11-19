#!/usr/bin/env python2
from __future__ import print_function
import numpy as np
import numpy.matlib as matlib
import matplotlib.pyplot as plt
import qpoases

import IPython


class Pendulum(object):
    def __init__(self, dt):
        self.dt = dt
        self.g = 1.0
        self.l = 1.0

        self.m = 1  # state dimension
        self.n = 2  # input dimension
        self.p = 2  # output dimension

    def motion(self, x, u):
        ''' Motion model: x_k+1 = f(x_k, u_k) '''
        # TODO do I need to put modulo math in here?
        return x + self.dt * np.array([
            x[1],
            np.sin(x[0]) + u])

    def measure(self, x):
        ''' Measurement model: y_k = g(u_k) '''
        return x

    def calc_A(self, x):
        return np.array([
            [1, self.dt],
            [self.dt * self.g * np.cos(x[0]) / self.l, 1]])

    def calc_B(self, x):
        return np.array([[0], [1]])

    def calc_C(self, x):
        return np.eye(self.n)


def lookahead(sys, x0, Yd, U, Q, R, N):
    ''' Generate lifted matrices proprogating the state N timesteps into the
        future.

        sys: System model.
        x0:  Current state.
        Yd:  Desired output trajectory.
        U:   Input trajectory from the last iteration.
        Q:   Tracking error weighting matrix.
        R:   Input magnitude weighting matrix.
        N:   Number of timesteps into the future. '''

    n = sys.n
    m = sys.m
    p = sys.p


    # States from the last iteration
    X_last = np.zeros(n*(N+1))
    X_last[:n] = x0

    for k in range(1, N+1):
        x_prev = X_last[(k-1)*n:k*n]
        u_prev = U[(k-1)*m:k*m]
        x = sys.motion(x_prev, u_prev)
        X_last[k*n:(k+1)*n] = x

    # Arrays of the linearized matrices for each timestep.
    As = np.zeros((n*(N+1), n))
    Bs = np.zeros((n*N, m))
    Cs = np.zeros((p*(N+1), n))

    # Calculate B matrices: B_0..B_{N-1}
    for k in range(N):
        x = X_last[k*n:(k+1)*n]
        Bs[k*n:(k+1)*n, :] = sys.calc_B(x)

    # Calculate A and C matrices: A_1..A_N, C_1..C_N
    for k in range(1, N+1):
        x = X_last[k*n:(k+1)*n]
        As[k*n:(k+1)*n, :] = sys.calc_A(x)
        Cs[k*p:(k+1)*p, :] = sys.calc_C(x)

    Abar = np.zeros(p*N)
    Bbar = np.zeros((p*N, m*N))
    Qbar = np.zeros((p*N, p*N))
    Rbar = np.zeros((m*N, m*N))

    # Build Abar matrix
    for k in range(N):
        x = X_last[(k+1)*n:(k+2)*n]
        Abar[k*p:(k+1)*p] = sys.measure(x)

    # Build Bbar matrix
    for r in range(N):
        for c in range(r+1):
            B = Bs[c*n:(c+1)*n, :]
            C = Cs[(r+1)*p:(r+2)*p, :]
            A = np.eye(n)
            for i in range(c+1, r+1):
                A = As[i*n:(i+1)*n, :].dot(A)
            Bbar[r*p:(r+1)*p, c*m:(c+1)*m] = C.dot(A).dot(B)

    # Build lifted weight matrices
    for k in range(N):
        Qbar[k*p:(k+1)*p, k*p:(k+1)*p] = Q
        Rbar[k*m:(k+1)*m, k*m:(k+1)*m] = R

    H = Rbar + Bbar.T.dot(Qbar).dot(Bbar)
    g = (Abar - Yd).T.dot(Qbar).dot(Bbar)

    return H, g


class MPC(object):
    ''' Model predictive controller. '''
    def __init__(self, sys, Q, R, lb, ub):
        self.sys = sys
        self.Q = Q
        self.R = R
        self.lb = lb
        self.ub = ub

    def _lookahead(self, x0, Yd, U, N):
        return lookahead(self.sys, x0, Yd, U, self.Q, self.R, N)

    def _iterate(self, x0, Yd, U, N):
        # Create the QP, which we'll solve sequentially.
        qp = qpoases.PySQProblem(self.sys.m * N, self.sys.m * N)
        options = qpoases.PyOptions()
        options.printLevel = qpoases.PyPrintLevel.NONE
        qp.setOptions(options)

        # TODO put somewhere else
        nWSR = 100
        NUM_ITER = 10

        # Initial opt problem.
        H, g = self._lookahead(x0, Yd, U, N)
        lb = np.ones(N) * self.lb - U
        ub = np.ones(N) * self.ub - U
        qp.init(H, g, None, lb, ub, None, None, nWSR)
        dU = np.zeros(self.sys.m * N)
        qp.getPrimalSolution(dU)
        U = U + dU

        # Remaining sequence is hotstarted from the first.
        for i in range(NUM_ITER):
            H, g = self._lookahead(x0, Yd, U, N)

            # we currently do not have a constraint matrix A
            qp.hotstart(H, g, None, self.lb - U, self.ub - U, None, None, nWSR)
            qp.getPrimalSolution(dU)

            # TODO we could have a different step size here, since the
            # linearization is only locally valid
            U = U + dU

        # obj_val = qp.getObjVal()
        return U

    def solve(self, x0, Yd, N):
        ''' Solve the MPC problem at current state x0 given desired output
            trajectory Yd. '''
        # initialize optimal inputs
        U = np.zeros(self.sys.m * N)

        # iterate to final solution
        U = self._iterate(x0, Yd, U, N)

        # return first optimal input
        return U[:self.sys.m]


def step(t):
    ''' Desired trajectory '''
    # step
    if t < 1.0:
        return 0.0
    return 1.0


def main():
    N = 10
    dt = 0.01
    tf = 10.0
    num_steps = int(tf / dt)

    Q = np.eye(1) * 10
    R = np.eye(1)
    lb = -1.0
    ub = 1.0

    Kp = 1.0
    Ki = 0.1
    Kd = 2.0

    sys = Pendulum(dt)
    mpc = MPC(sys, Q, R, lb, ub)

    # desired trajectory is to just stabilize
    # Yd = np.ones(num_steps) * np.pi
    Yd = np.array([np.pi if i % 2 == 0 else 0 for i in xrange(num_steps * 2)])

    ts = np.array([i * dt for i in xrange(num_steps)])
    ys = np.zeros(num_steps)
    xs = np.zeros((num_steps, 2))
    us = np.zeros(num_steps)

    x = np.zeros(2)

    E = 0

    for i in xrange(num_steps - 1):
        # y = ys[i]
        x = xs[i, :]

        n = min(N, num_steps - i)
        u = mpc.solve(x, Yd[i*2:(i+n)*2], n)

        # PID control
        # e = Yd[i] - y
        # de = -x[1]  # assume ref velocity is 0
        # E += dt * e
        # u = Kp*e + Kd*de + Ki*E

        x = sys.motion(x, u)
        # y = sys.measure(x)

        us[i] = u
        xs[i+1, :] = x
        # ys[i+1] = y

    plt.plot(ts, xs[:, 0], label='x1')
    plt.plot(ts, xs[:, 1], label='x2')
    plt.plot(ts, us, label='u')
    plt.grid()
    plt.legend()
    plt.show()


if __name__ == '__main__':
    main()
