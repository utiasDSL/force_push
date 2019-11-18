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
        self.p = 1  # output dimension

    def motion(self, x, u):
        ''' Motion model: x_k+1 = f(x_k, u_k) '''
        return x + self.dt * np.array([
            x[1],
            self.g / self.l * np.sin(x[0]) + u])

    def measure(self, x):
        ''' Measurement model: y_k = g(u_k) '''
        return x[0]

    def calc_A(self, x):
        return np.array([
            [1, self.dt],
            [self.dt * self.g * np.cos(x[0]) / self.l, 1]])

    def calc_B(self, x):
        return np.array([[0], [1]])

    def calc_C(self, x):
        return np.array([[1, 0]])


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

    Abar = np.zeros((p*N, n))
    Bbar = np.zeros((p*N, m*N))
    Qbar = np.zeros((p*N, p*N))
    Rbar = np.zeros((m*N, m*N))

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

    Abar = np.zeros((p*N, 1))
    Bbar = np.zeros((p*N, m*N))

    # Build Abar matrix
    for k in range(1, N+1):
        x = X_last[k*n:(k+1)*n]
        Abar[k*p:k*(p+1)] = sys.measure(x)

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

    return np.array(H), np.array(g).flatten()


class LinearSystem(object):
    def __init__(self, x0, A, B, C):
        self.A = A
        self.B = B
        self.C = C

        self.u = 0
        self.x = x0

        self.n = A.shape[0]  # state dimension
        self.m = B.shape[1]  # input dimension
        self.p = C.shape[0]  # output dimension

    def apply(self, u):
        if u < -1.0:
            u = -1.0
        elif u > 1.0:
            u = 1.0
        self.u = u

    def step(self):
        self.x = self.A * self.x + self.B * self.u

    def lookahead(self, x0, Yd, Q, R, N):
        Abar = matlib.zeros((self.p*N, self.n))
        Bbar = matlib.zeros((self.p*N, self.m*N))
        Qbar = matlib.zeros((self.p*N, self.p*N))
        Rbar = matlib.zeros((self.m*N, self.m*N))

        # TODO right now we don't include a separate QN for terminal condition
        # Construct matrices
        for k in range(N):
            l = k*self.p
            u = (k+1)*self.p

            Abar[k*self.p:(k+1)*self.p, :] = self.C*self.A**(k+1)
            Qbar[k*self.p:(k+1)*self.p, k*self.p:(k+1)*self.p] = Q
            Rbar[k*self.m:(k+1)*self.m, k*self.m:(k+1)*self.m] = R

            for j in range(k + 1):
                Bbar[k*self.p:(k+1)*self.p, j*self.m:(j+1)*self.m] = self.C*self.A**(k-j-1)*self.B

        # TODO note H is independt of x0 and Yd, and is thus constant at each
        # step
        H = Rbar + Bbar.T * Qbar * Bbar
        g = (Abar * x0 - Yd).T * Qbar * Bbar

        return np.array(H), np.array(g).flatten()


class PID(object):
    ''' PID controller. '''
    def __init__(self, Kp, Ki, Kd):
        self.Kp = Kp
        self.Ki = Ki
        self.Kd = Kd

        self.e_prev = 0
        self.e_int = 0

    def control(self, e, dt):
        ''' Generate control signal u to drive e to zero. '''
        de = (e - self.e_prev) / dt
        self.e_prev = e
        self.e_int += dt * e
        u = self.Kp * e + self.Kd * de + self.Ki * self.e_int
        return u


class MPC(object):
    ''' Model predictive controller. '''
    def __init__(self, sys, Q, R, lb, ub, N):
        self.sys = sys
        self.Q = Q
        self.R = R
        self.lb = lb
        self.ub = ub
        self.N = N

    def _lookahead(self, x0, Yd, U):
        return lookahead(self.sys, x0, Yd, U, self.Q, self.R, self.N)

    def _iterate(self, x0, Yd, U):
        # Create the QP, which we'll solve sequentially.
        qp = qpoases.PySQProblem(self.sys.m * self.N, self.sys.m * self.N)
        options = qpoases.PyOptions()
        options.printLevel = qpoases.PyPrintLevel.NONE
        qp.setOptions(options)

        # TODO put somewhere else
        nWSR = 100
        NUM_ITER = 10

        # Initial opt problem.
        H, g = self._lookahead(x0, Yd, U)
        qp.init(H, g, None, self.lb - U, self.ub - U, None, None, nWSR)
        dU = np.zeros(self.sys.m * self.N)
        qp.getPrimalSolution(dU)
        U = U + dU

        # Remaining sequence is hotstarted from the first.
        for i in range(NUM_ITER):
            H, g = self._lookahead(x0, Yd, U)
            # we currently do not have a constraint matrix A
            qp.hotstart(H, g, None, self.lb - U, self.ub - U, None, None, nWSR)
            qp.getPrimalSolution(dU)
            # TODO we could have a different step size here, since the
            # linearization is only locally valid
            U = U + dU

        return U

    def solve(self, x0, Yd):
        # initialize optimal inputs
        U = np.zeros(self.sys.p*self.N)
        U = self._iterate(x0, Yd, U)

        # return first optimal input
        return U[:self.sys.p]


def step(t):
    ''' Desired trajectory '''
    # step
    if t < 1.0:
        return 0.0
    return 1.0


def main3():
    N = 10
    dt = 0.01
    tf = 10.0
    num_steps = int(tf / dt)

    Q = np.eye(1) * 10
    R = np.eye(1)
    lb = np.ones(N) * -1.0
    ub = np.ones(N) * 1.0

    sys = Pendulum(dt)
    mpc = MPC(sys, Q, R, lb, ub, N)

    # desired trajectory is to just stabilize
    Yd = np.ones(num_steps) * np.pi

    ts = np.zeros(num_steps)
    ys = np.zeros(num_steps)

    x = np.zeros(2)

    for i in xrange(num_steps - 1):
        y = ys[i]

        u = mpc.solve(y, Yd[i:i+N])
        print(u)
        x = sys.motion(x, u)
        y = sys.measure(x)

        ys[i+1] = y
        ts[i+1] = i * dt

    plt.plot(ts, ys)
    plt.show()


def main2():
    n = 2  # state dimension
    m = 1  # input dimension
    p = 1  # output dimension

    N = 10  # number of lookahead steps

    dt = 0.1
    tf = 10.0
    num_steps = int(tf / dt)

    pid = PID(Kp=1, Ki=0.1, Kd=1)

    # x+ = Ax + Bu
    # y  = Cx
    A = np.matrix([[1, dt], [0, 1]])
    B = np.matrix([[0], [dt]])
    C = np.matrix([1, 0])
    x0 = np.matrix([[0], [0]])

    sys1 = LinearSystem(x0, A, B, C)
    sys2 = LinearSystem(x0, A, B, C)

    # V = x'Qx + u'Ru
    Q = matlib.eye(p) * 10
    R = matlib.eye(p)
    lb = np.ones(N) * -1.0
    ub = np.ones(N) * 1.0
    # nWSR = np.array([100])
    nWSR = 100

    t = np.zeros(num_steps)
    x1 = np.zeros(sys1.n * num_steps)
    x2 = np.zeros(sys2.n * num_steps)
    yd = np.zeros(sys1.p * num_steps)

    qp = qpoases.PyQProblemB(sys2.m*N)
    options = qpoases.PyOptions()
    options.printLevel = qpoases.PyPrintLevel.NONE
    qp.setOptions(options)

    Yd = np.matrix([[step(t[0] + dt * j)] for j in xrange(N)])
    H, g = sys2.lookahead(x0, Yd, Q, R, N)
    qp.init(H, g, lb, ub, nWSR)

    for i in xrange(num_steps - 1):
        # desired trajectory
        yd[i] = step(t[i])

        # PID control
        x0 = np.matrix(x1[i*n:(i+1)*n]).T
        e = yd[i] - sys1.C*x0
        u = pid.control(e, dt)

        sys1.apply(u)
        sys1.step()

        # MPC
        x0 = np.matrix(x2[i*n:(i+1)*n]).T
        Yd = np.matrix([[step(t[i] + dt * j)] for j in xrange(N)])
        _, g = sys2.lookahead(x0, Yd, Q, R, N)
        qp.hotstart(g, lb, ub, nWSR)
        U = np.zeros(sys2.m * N)
        qp.getPrimalSolution(U)
        u = U[0]

        sys2.apply(u)
        sys2.step()

        x1[(i+1)*sys1.n:(i+2)*sys1.n] = np.array(sys1.x).flatten()
        x2[(i+1)*sys2.n:(i+2)*sys2.n] = np.array(sys2.x).flatten()
        t[i + 1] = t[i] + dt

    plt.subplot(211)
    plt.plot(t, yd, label='desired')
    plt.plot(t, x1[::2], label='actual')
    plt.title('PID control')
    plt.legend()

    plt.subplot(212)
    plt.plot(t, yd, label='desired')
    plt.plot(t, x2[::2], label='actual')
    plt.title('MPC')
    plt.legend()
    plt.xlabel('Time (s)')

    plt.show()


def main():
    H = np.array([[1, 0],
                  [0, 0.5]])
    g = np.array([1.5, 1.0])
    lb = np.array([0.5, -2.0])
    ub = np.array([5.0, 2.0])

    # use QProblemB when lbA <= Ax <= ubA constraints are not present
    # argument is the number of decision variables
    qp = qpoases.PyQProblemB(2)

    nWSR = 10  # number of working set recomputations
    qp.init(H, g, lb, ub, nWSR)


if __name__ == '__main__':
    main3()
