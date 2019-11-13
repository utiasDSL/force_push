#!/usr/bin/env python2
import numpy as np
import numpy.matlib as matlib
import matplotlib.pyplot as plt
import qpoases

import IPython


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
    def __init__(self, Q, R, lb, ub, N):
        self.Q = Q
        self.R = R
        self.lb = lb
        self.ub = ub


def step(t):
    ''' Desired trajectory '''
    # step
    if t < 1.0:
        return 0.0
    return 1.0


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

    IPython.embed()


if __name__ == '__main__':
    main2()
