#!/usr/bin/env python2
from __future__ import print_function
import numpy as np
import matplotlib.pyplot as plt
import qpoases

import IPython


L1 = 1
L2 = 2


class MM(object):
    def __init__(self):
        pass

    def forward(self, q):
        return np.array([q[0] + L1*np.cos(q[1]) + L2*np.cos(q[1]+q[2]),
                                L1*np.sin(q[1]) + L2*np.sin(q[1]+q[2])])

    def jacobian(self, q):
        return np.array([
            [1, -L1*np.sin(q[1])-L2*np.sin(q[1]+q[2]), -L2*np.sin(q[1]+q[2])],
            [0,  L1*np.cos(q[1])-L2*np.cos(q[1]+q[2]), -L2*np.cos(q[1]+q[2])]])


class MPC(object):
    ''' Model predictive controller. '''
    def __init__(self, model, dt, Q, R, lb, ub):
        self.model = model
        self.dt = dt
        self.Q = Q
        self.R = R
        self.lb = lb
        self.ub = ub

    def _lookahead(self, q0, pr, dq, N):
        ''' Generate lifted matrices proprogating the state N timesteps into the
            future.

            sys: System model.
            q0:  Current joint positions.
            pr:  Desired Cartesian output trajectory.
            dq:  Input joint velocties from the last iteration.
            Q:   Tracking error weighting matrix.
            R:   Input magnitude weighting matrix.
            N:   Number of timesteps into the future. '''

        n = 3  # number of joints
        p = 2  # number of Cartesian outputs

        qbar = np.zeros(n*(N+1))
        qbar[:n] = q0

        # Integrate joint positions from the last iteration
        for k in range(1, N+1):
            q_prev = qbar[(k-1)*n:k*n]
            dq_prev = dq[(k-1)*n:k*n]
            q = q_prev + self.dt * dq_prev

            qbar[k*n:(k+1)*n] = q

        fbar = np.zeros(p*N)         # Lifted forward kinematics
        Jbar = np.zeros((p*N, n*N))  # Lifted Jacobian
        Qbar = np.zeros((p*N, p*N))
        Rbar = np.zeros((n*N, n*N))

        for k in range(N):
            q = qbar[(k+1)*n:(k+2)*n]

            fbar[k*np:(k+1)*np] = self.model.forward(q)
            Jbar[k*np:(k+1)*np, k*n:(k+1)*n] = 0

            Qbar[k*p:(k+1)*p, k*p:(k+1)*p] = self.Q
            Rbar[k*n:(k+1)*n, k*n:(k+1)*n] = self.R

        dbar = fbar - pr

        H = Rbar + self.dt**2*Jbar.T.dot(Qbar).dot(Jbar)
        g = dq.T.dot(Rbar) + self.dt*dbar.T.dot(Qbar).dot(Jbar)

        return H, g

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


def main():
    N = 10
    dt = 0.05
    tf = 10.0
    num_steps = int(tf / dt)

    sys = Pendulum(dt)

    Q = np.diag([1, 0.1])
    R = np.eye(sys.m) * 0.1
    lb = -1.0
    ub = 1.0

    Kp = 1.0
    Ki = 0.1
    Kd = 2.0

    mpc = MPC(sys, Q, R, lb, ub)

    ts = np.array([i * dt for i in xrange(num_steps)])
    ys = np.zeros((num_steps, sys.p))
    xs = np.zeros((num_steps, sys.n))
    us = np.zeros(num_steps)

    # desired trajectory
    pd = np.array([np.pi if ts[i] > 1 else 0 for i in xrange(num_steps)])
    vd = np.zeros(num_steps)
    Yd = np.zeros(num_steps * 2)
    Yd[::2] = pd
    Yd[1::2] = vd

    x = np.zeros(sys.n)

    E = 0

    for i in xrange(num_steps - 1):
        y = ys[i, :]
        x = xs[i, :]

        n = min(N, num_steps - i)
        u = mpc.solve(x, Yd[i*sys.p:(i+n)*sys.p], n)

        # PID control
        # e = Yd[i*2] - x[0]
        # de = Yd[i*2+1] - x[1]
        # E += dt * e
        # u = Kp*e + Kd*de + Ki*E

        # bound u
        if u < -1.0:
            u = -1.0
        elif u > 1.0:
            u = 1.0

        x = sys.motion(x, u)
        y = sys.measure(x)

        us[i] = u
        xs[i+1, :] = x
        ys[i+1, :] = y

    plt.plot(ts, pd, label='$\\theta_d$', color='k', linestyle='--')
    plt.plot(ts, xs[:, 0], label='$\\theta$')
    plt.plot(ts, xs[:, 1], label='$\dot{\\theta}$')
    plt.plot(ts, us, label='$u$')
    plt.grid()
    plt.legend()
    plt.show()


if __name__ == '__main__':
    main()
