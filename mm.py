#!/usr/bin/env python2
from __future__ import print_function
import numpy as np
import matplotlib.pyplot as plt
import qpoases

import IPython


L1 = 1
L2 = 1

NUM_WSR = 100
NUM_ITER = 10


def bound_array(a, lb, ub):
    ''' Elementwise bound array above and below. '''
    return np.minimum(np.maximum(a, lb), ub)


class MM(object):
    def __init__(self):
        self.n = 3  # number of joints (inputs)
        self.p = 2  # number of outputs (Cartesian DOFs)

    def forward(self, q):
        return np.array([q[0] + L1*np.cos(q[1]) + L2*np.cos(q[1]+q[2]),
                                L1*np.sin(q[1]) + L2*np.sin(q[1]+q[2])])

    def jacobian(self, q):
        return np.array([
            [1, -L1*np.sin(q[1])-L2*np.sin(q[1]+q[2]), -L2*np.sin(q[1]+q[2])],
            [0,  L1*np.cos(q[1])+L2*np.cos(q[1]+q[2]),  L2*np.cos(q[1]+q[2])]])


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

        n = self.model.n  # number of joints
        p = self.model.p  # number of Cartesian outputs

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

            fbar[k*p:(k+1)*p] = self.model.forward(q)
            Jbar[k*p:(k+1)*p, k*n:(k+1)*n] = self.model.jacobian(q)

            Qbar[k*p:(k+1)*p, k*p:(k+1)*p] = self.Q
            Rbar[k*n:(k+1)*n, k*n:(k+1)*n] = self.R

        dbar = fbar - pr

        H = Rbar + self.dt**2*Jbar.T.dot(Qbar).dot(Jbar)
        g = dq.T.dot(Rbar) + self.dt*dbar.T.dot(Qbar).dot(Jbar)

        # IPython.embed()

        return H, g

    def _iterate(self, q0, pr, dq, N):
        n = self.model.n

        # Create the QP, which we'll solve sequentially.
        # num vars, num constraints (note that constraints only refer to matrix
        # constraints rather than bounds)
        qp = qpoases.PySQProblem(n * N, 0)
        options = qpoases.PyOptions()
        options.printLevel = qpoases.PyPrintLevel.NONE
        qp.setOptions(options)

        # Initial opt problem.
        H, g = self._lookahead(q0, pr, dq, N)

        # TODO revisit damper formulation
        # TODO handle individual bounds for different joints
        lb = np.ones(n * N) * self.lb - dq
        ub = np.ones(n * N) * self.ub - dq
        ret = qp.init(H, g, None, lb, ub, None, None, NUM_WSR)
        delta = np.zeros(n * N)
        qp.getPrimalSolution(delta)
        dq = dq + delta

        # IPython.embed()

        # Remaining sequence is hotstarted from the first.
        for i in range(NUM_ITER):
            H, g = self._lookahead(q0, pr, dq, N)

            # we currently do not have a constraint matrix A
            lb = np.ones(n*N) * self.lb - dq
            ub = np.ones(n*N) * self.ub - dq
            qp.hotstart(H, g, None, lb, ub, None, None, NUM_WSR)
            qp.getPrimalSolution(delta)

            # TODO we could have a different step size here, since the
            # linearization is only locally valid
            dq = dq + delta

            # IPython.embed()

        # TODO this isn't actually that valuable since it's for the step not
        # the actual velocity
        obj = qp.getObjVal()

        return dq, obj

    def solve(self, q0, pr, N):
        ''' Solve the MPC problem at current state x0 given desired output
            trajectory Yd. '''
        # initialize optimal inputs
        dq = np.zeros(self.model.n * N)

        # iterate to final solution
        dq, obj = self._iterate(q0, pr, dq, N)

        # return first optimal input
        return dq[:self.model.n], obj


def main():
    N = 10
    dt = 0.1
    tf = 10.0
    num_steps = int(tf / dt)

    model = MM()

    Q = np.eye(model.p)
    R = np.eye(model.n) * 0.01
    lb = -1.0
    ub = 1.0

    Kp = 1.0
    Ki = 0.1
    Kd = 2.0

    mpc = MPC(model, dt, Q, R, lb, ub)

    ts = np.array([i * dt for i in xrange(num_steps)])
    ps = np.zeros((num_steps, model.p))
    qs = np.zeros((num_steps, model.n))
    dqs = np.zeros((num_steps, model.n))

    # reference trajectory
    dist = 2.0
    # xr = np.array([i*dist/num_steps for i in xrange(num_steps)])
    xr = np.ones(num_steps)
    yr = np.ones(num_steps)
    pr = np.zeros(num_steps * 2)
    pr[::2] = xr
    pr[1::2] = yr

    q0 = np.array([0, np.pi/4.0, -np.pi/2.0])
    q = q0
    ps[0, :] = model.forward(q)

    objs = np.zeros(num_steps)

    E = 0

    for i in xrange(num_steps-1):
        n = min(N, num_steps - i)
        dq, obj = mpc.solve(q, pr[i*model.p:(i+n)*model.p], n)

        # PID control
        # e = Yd[i*2] - x[0]
        # de = Yd[i*2+1] - x[1]
        # E += dt * e
        # u = Kp*e + Kd*de + Ki*E

        # bound joint velocities
        dq = bound_array(dq, lb, ub)

        q = q + dt * dq
        p = model.forward(q)

        objs[i] = obj
        dqs[i, :] = dq
        qs[i+1, :] = q
        ps[i+1, :] = p

    # plt.plot(ts, pr, label='$\\theta_d$', color='k', linestyle='--')
    plt.figure(1)
    plt.plot(ts, pr[::2],  label='$x_d$', color='b', linestyle='--')
    plt.plot(ts, pr[1::2], label='$y_d$', color='r', linestyle='--')
    plt.plot(ts, ps[:, 0], label='$x$', color='b')
    plt.plot(ts, ps[:, 1], label='$y$', color='r')
    plt.grid()
    plt.legend()
    plt.xlabel('Time (s)')
    plt.ylabel('Position')

    plt.figure(2)
    plt.plot(ts, dqs[:, 0], label='$\\dot{q}_x$')
    plt.plot(ts, dqs[:, 1], label='$\\dot{q}_1$')
    plt.plot(ts, dqs[:, 2], label='$\\dot{q}_2$')
    plt.grid()
    plt.legend()
    plt.xlabel('Time (s)')
    plt.ylabel('Velocity')

    plt.figure(3)
    plt.plot(ts, objs)
    plt.grid()
    plt.xlabel('Time (s)')
    plt.ylabel('Objective value')

    plt.show()


if __name__ == '__main__':
    main()
