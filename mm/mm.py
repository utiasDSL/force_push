#!/usr/bin/env python2
from __future__ import print_function
import numpy as np
import matplotlib.pyplot as plt
import qpoases

from plotter import RobotPlotter

import IPython


# robot parameters
L1 = 1
L2 = 1

DT = 0.1  # timestep
DURATION = 25.0  # seconds
NUM_HORIZON = 10  # number of time steps for prediction horizon
NUM_WSR = 100  # number of working set recalculations
NUM_ITER = 10  # number of linearizations/iterations


def rms(e):
    ''' Calculate root mean square of a vector of data. '''
    return np.sqrt(np.mean(np.square(e)))


def bound_array(a, lb, ub):
    ''' Elementwise bound array above and below. '''
    return np.minimum(np.maximum(a, lb), ub)


class Wall(object):
    def __init__(self, k, x):
        self.k = k  # spring constant
        self.x = x  # location

    def apply_force(self, p):
        ''' Apply force based on end effector position p. '''
        if p[0] > self.x:
            dx = p[0] - self.x
            f = np.array([self.k * dx, 0])
        else:
            f = np.zeros(2)
        return f


class Circle(object):
    def __init__(self, k, c, r):
        self.k = k
        self.c = c
        self.r = r

    def apply_force(self, p):
        a = p[:2] - self.c
        d = np.linalg.norm(a)
        b = self.r * a / d
        if d < self.r:
            dx = a - b
            f = self.k * dx
        else:
            f = np.zeros(2)
        return f

    def draw(self, ax):
        ax.add_patch(plt.Circle(self.c, self.r, color='k', fill=False))


# TODO but no, we want to do orthogonal to the plane
def desired_force1(f, f_contact, f_threshold):
    fd = np.zeros(2)
    if f[0] >= f_threshold[0]:
        fd[0] = f_contact
    if f[1] >= f_threshold[1]:
        fd[1] = f_contact
    return fd


def desired_force2(f, f_contact, f_threshold):
    ''' Calculate desired force based on whether contact is detected.

        f:            Measured force
        f_contact:   Desired contact force
        f_threshold: Threshold for force that indicates contact has been made. '''
    f_norm = np.linalg.norm(f)
    if f_norm > f_contact:
        # project contact back onto f
        return f_contact * f / f_norm, True

    # otherwise we assume we're in freespace
    return np.zeros(2), False


class MM(object):
    def __init__(self):
        self.n = 3  # number of joints (inputs/DOFs)
        self.p = 2  # number of outputs (end effector coords)

    def forward(self, q):
        f = np.array([q[0] + L1*np.cos(q[1]) + L2*np.cos(q[1]+q[2]),
                      L1*np.sin(q[1]) + L2*np.sin(q[1]+q[2]),
                      q[1] + q[2]])
        return f[:self.p]

    def jacobian(self, q):
        J = np.array([
            [1, -L1*np.sin(q[1])-L2*np.sin(q[1]+q[2]), -L2*np.sin(q[1]+q[2])],
            [0,  L1*np.cos(q[1])+L2*np.cos(q[1]+q[2]),  L2*np.cos(q[1]+q[2])],
            [0, 1, 1]])
        return J[:self.p, :]


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


# Trajectories

def spiral(p0, ts):
    a = 0.1
    b = 0.08
    x = p0[0] + (a + b*ts) * np.cos(ts)
    y = p0[1] + (a + b*ts) * np.sin(ts)
    return x, y


def point(p0, ts):
    x = p0[0] * np.ones(ts.shape[0])
    y = p0[1] * np.ones(ts.shape[0])
    return x, y


def line(p0, ts):
    v = 0.125
    x = p0[0] + np.array([v*t for t in ts])
    y = p0[1] * np.ones(ts.shape[0])
    theta = np.ones(ts.shape[0]) * np.pi * 0
    return x, y, theta


def main():
    num_steps = int(DURATION / DT)

    model = MM()

    Q = np.eye(model.p)
    R = np.eye(model.n) * 0.01
    lb = -10.0
    ub = 10.0

    mpc = MPC(model, DT, Q, R, lb, ub)

    ts = np.array([i * DT for i in xrange(num_steps+1)])
    ps = np.zeros((num_steps+1, model.p))
    qs = np.zeros((num_steps+1, model.n))
    dqs = np.zeros((num_steps, model.n))
    objs = np.zeros(num_steps)

    q0 = np.array([0, np.pi/4.0, -np.pi/4.0])
    p0 = model.forward(q0)

    # reference trajectory
    # xr, yr = spiral(p0, ts[1:])
    xr, yr, thetar = line(p0, ts[1:])
    pr = np.zeros(num_steps * model.p)
    pr[0::model.p] = xr
    pr[1::model.p] = yr
    # pr[2::model.p] = thetar

    # obstacles
    # obs = Wall(k=1000, x=3.0)
    obs = Circle(k=10000, c=np.array([3.5, 1.5]), r=1)

    # force control
    K_pf = 0.0
    K_if = 0.001
    F = 0

    q = q0
    p = p0
    qs[0, :] = q0
    ps[0, :] = p0

    plotter = RobotPlotter(L1, L2)
    plotter.start(q0, xr, yr, obs)

    thetar = 0

    for i in xrange(num_steps):
        f = obs.apply_force(p)

        fd, contact = desired_force2(f, f_contact=0.1, f_threshold=0.1)
        f_err = fd - f
        F += DT * f_err  # integrate
        pf = K_pf * f_err + K_if * F  # control law
        F = 0.9 * F  # discharge term, so we can return to original trajectory
        print('f = {}'.format(f))

        # if contact:
        #     d = f / np.linalg.norm(f)  # unit vector in direction of force
        #     a = np.arctan2(d[1], d[0])  # angle in direction of force
        #     thetar = a
        # pf = np.array([pf[0], pf[1], thetar])

        n = min(NUM_HORIZON, num_steps - i)
        pd = pr[i*model.p:(i+n)*model.p] + np.tile(pf, n)

        dq, obj = mpc.solve(q, pd, n)

        # bound joint velocities (i.e. enforce actuation constraints)
        dq = bound_array(dq, lb, ub)

        q = q + DT * dq
        p = model.forward(q)

        objs[i] = obj
        dqs[i, :] = dq
        qs[i+1, :] = q
        ps[i+1, :] = p

        plotter.update(q)

    plt.ioff()

    xe = pr[0::model.p] - ps[1:, 0]
    ye = pr[1::model.p] - ps[1:, 1]
    print('RMSE(x) = {}'.format(rms(xe)))
    print('RMSE(y) = {}'.format(rms(ye)))

    # plt.plot(ts, pr, label='$\\theta_d$', color='k', linestyle='--')
    plt.figure()
    plt.plot(ts[1:], pr[0::model.p], label='$x_d$', color='b', linestyle='--')
    plt.plot(ts[1:], pr[1::model.p], label='$y_d$', color='r', linestyle='--')
    plt.plot(ts, ps[:, 0], label='$x$', color='b')
    plt.plot(ts, ps[:, 1], label='$y$', color='r')
    plt.grid()
    plt.legend()
    plt.xlabel('Time (s)')
    plt.ylabel('Position')
    plt.title('End effector position')

    plt.figure()
    plt.plot(ts[:-1], dqs[:, 0], label='$\\dot{q}_x$')
    plt.plot(ts[:-1], dqs[:, 1], label='$\\dot{q}_1$')
    plt.plot(ts[:-1], dqs[:, 2], label='$\\dot{q}_2$')
    plt.grid()
    plt.legend()
    plt.title('Commanded joint velocity')
    plt.xlabel('Time (s)')
    plt.ylabel('Velocity')

    # plt.figure(3)
    # plt.plot(ts, objs)
    # plt.grid()
    # plt.xlabel('Time (s)')
    # plt.ylabel('Objective value')

    plt.show()


if __name__ == '__main__':
    main()
