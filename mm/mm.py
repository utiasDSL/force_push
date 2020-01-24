#!/usr/bin/env python2
from __future__ import print_function
import numpy as np
import matplotlib.pyplot as plt
import qpoases

from plotter import RobotPlotter
from obstacle import Wall, Circle
from trajectory import circle, spiral

import IPython


# robot parameters
L1 = 1
L2 = 1

DT = 0.1         # timestep (s)
DURATION = 10.0  # duration of trajectory (s)

# mpc parameters
NUM_HORIZON = 10  # number of time steps for prediction horizon
NUM_WSR = 100     # number of working set recalculations
NUM_ITER = 10     # number of linearizations/iterations

Q = np.diag([1.0, 1.0, 0.00001])
R = np.eye(3) * 0.01
LB = -1.0
UB = 1.0


# force control params
K_pf = 0
K_if = 0.01
K_df = 0.9  # discharge term

K_a = 100.0
K_v = 100.0

F_CONTACT = 0.0


def rms(e):
    ''' Calculate root mean square of a vector of data. '''
    return np.sqrt(np.mean(np.square(e)))


def bound_array(a, lb, ub):
    ''' Elementwise bound array above and below. '''
    return np.minimum(np.maximum(a, lb), ub)


def desired_force(f, fc):
    ''' Calculate desired force based on whether contact is detected.

        f:           Measured force
        f_contact:   Desired contact force
        f_threshold: Threshold for force that indicates contact has been made. '''
    f_norm = np.linalg.norm(f)
    if f_norm > fc:
        # project contact back onto f
        return fc * f / f_norm, True

    # otherwise we assume we're in freespace
    return f, False


class MM(object):
    def __init__(self):
        self.n = 3  # number of joints (inputs/DOFs)
        self.p = 3  # number of outputs (end effector coords)

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
        Ebar = np.tril(np.ones((n*N, n*N)))
        # Ebar = np.eye(n*N)

        for k in range(N):
            q = qbar[(k+1)*n:(k+2)*n]

            fbar[k*p:(k+1)*p] = self.model.forward(q)
            Jbar[k*p:(k+1)*p, k*n:(k+1)*n] = self.model.jacobian(q)

            Qbar[k*p:(k+1)*p, k*p:(k+1)*p] = self.Q
            Rbar[k*n:(k+1)*n, k*n:(k+1)*n] = self.R

        dbar = fbar - pr

        H = Rbar + self.dt**2*Ebar.T.dot(Jbar.T).dot(Qbar).dot(Jbar).dot(Ebar)
        g = dq.T.dot(Rbar) + self.dt*dbar.T.dot(Qbar).dot(Jbar).dot(Ebar)

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


def main():
    num_steps = int(DURATION / DT)

    model = MM()
    mpc = MPC(model, DT, Q, R, LB, UB)

    ts = np.array([i * DT for i in xrange(num_steps+1)])
    ps = np.zeros((num_steps+1, model.p))
    qs = np.zeros((num_steps+1, model.n))
    dqs = np.zeros((num_steps, model.n))
    dps = np.zeros((num_steps, model.p))
    fs = np.zeros((num_steps, 2))

    q0 = np.array([0, np.pi/4.0, -np.pi/4.0])
    p0 = model.forward(q0)

    # reference trajectory
    xr, yr = spiral(p0, ts[1:])
    # xr, yr, thetar = circle(p0, ts[1:])
    pr = np.zeros(num_steps * model.p)
    pr[0::model.p] = xr
    pr[1::model.p] = yr
    #pr[2::model.p] = thetar

    # obstacles
    obs = Wall(x=2.5)
    # obs = Circle(c=np.array([3.0, 1.5]), r=1)

    F = 0  # force control integral term

    q = q0
    p = p0
    qs[0, :] = q0
    ps[0, :] = p0

    plotter = RobotPlotter(L1, L2)
    plotter.start(q0, xr, yr, obs)

    thetar = 0

    flag = False

    pf = np.zeros(2)
    dpf = np.zeros(2)

    ddpf_max = 1.0
    dpf_max = 1.0

    # IPython.embed()

    for i in xrange(num_steps):
        # apply force
        f = obs.apply_force(p, dps[i-1, :])
        # f = np.zeros(2)

        n = min(NUM_HORIZON, num_steps - i)

        # alternative force control
        # remove desired orientation
        # ref = pr[i*model.p:(i+n)*model.p].reshape((n, model.p))
        # P = ref[:, :2]
        # theta = ref[:, 2]
        # f_norm = np.linalg.norm(f)
        #
        # if f_norm > F_CONTACT:
        #     fd = F_CONTACT
        # else:
        #     fd = f_norm
        #
        # if f_norm > 0:
        #     fn = f / f_norm
        #     f_err = fd - f_norm
        #     F += DT * f_err
        #     pf = K_pf * f_err + K_if * F
        #
        #     Pr = P - np.outer(P.dot(fn), fn) #+ pf * np.tile(fn, (n, 1))
        #     pd = np.concatenate((Pr, theta[:, None]), axis=1).flatten()
        # else:
        #     pd = ref.flatten()

        # force control
        fd, contact = desired_force(f, fc=F_CONTACT)
        f_err = fd - f
        F += DT * f_err
        pf = K_pf * f_err + K_if * F
        F = K_df * F  # discharge term, so we can return to original trajectory

        ddpf = (f_err - K_v*dpf) / K_a
        # if np.linalg.norm(ddpf) > ddpf_max:
        #     ddpf = ddpf_max * ddpf / np.linalg.norm(ddpf)

        # pf = pf + DT * dpf

        dpf = dpf + DT * ddpf
        # if np.linalg.norm(dpf) > dpf_max:
        #     dpf = dpf_max * dpf / np.linalg.norm(dpf)

        if contact and not flag:
            flag = True

        if flag and np.linalg.norm(f) > 0:
            d = f / np.linalg.norm(f)  # unit vector in direction of force
            a = np.arctan2(d[1], d[0])  # angle in direction of force
            thetar = a
        pf3 = np.array([pf[0], pf[1], thetar])

        # only do discharge when not in contact
        # if not contact:
        dpf = K_df * dpf
        pf = K_df * pf

        pf3 = np.zeros(3)

        # MPC
        pd = pr[i*model.p:(i+n)*model.p] + np.tile(pf3, n)
        dq, _ = mpc.solve(q, pd, n)

        # bound joint velocities (i.e. enforce actuation constraints)
        dq = bound_array(dq, LB, UB)

        # record end effector velocity
        dps[i, :] = model.jacobian(q).dot(dq)

        # integrate
        q = q + DT * dq
        p = model.forward(q)

        fs[i, :] = f
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

    plt.figure()
    plt.plot(ts[:-1], fs[:, 0], label='$f_x$')
    plt.plot(ts[:-1], fs[:, 1], label='$f_y$')
    plt.plot(ts[:-1], np.sqrt(np.sum(fs**2, axis=1)), label='$f$')
    plt.grid()
    plt.legend()
    plt.title('Force')
    plt.xlabel('Time (s)')
    plt.ylabel('Force')

    # plt.figure(3)
    # plt.plot(ts, objs)
    # plt.grid()
    # plt.xlabel('Time (s)')
    # plt.ylabel('Objective value')

    plt.show()


if __name__ == '__main__':
    main()