import numpy as np
from scipy import sparse
import qpoases
import IPython


# mpc parameters
NUM_HORIZON = 1  # number of time steps for prediction horizon
NUM_WSR = 100    # number of working set recalculations
NUM_ITER = 3     # number of linearizations/iterations


class MPC(object):
    ''' Model predictive controller. '''
    def __init__(self, model, dt, Q, R, vel_lim, acc_lim):
        self.model = model
        self.dt = dt
        self.Q = Q
        self.R = R
        self.vel_lim = vel_lim
        self.acc_lim = acc_lim

    def _lookahead(self, q0, pr, u, N):
        ''' Generate lifted matrices proprogating the state N timesteps into the
            future.

            sys: System model.
            q0:  Current joint positions.
            pr:  Desired Cartesian output trajectory.
            u:   Input joint velocties from the last iteration.
            Q:   Tracking error weighting matrix.
            R:   Input magnitude weighting matrix.
            N:   Number of timesteps into the future. '''

        ni = self.model.ni  # number of joints
        no = self.model.no  # number of Cartesian outputs

        fbar = np.zeros(no*N)         # Lifted forward kinematics
        Jbar = np.zeros((no*N, ni*N))  # Lifted Jacobian
        Qbar = np.zeros((no*N, no*N))
        Rbar = np.zeros((ni*N, ni*N))

        # lower triangular matrix of ni*ni identity matrices
        Ebar = np.kron(np.tril(np.ones((N, N))), np.eye(ni))

        # Integrate joint positions from the last iteration
        qbar = np.tile(q0, N+1)
        qbar[ni:] += self.dt * Ebar.dot(u)

        for k in range(N):
            q = qbar[(k+1)*ni:(k+2)*ni]

            fbar[k*no:(k+1)*no] = self.model.forward(q)
            Jbar[k*no:(k+1)*no, k*ni:(k+1)*ni] = self.model.jacobian(q)

            Qbar[k*no:(k+1)*no, k*no:(k+1)*no] = self.Q
            Rbar[k*ni:(k+1)*ni, k*ni:(k+1)*ni] = self.R

        dbar = fbar - pr

        H = Rbar + self.dt**2*Ebar.T.dot(Jbar.T).dot(Qbar).dot(Jbar).dot(Ebar)
        g = u.T.dot(Rbar) + self.dt*dbar.T.dot(Qbar).dot(Jbar).dot(Ebar)

        return H, g

    def _calc_vel_limits(self, u, ni, N):
        L = np.ones(ni * N) * self.vel_lim
        lb = -L - u
        ub = L - u
        return lb, ub

    def _calc_acc_limits(self, u, dq0, ni, N):
        # u_prev consists of [dq0, u_0, u_1, ..., u_{N-2}]
        # u is [u_0, ..., u_{N-1}]
        u_prev = np.zeros(ni * N)
        u_prev[:ni] = dq0
        u_prev[ni:] = u[:-ni]

        L = self.dt * np.ones(ni * N) * self.acc_lim
        lbA = -L - u + u_prev
        ubA = L - u + u_prev

        d1 = np.ones(N)
        d2 = -np.ones(N - 1)

        # A0 is NxN
        A0 = sparse.diags((d1, d2), [0, -1]).toarray()

        # kron to make it work for n-dimensional inputs
        A = np.kron(A0, np.eye(ni))

        return A, lbA, ubA

    def _iterate(self, q0, dq0, pr, u, N):
        ni = self.model.ni

        # Create the QP, which we'll solve sequentially.
        # num vars, num constraints (note that constraints only refer to matrix
        # constraints rather than bounds)
        num_var = ni * N
        num_constraints = ni * N
        qp = qpoases.PySQProblem(num_var, num_constraints)
        options = qpoases.PyOptions()
        options.printLevel = qpoases.PyPrintLevel.NONE
        qp.setOptions(options)

        # Initial opt problem.
        H, g = self._lookahead(q0, pr, u, N)

        # TODO revisit velocity damper formulation
        lb, ub = self._calc_vel_limits(u, ni, N)
        A, lbA, ubA = self._calc_acc_limits(u, dq0, ni, N)
        # A = np.zeros((ni * N, ni * N))
        # lbA = ubA = np.zeros(ni * N)

        ret = qp.init(H, g, A, lb, ub, lbA, ubA, NUM_WSR)
        delta = np.zeros(ni * N)
        qp.getPrimalSolution(delta)
        u = u + delta

        # Remaining sequence is hotstarted from the first.
        for i in range(NUM_ITER):
            H, g = self._lookahead(q0, pr, u, N)
            lb, ub = self._calc_vel_limits(u, ni, N)
            A, lbA, ubA = self._calc_acc_limits(u, dq0, ni, N)

            qp.hotstart(H, g, A, lb, ub, lbA, ubA, NUM_WSR)
            qp.getPrimalSolution(delta)

            # TODO we could have a different step size here, since the
            # linearization is only locally valid
            u = u + delta

        return u

    def solve(self, q0, dq0, pr, N):
        ''' Solve the MPC problem at current state x0 given desired output
            trajectory Yd. '''
        # initialize optimal inputs
        u = np.zeros(self.model.ni * N)

        # iterate to final solution
        u = self._iterate(q0, dq0, pr, u, N)

        # return first optimal input
        return u[:self.model.ni]


class OptimizingForceController(object):
    def __init__(self, model, dt, Q, R, lb, ub):
        self.model = model
        self.dt = dt
        self.Q = Q
        self.R = R
        self.lb = lb
        self.ub = ub

    def solve(self, q, pr, f, fd):
        n = self.model.n

        pe = self.model.forward(q)
        d = pr - pe
        J = self.model.jacobian(q)

        H = self.dt**2 * J.T.dot(self.Q).dot(J) + self.R
        g = -self.dt * d.T.dot(self.Q).dot(J)

        lb = np.ones(n) * self.lb
        ub = np.ones(n) * self.ub

        # force control
        Kp = 0.01
        f_norm = np.linalg.norm(f)

        # TODO we can actually set nf in the case we have f=0 but fd != 0
        if f_norm > 0:
            nf = f / f_norm  # unit vector in direction of force
            df = fd - f_norm

            # only the first two rows of J are used since we only want to
            # constrain position
            A = nf.T.dot(J[:2,:])
            lbA = ubA = np.array([Kp * df])
            # lbA = np.array([0.0]) if df > 0 else None
            # ubA = np.array([0.0]) if df < 0 else None
            # A = A.reshape((1, 3))

            # H += np.outer(A, A)
            # g -= Kp * df * A


            # # admittance control
            Qf = np.eye(2)
            Bf = 0 * np.eye(2)
            Kf = 0.001 * np.eye(2)

            Jp = J[:2, :]

            K = -(self.dt*Kf + Bf).dot(Jp)
            d = K.dot(pr-pe) - f

            H += K.T.dot(Qf).dot(K)
            g += d.T.dot(Qf).dot(K)

            # Create the QP, which we'll solve sequentially.
            # num vars, num constraints (note that constraints only refer to
            # matrix constraints rather than bounds)

            # qp = qpoases.PyQProblem(n, 1)
            # options = qpoases.PyOptions()
            # options.printLevel = qpoases.PyPrintLevel.NONE
            # qp.setOptions(options)
            # ret = qp.init(H, g, A, lb, ub, lbA, ubA, NUM_WSR)

            qp = qpoases.PyQProblemB(n)
            options = qpoases.PyOptions()
            options.printLevel = qpoases.PyPrintLevel.NONE
            qp.setOptions(options)
            ret = qp.init(H, g, lb, ub, NUM_WSR)
        else:
            qp = qpoases.PyQProblemB(n)
            options = qpoases.PyOptions()
            options.printLevel = qpoases.PyPrintLevel.NONE
            qp.setOptions(options)
            ret = qp.init(H, g, lb, ub, NUM_WSR)

        dq = np.zeros(n)
        qp.getPrimalSolution(dq)
        return dq


class BaselineController(object):
    ''' Baseline optimizing controller.
        Solves:
            min  0.5*u'Wu
            s.t. Ju = v
                 lb <= u <= ub
        where
            v = K*(pd-p) + vd '''
    def __init__(self, model, W, K, lb, ub, verbose=False):
        self.model = model
        self.W = W
        self.K = K
        self.lb = lb
        self.ub = ub

        self.verbose = verbose

    def solve(self, q, pd, vd, C=None):
        ''' Solve for the optimal inputs. '''
        ni = self.model.ni
        no = self.model.no

        # forward kinematics
        p = self.model.forward(q)
        J = self.model.jacobian(q)

        # calculate velocity reference
        v = self.K.dot(pd - p) + vd

        # setup the QP
        H = self.W
        g = np.zeros(ni)

        # optionally add additional constraints to decouple the system
        if C is not None:
            A = np.vstack((J, C))
            lbA = ubA = np.concatenate((v, np.zeros(C.shape[0])))
            nc = no + C.shape[0]
        else:
            A = J
            lbA = ubA = v
            nc = no

        # bounds on the computed input
        lb = np.ones(ni) * self.lb
        ub = np.ones(ni) * self.ub

        qp = qpoases.PyQProblem(ni, nc)
        if not self.verbose:
            options = qpoases.PyOptions()
            options.printLevel = qpoases.PyPrintLevel.NONE
            qp.setOptions(options)
        ret = qp.init(H, g, A, lb, ub, lbA, ubA, NUM_WSR)

        dq = np.zeros(ni)
        qp.getPrimalSolution(dq)
        return dq


class DiffIKController(object):
    ''' Basic differential IK controller.
        Solves:
            min  0.5*||Ju - v||^2 + 0.5*u'Wu
            s.t. lb <= u <= ub
        where
            v = K*(pd-p) + vd '''
    def __init__(self, model, W, K, dt, vel_lim, acc_lim, verbose=False):
        self.model = model
        self.W = W
        self.K = K
        self.dt = dt
        self.vel_lim = vel_lim
        self.acc_lim = acc_lim
        self.verbose = verbose

    def solve(self, q, dq, pd, vd):
        ''' Solve for the optimal inputs. '''
        ni = self.model.ni

        # forward kinematics
        p = self.model.forward(q)
        J = self.model.jacobian(q)

        # calculate velocity reference
        v = self.K.dot(pd - p) + vd

        # setup the QP
        H = J.T.dot(J) + self.W
        g = -J.T.dot(v)

        # bounds on the computed input
        vel_ub = np.ones(ni) * self.vel_lim
        vel_lb = -vel_ub

        acc_ub = np.ones(ni) * self.acc_lim * self.dt + dq
        acc_lb = -np.ones(ni) * self.acc_lim * self.dt + dq

        ub = np.maximum(vel_ub, acc_ub)
        lb = np.minimum(vel_lb, acc_lb)

        qp = qpoases.PyQProblemB(ni)
        if not self.verbose:
            options = qpoases.PyOptions()
            options.printLevel = qpoases.PyPrintLevel.NONE
            qp.setOptions(options)
        ret = qp.init(H, g, lb, ub, NUM_WSR)

        u = np.zeros(ni)
        qp.getPrimalSolution(u)
        return u


# TODO for some reason this has trouble tracking an EE reference exactly
class AccelerationController(object):
    def __init__(self, model, W, Kp, Kv, dt, vel_lim, acc_lim, verbose=False):
        self.model = model
        self.W = W
        self.dt = dt
        self.vel_lim = vel_lim
        self.acc_lim = acc_lim
        self.verbose = verbose

        self.Kp = Kp
        self.Kv = Kv

    def solve(self, q, dq, pd, vd, ad):
        ''' Solve for the optimal inputs. '''
        ni = self.model.ni

        # forward kinematics
        p = self.model.forward(q)
        J = self.model.jacobian(q)
        Jdot = self.model.dJdt(q, dq)

        # calculate acceleration reference
        v = J.dot(dq)
        a_ref = self.Kp.dot(pd - p) + self.Kv.dot(vd - v) + ad

        # setup the QP
        d = Jdot.dot(dq) - a_ref
        H = J.T.dot(J) + self.W
        g = J.T.dot(d)

        # bounds on the computed input
        acc_ub = np.ones(ni) * self.acc_lim
        acc_lb = -acc_ub

        ub = acc_ub
        lb = acc_lb

        qp = qpoases.PyQProblemB(ni)
        if not self.verbose:
            options = qpoases.PyOptions()
            options.printLevel = qpoases.PyPrintLevel.NONE
            qp.setOptions(options)
        ret = qp.init(H, g, lb, ub, NUM_WSR)

        u = np.zeros(ni)
        qp.getPrimalSolution(u)
        return u
