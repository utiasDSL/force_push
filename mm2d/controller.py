import numpy as np
import qpoases
import IPython


# mpc parameters
NUM_HORIZON = 1  # number of time steps for prediction horizon
NUM_WSR = 100    # number of working set recalculations
NUM_ITER = 3     # number of linearizations/iterations


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

        ni = self.model.ni  # number of joints
        no = self.model.no  # number of Cartesian outputs

        qbar = np.zeros(ni*(N+1))
        qbar[:ni] = q0

        # Integrate joint positions from the last iteration
        for k in range(1, N+1):
            q_prev = qbar[(k-1)*ni:k*ni]
            dq_prev = dq[(k-1)*ni:k*ni]
            q = q_prev + self.dt * dq_prev

            qbar[k*ni:(k+1)*ni] = q

        fbar = np.zeros(no*N)         # Lifted forward kinematics
        Jbar = np.zeros((no*N, ni*N))  # Lifted Jacobian
        Qbar = np.zeros((no*N, no*N))
        Rbar = np.zeros((ni*N, ni*N))
        Ebar = np.tril(np.ones((ni*N, ni*N)))

        for k in range(N):
            q = qbar[(k+1)*ni:(k+2)*ni]

            fbar[k*no:(k+1)*no] = self.model.forward(q)
            Jbar[k*no:(k+1)*no, k*ni:(k+1)*ni] = self.model.jacobian(q)

            Qbar[k*no:(k+1)*no, k*no:(k+1)*no] = self.Q
            Rbar[k*ni:(k+1)*ni, k*ni:(k+1)*ni] = self.R

        dbar = fbar - pr

        H = Rbar + self.dt**2*Ebar.T.dot(Jbar.T).dot(Qbar).dot(Jbar).dot(Ebar)
        g = dq.T.dot(Rbar) + self.dt*dbar.T.dot(Qbar).dot(Jbar).dot(Ebar)

        return H, g

    def _iterate(self, q0, pr, dq, N):
        ni = self.model.ni

        # Create the QP, which we'll solve sequentially.
        # num vars, num constraints (note that constraints only refer to matrix
        # constraints rather than bounds)
        qp = qpoases.PySQProblem(ni * N, 0)
        options = qpoases.PyOptions()
        options.printLevel = qpoases.PyPrintLevel.NONE
        qp.setOptions(options)

        # Zeros, because we currently do not have a constraint matrix A.
        A = np.zeros((ni * N, ni * N))
        lbA = ubA = np.zeros(ni * N)

        # Initial opt problem.
        H, g = self._lookahead(q0, pr, dq, N)

        # TODO revisit velocity damper formulation
        # TODO handle individual bounds for different joints
        lb = np.ones(ni * N) * self.lb - dq
        ub = np.ones(ni * N) * self.ub - dq
        ret = qp.init(H, g, A, lb, ub, lbA, ubA, NUM_WSR)
        delta = np.zeros(ni * N)
        qp.getPrimalSolution(delta)
        dq = dq + delta

        # Remaining sequence is hotstarted from the first.
        for i in range(NUM_ITER):
            H, g = self._lookahead(q0, pr, dq, N)

            lb = np.ones(ni*N) * self.lb - dq
            ub = np.ones(ni*N) * self.ub - dq
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
        dq = np.zeros(self.model.ni * N)

        # iterate to final solution
        dq, obj = self._iterate(q0, pr, dq, N)

        # return first optimal input
        return dq[:self.model.ni], obj


class OptimizingForceController(object):
    ''' Optimizing controller. '''
    def __init__(self, model, dt, Q, R, lb, ub):
        self.model = model
        self.dt = dt
        self.Q = Q
        self.R = R
        self.lb = lb
        self.ub = ub

    def solve(self, q, pr, f, fd):
        ''' Solve the MPC problem at current state x0 given desired output
            trajectory Yd. '''
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


class BaselineController2(object):
    ''' Baseline optimizing controller.
        Solves:
            min  0.5*||Ju - v||^2 + 0.5*u'Wu
            s.t. lb <= u <= ub
        where
            v = K*(pd-p) + vd '''
    def __init__(self, model, W, K, lb, ub, verbose=False):
        self.model = model
        self.W = W
        self.K = K
        self.lb = lb
        self.ub = ub

        self.verbose = verbose

    def solve(self, q, dq, pd, vd, C=None):
        ''' Solve for the optimal inputs. '''
        ni = self.model.ni

        # forward kinematics
        p = self.model.forward(q)
        J = self.model.jacobian(q)

        # calculate velocity reference
        v = self.K.dot(pd - p) + vd

        # setup the QP
        H = J.T.dot(J) + self.W
        g = -J.T.dot(v) - self.W.dot(dq)

        # bounds on the computed input
        lb = np.ones(ni) * self.lb
        ub = np.ones(ni) * self.ub

        qp = qpoases.PyQProblemB(ni)
        if not self.verbose:
            options = qpoases.PyOptions()
            options.printLevel = qpoases.PyPrintLevel.NONE
            qp.setOptions(options)
        ret = qp.init(H, g, lb, ub, NUM_WSR)

        u = np.zeros(ni)
        qp.getPrimalSolution(u)
        return u
