import numpy as np
import qpoases


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
