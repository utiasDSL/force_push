import numpy as np
from scipy import sparse
import qpoases
import IPython


# mpc parameters
NUM_WSR = 100    # number of working set recalculations
NUM_ITER = 3     # number of linearizations/iterations


class ObstacleAvoidingMPC(object):
    ''' Model predictive controller. '''
    def __init__(self, model, dt, Q, R, vel_lim, acc_lim):
        self.model = model
        self.dt = dt
        self.Q = Q
        self.R = R
        self.vel_lim = vel_lim
        self.acc_lim = acc_lim

    def _lookahead(self, q0, pr, u, N, pc):
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
        qbar[ni:] = qbar[ni:] + self.dt * Ebar.dot(u)

        Abar = np.zeros((N, ni*N))
        lbA = np.zeros(N)

        for k in range(N):
            q = qbar[(k+1)*ni:(k+2)*ni]
            p = self.model.forward(q)
            J = self.model.jacobian(q)
            d = np.linalg.norm(p - pc)

            fbar[k*no:(k+1)*no] = p
            Jbar[k*no:(k+1)*no, k*ni:(k+1)*ni] = J

            Abar[k, k*ni:(k+1)*ni] = (p - pc).T.dot(J) / np.linalg.norm(p - pc)
            lbA[k] = 0.5 - d

            Qbar[k*no:(k+1)*no, k*no:(k+1)*no] = self.Q
            Rbar[k*ni:(k+1)*ni, k*ni:(k+1)*ni] = self.R

        dbar = fbar - pr

        H = Rbar + self.dt**2*Ebar.T.dot(Jbar.T).dot(Qbar).dot(Jbar).dot(Ebar)
        g = u.T.dot(Rbar) + self.dt*dbar.T.dot(Qbar).dot(Jbar).dot(Ebar)
        A = self.dt*Abar.dot(Ebar)

        return H, g, A, lbA

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

    def _iterate(self, q0, dq0, pr, u, N, pc):
        ni = self.model.ni

        # Create the QP, which we'll solve sequentially.
        # num vars, num constraints (note that constraints only refer to matrix
        # constraints rather than bounds)
        # num constraints = N obstacle constraints and ni*N joint acceleration
        # constraints
        num_var = ni * N
        num_constraints = N + ni * N
        qp = qpoases.PySQProblem(num_var, num_constraints)
        options = qpoases.PyOptions()
        options.printLevel = qpoases.PyPrintLevel.NONE
        qp.setOptions(options)

        # Initial opt problem.
        H, g, A_obs, lbA_obs = self._lookahead(q0, pr, u, N, pc)
        ubA_obs = np.infty * np.ones_like(lbA_obs)

        lb, ub = self._calc_vel_limits(u, ni, N)
        A_acc, lbA_acc, ubA_acc = self._calc_acc_limits(u, dq0, ni, N)

        A = np.vstack((A_obs, A_acc))
        lbA = np.concatenate((lbA_obs, lbA_acc))
        ubA = np.concatenate((ubA_obs, ubA_acc))

        ret = qp.init(H, g, A, lb, ub, lbA, ubA, np.array([NUM_WSR]))
        delta = np.zeros(ni * N)
        qp.getPrimalSolution(delta)
        u = u + delta

        # Remaining sequence is hotstarted from the first.
        for i in range(NUM_ITER - 1):
            H, g, A_obs, lbA_obs = self._lookahead(q0, pr, u, N, pc)
            lb, ub = self._calc_vel_limits(u, ni, N)
            A_acc, lbA_acc, ubA_acc = self._calc_acc_limits(u, dq0, ni, N)
            A = np.vstack((A_obs, A_acc))
            lbA = np.concatenate((lbA_obs, lbA_acc))
            ubA = np.concatenate((ubA_obs, ubA_acc))

            qp.hotstart(H, g, A, lb, ub, lbA, ubA, np.array([NUM_WSR]))
            qp.getPrimalSolution(delta)

            u = u + delta

        return u

    def solve(self, q0, dq0, pr, N, pc):
        ''' Solve the MPC problem at current state x0 given desired output
            trajectory Yd. '''
        # initialize optimal inputs
        u = np.zeros(self.model.ni * N)

        # iterate to final solution
        u = self._iterate(q0, dq0, pr, u, N, pc)

        # return first optimal input
        return u[:self.model.ni]


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
            future. '''
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
        qbar[ni:] = qbar[ni:] + self.dt * Ebar.dot(u)

        for k in range(N):
            q = qbar[(k+1)*ni:(k+2)*ni]
            p = self.model.forward(q)
            J = self.model.jacobian(q)

            fbar[k*no:(k+1)*no] = p
            Jbar[k*no:(k+1)*no, k*ni:(k+1)*ni] = J

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
        # num constraints = ni*N joint acceleration constraints
        num_var = ni * N
        num_constraints = ni * N
        qp = qpoases.PySQProblem(num_var, num_constraints)
        options = qpoases.PyOptions()
        options.printLevel = qpoases.PyPrintLevel.NONE
        qp.setOptions(options)

        # Initial opt problem.
        H, g = self._lookahead(q0, pr, u, N)
        lb, ub = self._calc_vel_limits(u, ni, N)
        A, lbA, ubA = self._calc_acc_limits(u, dq0, ni, N)

        ret = qp.init(H, g, A, lb, ub, lbA, ubA, np.array([NUM_WSR]))
        delta = np.zeros(ni * N)
        qp.getPrimalSolution(delta)
        u = u + delta

        # Remaining sequence is hotstarted from the first.
        for i in range(NUM_ITER - 1):
            H, g = self._lookahead(q0, pr, u, N)
            lb, ub = self._calc_vel_limits(u, ni, N)
            A, lbA, ubA = self._calc_acc_limits(u, dq0, ni, N)

            qp.hotstart(H, g, A, lb, ub, lbA, ubA, np.array([NUM_WSR]))
            qp.getPrimalSolution(delta)

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
