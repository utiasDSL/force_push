import numpy as np
from scipy import sparse
from mm2d import util
import qpoases
import IPython


# mpc parameters
NUM_WSR = 100    # number of working set recalculations
NUM_ITER = 3     # number of linearizations/iterations


# TODO experimental MPC controller that uses the SQP controller under the hood
# - is there is a significant penalty or wrapping things up as Python functions
#   rather than directly as arrays?
# - ideally, we'd make a library of objectives, bounds, and constraints that
#   could be put together for different behaviours
class TrackingMPC:
    def __init__(self, model, dt, Q, R, num_horizon):
        self.model = model
        self.dt = dt
        self.Q = Q
        self.R = R
        self.num_horizon = num_horizon

        ni = self.model.ni
        nv = num_horizon * ni

        # setup SQP values
        bounds = sqp.Bounds(-model.vel_lim*np.ones(nv), model.vel_lim*np.ones(nv))

        def obj_val(x0, xd, var):
            q = x0
            J = 0
            for k in range(num_horizon):
                u = var[k*ni:(k+1)*ni] # TODO would be nicer if var was 2D
                q = q + dt * u
                p = model.forward(q)
                J += 0.5 * (p @ Q @ p + u @ R @ u)
            return J



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
        Qbar = np.kron(np.eye(N), self.Q)
        Rbar = np.kron(np.eye(N), self.R)

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


class ObstacleAvoidingMPC(object):
    ''' Model predictive controller with obstacle avoidance. '''
    def __init__(self, model, dt, Q, R, vel_lim, acc_lim):
        self.model = model
        self.dt = dt
        self.Q = Q
        self.R = R
        self.vel_lim = vel_lim
        self.acc_lim = acc_lim

    def _lookahead(self, q0, pr, u, N, pc):
        ''' Generate lifted matrices proprogating the state N timesteps into the
            future. '''
        ni = self.model.ni  # number of joints
        no = self.model.no  # number of Cartesian outputs

        fbar = np.zeros(no*N)         # Lifted forward kinematics
        Jbar = np.zeros((no*N, ni*N))  # Lifted Jacobian
        Qbar = np.kron(np.eye(N), self.Q)
        Rbar = np.kron(np.eye(N), self.R)

        # lower triangular matrix of ni*ni identity matrices
        Ebar = np.kron(np.tril(np.ones((N, N))), np.eye(ni))

        # Integrate joint positions from the last iteration
        qbar = np.tile(q0, N+1)
        qbar[ni:] = qbar[ni:] + self.dt * Ebar.dot(u)

        num_body_pts = 2
        Abar = np.zeros((N*num_body_pts, ni*N))
        lbA = np.zeros(N*num_body_pts)

        for k in range(N):
            q = qbar[(k+1)*ni:(k+2)*ni]
            p = self.model.forward(q)
            J = self.model.jacobian(q)

            fbar[k*no:(k+1)*no] = p
            Jbar[k*no:(k+1)*no, k*ni:(k+1)*ni] = J

            # TODO hardcoded radius
            # EE and obstacle
            d_ee_obs = np.linalg.norm(p - pc) - 0.5
            Abar[k*num_body_pts, k*ni:(k+1)*ni] = (p - pc).T.dot(J) / np.linalg.norm(p - pc)
            lbA[k*num_body_pts] = -d_ee_obs

            # base and obstacle
            pb = q[:2]
            Jb = np.array([[1, 0, 0, 0, 0],
                           [0, 1, 0, 0, 0]])
            d_base_obs = np.linalg.norm(pb - pc) - 0.5 - 0.56
            Abar[k*num_body_pts+1, k*ni:(k+1)*ni] = (pb - pc).T.dot(Jb) / np.linalg.norm(pb - pc)
            lbA[k*num_body_pts+1] = -d_base_obs

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
        num_constraints = 2*N + ni * N
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


class ObstacleAvoidingMPC2(object):
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
            future. '''
        ni = self.model.ni  # number of joints
        no = self.model.no  # number of Cartesian outputs

        fbar = np.zeros(no*N)         # Lifted forward kinematics
        Jbar = np.zeros((no*N, ni*N))  # Lifted Jacobian
        Qbar = np.kron(np.eye(N), self.Q)
        Rbar = np.kron(np.eye(N), self.R)

        # lower triangular matrix of ni*ni identity matrices
        Ebar = np.kron(np.tril(np.ones((N, N))), np.eye(ni))

        # Integrate joint positions from the last iteration
        qbar = np.tile(q0, N+1)
        qbar[ni:] = qbar[ni:] + self.dt * Ebar.dot(u)

        num_body_pts = 2+1
        Abar = np.zeros((N*num_body_pts, ni*N))
        lbA = np.zeros(N*num_body_pts)

        for k in range(N):
            q = qbar[(k+1)*ni:(k+2)*ni]
            p = self.model.forward(q)
            J = self.model.jacobian(q)

            pm = self.model.forward_m(q)
            Jm = self.model.jacobian_m(q)

            fbar[k*no:(k+1)*no] = pm
            Jbar[k*no:(k+1)*no, k*ni:(k+1)*ni] = Jm

            # TODO hardcoded radius
            # EE and obstacle
            d_ee_obs = np.linalg.norm(p - pc) - 0.5
            Abar[k*num_body_pts, k*ni:(k+1)*ni] = (p - pc).T.dot(J) / np.linalg.norm(p - pc)
            lbA[k*num_body_pts] = -d_ee_obs

            # base and obstacle
            pb = q[:2]
            Jb = np.array([[1, 0, 0, 0, 0],
                           [0, 1, 0, 0, 0]])
            d_base_obs = np.linalg.norm(pb - pc) - 0.5 - 0.56
            Abar[k*num_body_pts+1, k*ni:(k+1)*ni] = (pb - pc).T.dot(Jb) / np.linalg.norm(pb - pc)
            lbA[k*num_body_pts+1] = -d_base_obs

            # pf and ee: these need to stay close together
            pf = self.model.forward_f(q)
            Jf = self.model.jacobian_f(q)
            d_pf_ee = np.linalg.norm(p - pf)
            A_pf_ee = -(pf - p).T.dot(Jf - J) / d_pf_ee
            lbA_pf_ee = d_pf_ee - 0.75
            Abar[k*num_body_pts+2, k*ni:(k+1)*ni] = A_pf_ee
            lbA[k*num_body_pts+2] = lbA_pf_ee

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
        num_constraints = 3*N + ni * N
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


# class MPC2(object):
#     ''' Model predictive controller. '''
#     def __init__(self, model, dt, Q, R, vel_lim, acc_lim):
#         self.model = model
#         self.dt = dt
#         self.Q = Q
#         self.R = R
#         self.vel_lim = vel_lim
#         self.acc_lim = acc_lim
#
#     def _lookahead(self, q0, pr, u, N):
#         ''' Generate lifted matrices proprogating the state N timesteps into the
#             future. '''
#         ni = self.model.ni  # number of joints
#         no = self.model.no  # number of Cartesian outputs
#
#         fbar = np.zeros(no*N)         # Lifted forward kinematics
#         Jbar = np.zeros((no*N, ni*N))  # Lifted Jacobian
        # Qbar = np.kron(np.eye(N), self.Q)
        # Rbar = np.kron(np.eye(N), self.R)
#
#         # lower triangular matrix of ni*ni identity matrices
#         Ebar = np.kron(np.tril(np.ones((N, N))), np.eye(ni))
#
#         # Integrate joint positions from the last iteration
#         qbar = np.tile(q0, N+1)
#         qbar[ni:] = qbar[ni:] + self.dt * Ebar.dot(u)
#
#         num_body_pts = 1
#         Abar = np.zeros((N*num_body_pts, ni*N))
#         lbA = np.zeros(N*num_body_pts)
#
#         for k in range(N):
#             q = qbar[(k+1)*ni:(k+2)*ni]
#             p = self.model.forward(q)
#             J = self.model.jacobian(q)
#
#             pm = self.model.forward_m(q)
#             Jm = self.model.jacobian_m(q)
#
#             fbar[k*no:(k+1)*no] = pm
#             Jbar[k*no:(k+1)*no, k*ni:(k+1)*ni] = Jm
#
#             # pf and ee
#             pf = self.model.forward_f(q)
#             Jf = self.model.jacobian_f(q)
#             d_pf_ee = np.linalg.norm(p - pf)
#             A_pf_ee = -(pf - p).T.dot(Jf - J) / d_pf_ee
#             lbA_pf_ee = d_pf_ee - 0.75
#             Abar[k*num_body_pts, k*ni:(k+1)*ni] = A_pf_ee
#             lbA[k*num_body_pts] = lbA_pf_ee
#
#         dbar = fbar - pr
#
#         H = Rbar + self.dt**2*Ebar.T.dot(Jbar.T).dot(Qbar).dot(Jbar).dot(Ebar)
#         g = u.T.dot(Rbar) + self.dt*dbar.T.dot(Qbar).dot(Jbar).dot(Ebar)
#         A = self.dt*Abar.dot(Ebar)
#
#         return H, g, A, lbA
#
#     def _calc_vel_limits(self, u, ni, N):
#         L = np.ones(ni * N) * self.vel_lim
#         lb = -L - u
#         ub = L - u
#         return lb, ub
#
#     def _calc_acc_limits(self, u, dq0, ni, N):
#         # u_prev consists of [dq0, u_0, u_1, ..., u_{N-2}]
#         # u is [u_0, ..., u_{N-1}]
#         u_prev = np.zeros(ni * N)
#         u_prev[:ni] = dq0
#         u_prev[ni:] = u[:-ni]
#
#         L = self.dt * np.ones(ni * N) * self.acc_lim
#         lbA = -L - u + u_prev
#         ubA = L - u + u_prev
#
#         d1 = np.ones(N)
#         d2 = -np.ones(N - 1)
#
#         # A0 is NxN
#         A0 = sparse.diags((d1, d2), [0, -1]).toarray()
#
#         # kron to make it work for n-dimensional inputs
#         A = np.kron(A0, np.eye(ni))
#
#         return A, lbA, ubA
#
#     def _iterate(self, q0, dq0, pr, u, N):
#         ni = self.model.ni
#
#         # Create the QP, which we'll solve sequentially.
#         # num vars, num constraints (note that constraints only refer to matrix
#         # constraints rather than bounds)
#         # num constraints = N obstacle constraints and ni*N joint acceleration
#         # constraints
#         num_var = ni * N
#         num_constraints = N + ni * N
#         qp = qpoases.PySQProblem(num_var, num_constraints)
#         options = qpoases.PyOptions()
#         options.printLevel = qpoases.PyPrintLevel.NONE
#         qp.setOptions(options)
#
#         # Initial opt problem.
#         H, g, A_obs, lbA_obs = self._lookahead(q0, pr, u, N)
#         ubA_obs = np.infty * np.ones_like(lbA_obs)
#
#         lb, ub = self._calc_vel_limits(u, ni, N)
#         A_acc, lbA_acc, ubA_acc = self._calc_acc_limits(u, dq0, ni, N)
#
#         A = np.vstack((A_obs, A_acc))
#         lbA = np.concatenate((lbA_obs, lbA_acc))
#         ubA = np.concatenate((ubA_obs, ubA_acc))
#
#         ret = qp.init(H, g, A, lb, ub, lbA, ubA, np.array([NUM_WSR]))
#         delta = np.zeros(ni * N)
#         qp.getPrimalSolution(delta)
#         u = u + delta
#
#         # Remaining sequence is hotstarted from the first.
#         for i in range(NUM_ITER - 1):
#             H, g, A_obs, lbA_obs = self._lookahead(q0, pr, u, N)
#             lb, ub = self._calc_vel_limits(u, ni, N)
#             A_acc, lbA_acc, ubA_acc = self._calc_acc_limits(u, dq0, ni, N)
#             A = np.vstack((A_obs, A_acc))
#             lbA = np.concatenate((lbA_obs, lbA_acc))
#             ubA = np.concatenate((ubA_obs, ubA_acc))
#
#             qp.hotstart(H, g, A, lb, ub, lbA, ubA, np.array([NUM_WSR]))
#             qp.getPrimalSolution(delta)
#
#             u = u + delta
#
#         return u
#
#     def solve(self, q0, dq0, pr, N):
#         ''' Solve the MPC problem at current state x0 given desired output
#             trajectory Yd. '''
#         # initialize optimal inputs
#         u = np.zeros(self.model.ni * N)
#
#         # iterate to final solution
#         u = self._iterate(q0, dq0, pr, u, N)
#
#         # return first optimal input
#         return u[:self.model.ni]


class MPC2(object):
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
            future. '''
        ni = self.model.ni  # number of joints
        no = self.model.no  # number of Cartesian outputs

        fbar = np.zeros(no*N)         # Lifted forward kinematics
        Jbar = np.zeros((no*N, ni*N))  # Lifted Jacobian
        Qbar = np.kron(np.eye(N), self.Q)
        Rbar = np.kron(np.eye(N), self.R)

        # lower triangular matrix of ni*ni identity matrices
        Ebar = np.kron(np.tril(np.ones((N, N))), np.eye(ni))

        # Integrate joint positions from the last iteration
        qbar = np.tile(q0, N+1)
        qbar[ni:] = qbar[ni:] + self.dt * Ebar.dot(u)

        num_body_pts = 2+1
        Abar = np.zeros((N*num_body_pts, ni*N))
        lbA = np.zeros(N*num_body_pts)

        for k in range(N):
            q = qbar[(k+1)*ni:(k+2)*ni]
            p = self.model.forward(q)
            J = self.model.jacobian(q)

            pm = self.model.forward_m(q)
            Jm = self.model.jacobian_m(q)

            fbar[k*no:(k+1)*no] = pm
            Jbar[k*no:(k+1)*no, k*ni:(k+1)*ni] = Jm

            # TODO hardcoded radius
            # EE and obstacle
            d_ee_obs = np.linalg.norm(p - pc) - 0.45
            Abar[k*num_body_pts, k*ni:(k+1)*ni] = (p - pc).T.dot(J) / np.linalg.norm(p - pc)
            lbA[k*num_body_pts] = -d_ee_obs

            # base and obstacle
            pb = q[:2]
            Jb = np.array([[1, 0, 0, 0, 0],
                           [0, 1, 0, 0, 0]])
            d_base_obs = np.linalg.norm(pb - pc) - 0.45 - 0.56
            Abar[k*num_body_pts+1, k*ni:(k+1)*ni] = (pb - pc).T.dot(Jb) / np.linalg.norm(pb - pc)
            lbA[k*num_body_pts+1] = -d_base_obs

            # pf and ee: these need to stay close together
            pf = self.model.forward_f(q)
            Jf = self.model.jacobian_f(q)
            d_pf_ee = np.linalg.norm(p - pf)
            A_pf_ee = -(pf - p).T.dot(Jf - J) / d_pf_ee
            lbA_pf_ee = d_pf_ee - 0.75
            Abar[k*num_body_pts+2, k*ni:(k+1)*ni] = A_pf_ee
            lbA[k*num_body_pts+2] = lbA_pf_ee

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
        num_constraints = 3*N + ni * N
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


class EmbraceMPC(object):
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
            future. '''
        ni = self.model.ni  # number of joints
        no = self.model.no  # number of Cartesian outputs

        fbar = np.zeros(no*N)         # Lifted forward kinematics
        Jbar = np.zeros((no*N, ni*N))  # Lifted Jacobian
        Qbar = np.kron(np.eye(N), self.Q)
        Rbar = np.kron(np.eye(N), self.R)

        # lower triangular matrix of ni*ni identity matrices
        Ebar = np.kron(np.tril(np.ones((N, N))), np.eye(ni))

        # Integrate joint positions from the last iteration
        qbar = np.tile(q0, N+1)
        qbar[ni:] = qbar[ni:] + self.dt * Ebar.dot(u)

        # TODO: need to integrate pc as well: this takes the place of fbar

        num_body_pts = 2+1
        Abar = np.zeros((N*num_body_pts, ni*N))
        lbA = np.zeros(N*num_body_pts)

        for k in range(N):
            q = qbar[(k+1)*ni:(k+2)*ni]

            # calculate points defining front of base
            pb = q[:2]
            θb = q[2]
            R = util.rotation_matrix(θb)
            rx = 0.5
            ry = 0.25
            p1 = R.dot(np.array([rx, ry]))
            p2 = R.dot(np.array([rx, -ry]))

            # pf is the closest point to the line segment
            pf, _ = util.dist_to_line_segment(pc, p1, p2)

            # transform into body frame
            b_pf = R.T.dot(pf - pb)

            JR = util.rotation_jacobian(θb)
            Jf = np.hstack((R, JR.dot(pb + b_pf)[:, None], np.zeros((2, 2))))

            pe = self.model.forward(q)
            Je = self.model.jacobian(q)

            re = (pc - pe) / np.linalg.norm(pc - pe)
            rf = (pc - pf) / np.linalg.norm(pc - pf)

            # propagate center of object forward
            pc = pc + self.dt*(Jf + Je).dot(u)

            # fbar[k*no:(k+1)*no] = pm
            # Jbar[k*no:(k+1)*no, k*ni:(k+1)*ni] = Jm

            # # TODO hardcoded radius
            # # EE and obstacle
            # d_ee_obs = np.linalg.norm(p - pc) - 0.45
            # Abar[k*num_body_pts, k*ni:(k+1)*ni] = (p - pc).T.dot(J) / np.linalg.norm(p - pc)
            # lbA[k*num_body_pts] = -d_ee_obs
            #
            # # base and obstacle
            # pb = q[:2]
            # Jb = np.array([[1, 0, 0, 0, 0],
            #                [0, 1, 0, 0, 0]])
            # d_base_obs = np.linalg.norm(pb - pc) - 0.45 - 0.56
            # Abar[k*num_body_pts+1, k*ni:(k+1)*ni] = (pb - pc).T.dot(Jb) / np.linalg.norm(pb - pc)
            # lbA[k*num_body_pts+1] = -d_base_obs
            #
            # # pf and ee: these need to stay close together
            # pf = self.model.forward_f(q)
            # Jf = self.model.jacobian_f(q)
            # d_pf_ee = np.linalg.norm(p - pf)
            # A_pf_ee = -(pf - p).T.dot(Jf - J) / d_pf_ee
            # lbA_pf_ee = d_pf_ee - 0.75
            # Abar[k*num_body_pts+2, k*ni:(k+1)*ni] = A_pf_ee
            # lbA[k*num_body_pts+2] = lbA_pf_ee

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
        num_constraints = 3*N + ni * N
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
