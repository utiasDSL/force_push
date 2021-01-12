import numpy as np
import qpoases


class Objective:
    def __init__(self, fun, jac, hess):
        self.fun = fun
        self.jac = jac
        self.hess = hess


class Constraints:
    def __init__(self, fun, jac, lb, ub):
        self.fun = fun
        self.jac = jac
        self.lb = lb
        self.ub = ub


class Bounds:
    def __init__(self, lb, ub):
        self.lb = lb
        self.ub = ub


class SQP(object):
    """Generic sequential quadratic program."""
    def __init__(self, nv, nc, objective, constraints, bounds, num_iter=3,
                 num_wsr=100, verbose=False):
        """Initialize the SQP."""
        self.nv = nv
        self.nc = nc
        self.num_iter = num_iter
        self.num_wsr = num_wsr

        self.objective = objective
        self.constraints = constraints
        self.bounds = bounds

        self.qp = qpoases.PySQProblem(nv, nc)
        if not verbose:
            options = qpoases.PyOptions()
            options.printLevel = qpoases.PyPrintLevel.NONE
            self.qp.setOptions(options)

        self.qp_initialized = False

    def _lookahead(self, x0, xd, var):
        ''' Generate lifted matrices proprogating the state N timesteps into the
            future. '''
        H = self.objective.hess(x0, xd, var)
        g = self.objective.jac(x0, xd, var)

        A = self.constraints.jac(x0, xd, var)
        a = self.constraints.fun(x0, xd, var)
        lbA = self.constraints.lb - a
        ubA = self.constraints.ub - a

        lb = self.bounds.lb - var
        ub = self.bounds.ub - var

        return np.array(H, dtype=np.float64), np.array(g, dtype=np.float64), \
               np.array(A, dtype=np.float64), np.array(lbA, dtype=np.float64), \
               np.array(ubA, dtype=np.float64), np.array(lb, dtype=np.float64), \
               np.array(ub, dtype=np.float64)

    def _iterate(self, x0, xd, var):
        delta = np.zeros(self.nv)

        # Initial opt problem.
        H, g, A, lbA, ubA, lb, ub = self._lookahead(x0, xd, var)
        if not self.qp_initialized:
            self.qp.init(H, g, A, lb, ub, lbA, ubA, np.array([self.num_wsr]))
            self.qp_initialized = True
        else:
            self.qp.hotstart(H, g, A, lb, ub, lbA, ubA, np.array([self.num_wsr]))
        self.qp.getPrimalSolution(delta)
        var = var + delta

        # Remaining sequence is hotstarted from the first.
        for i in range(self.num_iter - 1):
            H, g, A, lbA, ubA, lb, ub = self._lookahead(x0, xd, var)
            self.qp.hotstart(H, g, A, lb, ub, lbA, ubA, np.array([self.num_wsr]))
            self.qp.getPrimalSolution(delta)
            var = var + delta

        return var

    def solve(self, x0, xd):
        ''' Solve the MPC problem at current state x0 given desired trajectory
            xd. '''
        # initialize decision variables
        var = np.zeros(self.nv)

        # iterate to final solution
        var = self._iterate(x0, xd, var)

        # return first optimal input
        return var
