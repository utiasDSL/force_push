import numpy as np
import qpoases
import osqp
from scipy import sparse


class Objective:
    """Objective function fun(x) with Jacobian jac(x) and Hessian hess(x)."""
    def __init__(self, fun, jac, hess):
        self.fun = fun
        self.jac = jac
        self.hess = hess

    def value(self, *args, dtype=float):
        return np.array(self.fun(*args), dtype=dtype)

    def jacobian(self, *args, dtype=float):
        return np.array(self.jac(*args), dtype=dtype)

    def hessian(self, *args, dtype=float):
        return np.array(self.hess(*args), dtype=dtype)


# class MultiObjective:
#     def __init__(self, objectives):
#         self.objectives = objectives
#
#     def value(self, *args, dtype=float):
#         return sum([obj.value(*args, dtype=dtype) for obj in self.objectives])
#
#     def jacobian(self, *args, dtype=float):
#         return sum([obj.jacobian(*args, dtype=dtype) for obj in self.objectives])
#
#     def hessian(self, *args, dtype=float):
#         return sum([obj.hessian(*args, dtype=dtype) for obj in self.objectives])


class Constraints:
    """Constraints of the form lb <= fun(x) <= ub.

    jac is the Jacobian of fun w.r.t. x
    nz_idx is the (row, column) indices for elements of the linearized
    constraint matrix that are in general non-zero. This is used for approaches
    that represent matrices sparsely, such as OSQP.
    """
    def __init__(self, fun, jac, lb, ub, nz_idx=None):
        self.fun = fun
        self.jac = jac
        self.lb = lb
        self.ub = ub
        self.nz_idx = nz_idx


class Bounds:
    """Simple bounds of the form lb <= x <= ub."""
    def __init__(self, lb, ub):
        self.lb = lb
        self.ub = ub


def SQP(*args, solver="qpoases", **kwargs):
    if solver == "qpoases":
        return SQP_qpOASES(*args, **kwargs)
    elif solver == "osqp":
        return SQP_OSQP(*args, **kwargs)
    else:
        raise Exception(f"Unknown solver {solver}")


class SQP_qpOASES(object):
    """Generic sequential quadratic program based on qpOASES solver."""
    def __init__(self, nv, nc, obj_func, constraints, bounds, num_iter=3,
                 num_wsr=100, verbose=False):
        """Initialize the SQP."""
        self.nv = nv
        self.nc = nc
        self.num_iter = num_iter
        self.num_wsr = num_wsr

        self.obj_func = obj_func
        self.constraints = constraints
        self.bounds = bounds

        self.qp = qpoases.PySQProblem(nv, nc)
        options = qpoases.PyOptions()
        options.setToReliable()
        if verbose:
            options.printLevel = qpoases.PyPrintLevel.MEDIUM
        else:
            options.printLevel = qpoases.PyPrintLevel.LOW
        self.qp.setOptions(options)

        self.qp_initialized = False

    def _lookahead(self, x0, xd, var):
        """Generate lifted matrices proprogating the state N timesteps into the
           future."""
        H, g = self.obj_func(x0, xd, var)
        H = np.array(H, dtype=np.float64)
        g = np.array(g, dtype=np.float64)

        A = np.array(self.constraints.jac(x0, xd, var), dtype=np.float64)
        a = np.array(self.constraints.fun(x0, xd, var), dtype=np.float64)
        lbA = self.constraints.lb - a
        ubA = self.constraints.ub - a

        lb = self.bounds.lb - var
        ub = self.bounds.ub - var

        # if np.any(np.linalg.eigvals(H) < 0):
        #     import IPython
        #     IPython.embed()

        return H, g, \
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
        """Solve the MPC problem at current state x0 given desired trajectory
           xd."""
        # initialize decision variables
        var = np.zeros(self.nv)

        # iterate to final solution
        var = self._iterate(x0, xd, var)

        # return first optimal input
        return var


class SQP_OSQP(object):
    """Generic sequential quadratic program based on OSQP solver."""
    def __init__(self, nv, nc, obj_func, constraints, bounds, num_iter=3, verbose=False):
        """Initialize the SQP."""
        self.nv = nv
        self.nc = nc
        self.num_iter = num_iter

        self.obj_func = obj_func
        self.constraints = constraints
        self.bounds = bounds

        self.qp = osqp.OSQP()
        self.verbose = verbose
        self.qp_initialized = False

    def _lookahead(self, x0, xd, var):
        """Generate lifted matrices proprogating the state N timesteps into the
           future."""
        H, g = self.obj_func(x0, xd, var)

        A = self.constraints.jac(x0, xd, var)
        a = self.constraints.fun(x0, xd, var)
        lbA = np.array(self.constraints.lb - a)
        ubA = np.array(self.constraints.ub - a)

        lb = np.array(self.bounds.lb - var)
        ub = np.array(self.bounds.ub - var)

        H = sparse.triu(np.array(H), format="csc")
        g = np.array(g)

        # there may be some zeros that are always zero, so we use an explicit
        # sparsity pattern
        nz_idx = self.constraints.nz_idx
        A1 = sparse.csc_matrix((np.array(A)[nz_idx], nz_idx), shape=A.shape)
        A2 = sparse.eye(self.nv)
        A = sparse.vstack((A1, A2), format="csc")

        # since OSQP does not use separate bounds, we concatenate onto the
        # constraints
        lower = np.concatenate((lbA, lb))
        upper = np.concatenate((ubA, ub))

        # print(f"nnz(N) = {H.nnz}")
        # print(f"nnz(A) = {A.nnz}")

        return H, g, A, lower, upper

    def _iterate(self, x0, xd, var):
        # Initial opt problem.
        H, g, A, lower, upper = self._lookahead(x0, xd, var)
        if not self.qp_initialized:
            self.qp.setup(P=H, q=g, A=A, l=lower, u=upper, verbose=self.verbose)
            self.qp_initialized = True
        else:
            self.qp.update(Px=H.data, q=g, Ax=A.data, l=lower, u=upper)
        results = self.qp.solve()
        var = var + results.x

        # Remaining sequence is hotstarted from the first.
        for i in range(self.num_iter - 1):
            H, g, A, lower, upper = self._lookahead(x0, xd, var)
            self.qp.update(Px=H.data, q=g, Ax=A.data, l=lower, u=upper)
            results = self.qp.solve()
            var = var + results.x

        return var

    def solve(self, x0, xd):
        """Solve the MPC problem at current state x0 given desired trajectory
           xd."""
        # initialize decision variables
        var = np.zeros(self.nv)

        # iterate to final solution
        var = self._iterate(x0, xd, var)

        # return first optimal input
        return var
