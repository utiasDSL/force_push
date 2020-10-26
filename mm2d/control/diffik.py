import numpy as np
from scipy import sparse
import qpoases
import IPython


NUM_WSR = 100    # number of working set recalculations


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
        ret = qp.init(H, g, A, lb, ub, lbA, ubA, np.array([NUM_WSR]))

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
        ret = qp.init(H, g, lb, ub, np.array([NUM_WSR]))

        u = np.zeros(ni)
        qp.getPrimalSolution(u)
        return u


class ConstrainedDiffIKController(object):
    ''' Constrained differential IK controller.
        Solves:
            min  0.5*||Ju - v||^2 + 0.5*u'Wu
            s.t. lb  <=  u <= ub
                 lbA <= Au <= ubA
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

    def solve(self, q, dq, pd, vd, A=None, lbA=None, ubA=None):
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

        qp = qpoases.PyQProblem(ni, A.shape[0] if A is not None else 0)
        if not self.verbose:
            options = qpoases.PyOptions()
            options.printLevel = qpoases.PyPrintLevel.NONE
            qp.setOptions(options)
        ret = qp.init(H, g, A, lb, ub, lbA, ubA, np.array([NUM_WSR]))

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
        ret = qp.init(H, g, lb, ub, np.array([NUM_WSR]))

        u = np.zeros(ni)
        qp.getPrimalSolution(u)
        return u
