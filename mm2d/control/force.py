import numpy as np
import qpoases
import IPython


NUM_WSR = 100    # number of working set recalculations


class AdmittanceController(object):
    ''' Basic EE admittance controller. This is a minor variation of the
        DiffIKController with a different reference velocity.
        Solves:
            min  0.5*||Ju - v||^2 + 0.5*u'Wu
            s.t. lb <= u <= ub
        where
            v = K*(pd-p) + vd - inv(C)f '''
    def __init__(self, model, W, K, C, dt, vel_lim, acc_lim, verbose=False):
        self.model = model
        self.W = W
        self.K = K
        self.Cinv = np.linalg.inv(C)
        self.dt = dt
        self.vel_lim = vel_lim
        self.acc_lim = acc_lim
        self.verbose = verbose

    def solve(self, q, dq, pd, vd, f):
        ''' Solve for the optimal inputs. '''
        ni = self.model.ni

        # forward kinematics
        p = self.model.forward(q)
        J = self.model.jacobian(q)

        # calculate velocity reference
        v = self.K.dot(pd - p) + vd - self.Cinv.dot(f)

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


# TODO need to revisit this
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
            ret = qp.init(H, g, lb, ub, np.array([NUM_WSR]))
        else:
            qp = qpoases.PyQProblemB(n)
            options = qpoases.PyOptions()
            options.printLevel = qpoases.PyPrintLevel.NONE
            qp.setOptions(options)
            ret = qp.init(H, g, lb, ub, np.array([NUM_WSR]))

        dq = np.zeros(n)
        qp.getPrimalSolution(dq)
        return dq
