# Models for mm2d.
# Philosophy of models is that they are actually stateless, but rather just
# store parameters and differential equations that define the system's
# evolution.
import numpy as np
from .util import bound_array


class InvertedPendulum(object):
    def __init__(self, length, mass, gravity=9.81):
        self.length = length
        self.mass = mass
        self.gravity = gravity

        # matrices of the linearized system
        self.A = np.array([[0, 1, 0, 0],
                           [gravity/length, 0, 0, 0],
                           [0, 0, 0, 1],
                           [0, 0, 0, 0]])
        self.B = np.array([0, 1/length, 0, 1])

    def calc_force(self, X, x_acc):
        angle = X[0]
        s = np.sin(angle)
        c = np.cos(angle)
        acc_normal = self.gravity * c - x_acc * s

        # force exerted on the EE by the pendulum
        f = np.array([self.mass * acc_normal * s, -self.mass * acc_normal * c])

        return f

    def step(self, X, u, dt):
        ''' State X = [angle, dangle, x, dx], input u = ddx '''
        angle = X[0]
        s = np.sin(angle)
        c = np.cos(angle)

        acc_tangential = self.gravity * s + u * c
        angle_acc = acc_tangential / self.length

        dX = np.array([X[1], angle_acc, X[3], u])
        X = X + dt * dX

        return X


class ThreeInputModel(object):
    ''' Three-input 2D mobile manipulator. Consists of mobile base (1 input)
        and 2-link arm (2 inputs). State is q = [x_b, q_1, q_2]; inputs u = dq. '''
    def __init__(self, l1, l2, vel_lim, acc_lim, output_idx=[0,1,2]):
        self.ni = 3  # number of joints (inputs/DOFs)

        # control which outputs are used
        # possible outputs are: x, y, theta
        self.no = len(output_idx)
        self.output_idx = output_idx

        self.l1 = l1
        self.l2 = l2

        self.vel_lim = vel_lim
        self.acc_lim = acc_lim

    def pos_all(self, q):
        # elementary points
        p0 = np.array([q[0], 0])
        p1 = p0 + self.l1 * np.array([np.cos(q[1]), np.sin(q[1])])
        p2 = p1 + self.l2 * np.array([np.cos(q[1]+q[2]), np.sin(q[1]+q[2])])

        # evenly divide the links
        l1s = np.linspace(0, self.l1, 10)
        l2s = np.linspace(0, self.l2, 10)

        p0s = p0 + np.array([[-0.5, 0], [0.5, 0], [0, 0]])
        p1s = p0 + np.outer(l1s, np.array([np.cos(q[1]), np.sin(q[1])]))
        p2s = p1 + np.outer(l2s, np.array([np.cos(q[1]+q[2]), np.sin(q[1]+q[2])]))

        ps = np.concatenate((p0s, p1s, p2s))

        ps = np.array([
            [q[0] - 0.5, 0],
            [q[0] + 0.5, 0],
            [q[0], 0],
            [q[0] + self.l1*np.cos(q[1]), self.l1*np.sin(q[1])],
            [q[0] + self.l1*np.cos(q[1]) + self.l2*np.cos(q[1]+q[2]), self.l1*np.sin(q[1]) + self.l2*np.sin(q[1]+q[2])]])
        return ps

    def jac_all(self, q):
        # elementary Jacobians
        J0 = np.array([[1, 0, 0],
                       [0, 0, 0]])
        J1 = np.array([[0, -np.sin(q[1]), 0],
                       [0,  np.cos(q[1]), 0]])
        J2 = np.array([[0, -np.sin(q[1]+q[2]), -np.sin(q[1]+q[2])],
                       [0,  np.cos(q[1]+q[2]),  np.cos(q[1]+q[2])]])

        l1s = np.linspace(0, self.l1, 10)
        l2s = np.linspace(0, self.l2, 10)

        J0s = np.kron(np.ones((3, 1)), J0)
        J1s = J0s[-1, :, :] + np.kron(l1s[:, None], J1)
        J2s = J1s[-1, :, :] + np.kron(l2s[:, None], J2)

        Js = np.concatenate((J0s, J1s, J2s))


        Js = np.zeros((5, 2, 3))
        Js[0, :, :] = Js[1, :, :] = Js[2, :, :] = np.array([[1, 0, 0], [0, 0, 0]])
        Js[3, :, :] = np.array([
            [1, -self.l1*np.sin(q[1]), 0],
            [0,  self.l1*np.cos(q[1]), 0]])
        Js[4, :, :] = np.array([
            [1, -self.l1*np.sin(q[1])-self.l2*np.sin(q[1]+q[2]), -self.l2*np.sin(q[1]+q[2])],
            [0,  self.l1*np.cos(q[1])+self.l2*np.cos(q[1]+q[2]),  self.l2*np.cos(q[1]+q[2])]])
        return Js

    def dJdt_all(self, q, dq):
        dJs = np.zeros((5, 2, 3))
        dJs[3, :, :] = np.array([
            [0, -self.l1*np.cos(q[1])*dq[1], 0],
            [0, -self.l1*np.sin(q[1])*dq[1], 0]])
        q12 = q[1] + q[2]
        dq12 = dq[1] + dq[2]
        dJs[4, :, :] = np.array([
            [0, -self.l1*np.cos(q[1])*dq[1]-self.l2*np.cos(q12)*dq12, -self.l2*np.cos(q12)*dq12],
            [0, -self.l1*np.sin(q[1])*dq[1]-self.l2*np.sin(q12)*dq12, -self.l2*np.sin(q12)*dq12]])
        return dJs

    def forward(self, q):
        ''' Forward kinematic transform for the end effector. '''
        p = np.array([q[0] + self.l1*np.cos(q[1]) + self.l2*np.cos(q[1]+q[2]),
                      self.l1*np.sin(q[1]) + self.l2*np.sin(q[1]+q[2]),
                      q[1] + q[2]])
        return p[self.output_idx]

    def jacobian(self, q):
        ''' End effector Jacobian. '''
        J = np.array([
            [1, -self.l1*np.sin(q[1])-self.l2*np.sin(q[1]+q[2]), -self.l2*np.sin(q[1]+q[2])],
            [0,  self.l1*np.cos(q[1])+self.l2*np.cos(q[1]+q[2]),  self.l2*np.cos(q[1]+q[2])],
            [0, 1, 1]])
        return J[self.output_idx, :]

    def dJdt(self, q, dq):
        ''' Derivative of EE Jacobian w.r.t. time. '''
        q12 = q[1] + q[2]
        dq12 = dq[1] + dq[2]
        J = np.array([
            [0, -self.l1*np.cos(q[1])*dq[1]-self.l2*np.cos(q12)*dq12, -self.l2*np.cos(q12)*dq12],
            [0, -self.l1*np.sin(q[1])*dq[1]-self.l2*np.sin(q12)*dq12, -self.l2*np.sin(q12)*dq12],
            [0, 0, 0]])
        return J[self.output_idx, :]

    def step(self, q, u, dt, dq_last=None):
        ''' Step forward one timestep. '''
        # velocity limits
        dq = bound_array(u, -self.vel_lim, self.vel_lim)

        # acceleration limits
        if dq_last is not None:
            dq = bound_array(dq, -self.acc_lim * dt + dq_last, self.acc_lim * dt + dq_last)

        q = q + dt * dq
        return q, dq
