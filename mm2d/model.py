import numpy as np
from .util import bound_array


class ThreeInputModel(object):
    def __init__(self, l1, l2, lb, ub, output_idx=[0,1,2]):
        self.ni = 3  # number of joints (inputs/DOFs)

        # control which outputs are used
        # possible outputs are: x, y, theta
        self.no = len(output_idx)
        self.output_idx = output_idx

        self.l1 = l1
        self.l2 = l2

        self.lb = lb
        self.ub = ub

    def pos_all(self, q):
        ps = np.array([
            [q[0] - 0.5, 0],
            [q[0] + 0.5, 0],
            [q[0], 0],
            [q[0] + self.l1*np.cos(q[1]), self.l1*np.sin(q[1])],
            [q[0] + self.l1*np.cos(q[1]) + self.l2*np.cos(q[1]+q[2]), self.l1*np.sin(q[1]) + self.l2*np.sin(q[1]+q[2])]])
        return ps

    def jac_all(self, q):
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

    def base(self, q):
        ''' Generate an array of points representing the base of the robot. '''
        x0 = q[0]
        y0 = 0
        r = 0.5
        h = 0.25

        x = np.array([x0, x0 - r, x0 - r, x0 + r, x0 + r, x0])
        y = np.array([y0, y0, y0 - h, y0 - h, y0, y0])

        return x, y

    def arm(self, q):
        ''' Generate an array of points representing the arm of the robot. '''
        x0 = q[0]
        x1 = x0 + self.l1*np.cos(q[1])
        x2 = x1 + self.l2*np.cos(q[1]+q[2])

        y0 = 0
        y1 = y0 + self.l1*np.sin(q[1])
        y2 = y1 + self.l2*np.sin(q[1]+q[2])

        x = np.array([x0, x1, x2])
        y = np.array([y0, y1, y2])

        return x, y

    def step(self, q, u, dt):
        ''' Step forward one timestep. '''
        dq = bound_array(u, self.lb, self.ub)
        q = q + dt * dq
        return q, dq
