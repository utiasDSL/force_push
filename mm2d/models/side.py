# Two-dimensional model of a mobile manipulator, with a base and two link arm.
# Kinematic and dynamic models are provided.
import numpy as np
from mm2d.util import bound_array

# default parameters
Mb = 1
M1 = 1
M2 = 1

Lx = 0
Ly = 0
L1 = 1
L2 = 0.5

G = 9.8


class ThreeInputKinematicModel:
    ''' Three-input 2D mobile manipulator. Consists of mobile base (1 input)
        and 2-link arm (2 inputs). State is q = [x_b, q_1, q_2]; inputs u = dq. '''
    def __init__(self, vel_lim, acc_lim, lx=Lx, ly=Ly, l1=L1, l2=L2,
                 output_idx=[0, 1, 2]):
        self.ni = 3  # number of joints (inputs/DOFs)

        # control which outputs are used
        # possible outputs are: x, y, theta
        self.no = len(output_idx)
        self.output_idx = output_idx

        # TODO need to account for LX and LY in the forward and differential
        # kinematics
        self.lx = lx
        self.ly = ly
        self.l1 = l1
        self.l2 = l2

        self.vel_lim = vel_lim
        self.acc_lim = acc_lim

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

    def step(self, q, u, dt, dq_last=None):
        ''' Step forward one timestep. '''
        # velocity limits
        dq = bound_array(u, -self.vel_lim, self.vel_lim)

        # acceleration limits
        if dq_last is not None:
            dq = bound_array(dq, -self.acc_lim * dt + dq_last, self.acc_lim * dt + dq_last)

        # if not (u == dq).all():
        #     print('limits hit')

        q = q + dt * dq
        return q, dq


class ThreeInputDynamicModel:
    ''' Three-input 2D mobile manipulator. Consists of mobile base (1 input)
        and 2-link arm (2 inputs). State is q = [x_b, q_1, q_2]; inputs u = dq. '''
    def __init__(self, tau_lim, lx=Lx, ly=Ly, l1=L1, l2=L2, mb=Mb, m1=M1,
                 m2=M2, gravity=G, output_idx=[0, 1, 2]):
        self.ni = 3  # number of joints (inputs/DOFs)

        # control which outputs are used
        # possible outputs are: x, y, theta
        self.no = len(output_idx)
        self.output_idx = output_idx

        # lengths
        self.lx = lx
        self.ly = ly
        self.l1 = l1
        self.l2 = l2

        # link masses
        self.mb = mb
        self.m1 = m1
        self.m2 = m2

        # link inertias
        self.I1 = m1 * l1**2 / 12
        self.I2 = m2 * l2**2 / 12

        self.gravity = gravity

        self.tau_lim = tau_lim

    def mass_matrix(self, q):
        ''' Compute dynamic mass matrix. '''
        xb, θ1, θ2 = q
        θ12 = θ1 + θ2

        m11 = self.mb + self.m1 + self.m2
        m12 = -(0.5*self.m1+self.m2)*self.l1*np.sin(θ1) \
                - 0.5*self.m2*self.l2*np.sin(θ12)
        m13 = -0.5*self.m2*self.l2*np.sin(θ12)

        m22 = (0.25*self.m1+self.m2)*self.l1**2 + 0.25*self.m2*self.l2**2 \
                + self.m2*self.l1*self.l2*np.cos(θ2) + self.I1 + self.I2
        m23 = 0.5*self.m2*self.l2*(0.5*self.l2+self.l1*np.cos(θ2)) + self.I2

        m33 = 0.25*self.m2*self.l2**2 + self.I2

        return np.array([
            [m11, m12, m13],
            [m12, m22, m23],
            [m13, m23, m33]])

    def christoffel_matrix(self, q):
        ''' Compute 3D matrix Γ of Christoffel symbols, as in the dynamic
            equations of motion:
                M @ ddq + dq @ Γ @ dq + g = τ.
            Note that C = dq @ Γ.
        '''
        xb, θ1, θ2 = q
        θ12 = θ1 + θ2

        # Partial derivatives of mass matrix
        dMdxb = np.zeros((3, 3))

        dMdθ1_12 = -0.5*self.m1*self.l1*np.cos(θ1) \
                - self.m2*self.l1*np.cos(θ1) - 0.5*self.m2*self.l2*np.cos(θ12)
        dMdθ1_13 = -0.5*self.m2*self.l2*np.cos(θ12)
        dMdθ1 = np.array([
            [0, dMdθ1_12, dMdθ1_13],
            [dMdθ1_12, 0, 0],
            [dMdθ1_13, 0, 0]])

        dMdθ2_12 = -0.5*self.m2*self.l2*np.cos(θ12)
        dMdθ2_13 = -0.5*self.m2*self.l2*np.cos(θ12)
        dMdθ2_22 = -self.m2*self.l1*self.l2*np.sin(θ2)
        dMdθ2_23 = -0.5*self.m2*self.l1*self.l2*np.sin(θ2)
        dMdθ2 = np.array([
            [0,        dMdθ2_12, dMdθ2_13],
            [dMdθ2_12, dMdθ2_22, dMdθ2_23],
            [dMdθ2_13, dMdθ2_23, 0]])

        dMdq = np.zeros((3, 3, 3))
        dMdq[:, :, 0] = dMdxb
        dMdq[:, :, 1] = dMdθ1
        dMdq[:, :, 2] = dMdθ2

        return dMdq - 0.5*dMdq.T

    def gravity_vector(self, q):
        ''' Calculate the gravity vector. '''
        xb, θ1, θ2 = q
        θ12 = θ1 + θ2
        return np.array([
            0,
            (0.5*self.m1+self.m2)*self.gravity*self.l1*np.cos(θ1) \
                    + 0.5*self.m2*self.l2*self.gravity*np.cos(θ12),
            0.5*self.m2*self.l2*self.gravity*np.cos(θ12)])

    def command_torque(self, q, dq, tau, dt):
        ''' Calculate the new state [q, dq] from current state [q, dq] and
            torque input tau. '''
        M = self.mass_matrix(q)
        Γ = self.christoffel_matrix(q)
        g = self.gravity_vector(q)

        # solve for acceleration
        ddq = np.linalg.solve(M, tau - dq @ Γ @ dq - g)

        # integrate the state
        q = q + dt * dq
        dq = dq + dt * ddq

        return q, dq
