# Two-dimensional model of a mobile manipulator, with a base and two link arm.
# Kinematic and dynamic models are provided.
import numpy as np
from mm2d.util import bound_array

# default parameters
Mb = 10
M1 = 1
M2 = 1

Lx = 0
Ly = 0
L1 = 1
L2 = 1

# base width and height
Bw = 1.0
Bh = 0.25

G = 9.8

# limits
VEL_LIM = 1
ACC_LIM = 1
TAU_LIM = 100


class ThreeInputModel:
    """Three-input 2D mobile manipulator kinematic and dynamic model.

    The robot consists of mobile base (1 DOF: base x-position xb) and a 2-link
    arm (2 DOF: joint angles θ1 and θ2).
    The configuration vector is thus q = [xb, θ1, θ2].
    """
    def __init__(self, bh=Bh, bw=Bw, lx=Lx, ly=Ly, l1=L1, l2=L2, mb=Mb, m1=M1,
                 m2=M2, gravity=G, vel_lim=VEL_LIM, acc_lim=ACC_LIM,
                 tau_lim=TAU_LIM, output_idx=[0, 1, 2]):
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

        # base size
        self.bh = bh
        self.bw = bw

        # link masses
        self.mb = mb
        self.m1 = m1
        self.m2 = m2

        # link inertias
        self.I1 = m1 * l1**2 / 12
        self.I2 = m2 * l2**2 / 12

        self.gravity = gravity

        self.tau_lim = tau_lim
        self.vel_lim = vel_lim
        self.acc_lim = acc_lim

    def forward(self, q):
        ''' Forward kinematic transform for the end effector. '''
        p = np.array([
            self.lx + q[0] + self.l1*np.cos(q[1]) + self.l2*np.cos(q[1]+q[2]),
            self.ly + self.l1*np.sin(q[1]) + self.l2*np.sin(q[1]+q[2]),
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

    def base_corners(self, q):
        ''' Calculate the corners of the base of the robot. '''
        x0 = q[0]
        y0 = 0
        r = self.bw * 0.5
        h = self.bh

        x = np.array([x0 - r, x0 - r, x0 + r, x0 + r])
        y = np.array([y0, y0 - h, y0 - h, y0])

        return x, y

    def arm_points(self, q):
        ''' Calculate points on the arm. '''
        x0 = q[0] + self.lx
        x1 = x0 + self.l1*np.cos(q[1])
        x2 = x1 + self.l2*np.cos(q[1]+q[2])

        y0 = self.ly
        y1 = y0 + self.l1*np.sin(q[1])
        y2 = y1 + self.l2*np.sin(q[1]+q[2])

        x = np.array([x0, x1, x2])
        y = np.array([y0, y1, y2])

        return x, y

    def sample_points(self, q):
        ''' Sample points across the robot body. '''
        ax, ay = self.arm_points(q)
        ps = np.array([
            [q[0] - 0.5, 0],
            [q[0] + 0.5, 0],
            [q[0], 0],
            [ax[1], ay[1]],
            [ax[2], ay[2]]])
        return ps

    def sample_jacobians(self, q):
        ''' Jacobians of points sampled across the robot body. '''
        Js = np.zeros((5, 2, 3))
        Js[0, :, :] = Js[1, :, :] = Js[2, :, :] = np.array([[1, 0, 0], [0, 0, 0]])
        Js[3, :, :] = np.array([
            [1, -self.l1*np.sin(q[1]), 0],
            [0,  self.l1*np.cos(q[1]), 0]])
        Js[4, :, :] = np.array([
            [1, -self.l1*np.sin(q[1])-self.l2*np.sin(q[1]+q[2]), -self.l2*np.sin(q[1]+q[2])],
            [0,  self.l1*np.cos(q[1])+self.l2*np.cos(q[1]+q[2]),  self.l2*np.cos(q[1]+q[2])]])
        return Js

    def sample_dJdt(self, q, dq):
        ''' Time-derivative of Jacobians of points sampled across the robot
            body. '''
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

    def calc_torque(self, q, dq, ddq):
        ''' Calculate the required torque for the given joint positions,
            velocity, and accelerations. '''
        M = self.mass_matrix(q)
        Γ = self.christoffel_matrix(q)
        g = self.gravity_vector(q)

        return M @ ddq + dq @ Γ @ dq + g

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

    def step(self, q, u, dt, dq_last=None):
        ''' Step forward one timestep. '''
        # velocity limits
        dq = bound_array(u, -self.vel_lim, self.vel_lim)

        # acceleration limits
        if dq_last is not None:
            dq = bound_array(dq, -self.acc_lim * dt + dq_last, self.acc_lim * dt + dq_last)

        q = q + dt * dq
        return q, dq
