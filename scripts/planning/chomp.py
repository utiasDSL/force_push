#!/usr/bin/env python

import numpy as np
import matplotlib.pyplot as plt
from scipy import sparse
import IPython

from mm2d import models


# optimize over q1...qn, with q0 and qn+1 the fixed end points


class CircleField:
    def __init__(self, c, r):
        self.c = c
        self.r = r

    def signed_dist(self, x):
        return np.linalg.norm(x - self.c) - self.r

    def signed_dist_grad(self, x):
        return (x - self.c) / np.linalg.norm(x - self.c)

    def cost(self, x, eps):
        d = self.signed_dist(x)
        if d <= 0:
            return -d + 0.5 * eps
        elif d <= eps:
            return (d-eps)**2 / (2*eps)
        return 0

    def cost_grad(self, x, eps):
        d = self.signed_dist(x)
        dg = self.signed_dist_grad(x)
        if d <= 0:
            return -dg
        elif d <= eps:
            return -(d - eps) * dg / eps
        return np.zeros(dg.shape)


class FloorField:
    def __init__(self, y):
        self.y = y

    def signed_dist(self, p):
        return p[1] - self.y

    def signed_dist_grad(self, p):
        return np.sign([0, p[1]])

    def cost(self, p, eps):
        d = self.signed_dist(p)
        if d <= 0:
            return d**2
        return 0

    def cost_grad(self, x, eps):
        d = self.signed_dist(x)
        dg = self.signed_dist_grad(x)
        if d <= 0:
            return 2*d*dg
        return np.zeros(dg.shape)


class ObstacleField:
    def __init__(self, obstacles):
        self.obstacles = obstacles

    def cost(self, p, eps):
        cost = np.sum([obs.cost(p, eps) for obs in self.obstacles])
        return cost

    def cost_grad(self, p, eps):
        grad = np.sum([obs.cost_grad(p, eps) for obs in self.obstacles], axis=0)
        return grad


def fd1(N, n, q0, qf):
    ''' First-order finite differencing matrix. '''
    # construct the finite differencing matrix
    d1 = np.ones(N + 1)
    d2 = -np.ones(N)

    # K0 is N+1 x N
    K0 = sparse.diags((d1, d2), [0, -1]).toarray()[:, :-1]

    # kron to make it work for n-dimensional inputs
    K = np.kron(K0, np.eye(n))

    e = np.zeros((N+1) * n)
    e[:n] = -q0
    e[-n:] = qf

    return K, e


def fd2(N, n, q0, qf):
    ''' Second-order finite differencing matrix. '''
    # construct the finite differencing matrix
    d1 = -2*np.ones(N)
    d2 = np.ones(N - 1)

    # K0 is N x N
    K0 = sparse.diags((d2, d1, d2), [1, 0, -1]).toarray()

    # kron to make it work for n-dimensional inputs
    K = np.kron(K0, np.eye(n))

    e = np.zeros(N * n)
    e[:n] = q0
    e[-n:] = qf

    return K, e


def motion_grad(model, traj, q0, qf, N):
    ''' Compute the prior motion/smoothness gradient for the entire trajectory. '''
    # velocity weighting
    wv = 1

    n = q0.shape[0]

    # construct first-order finite differencing matrix (velocity level)
    Kv, ev = fd1(N, n, q0, qf)

    A = Kv.T @ Kv
    b = Kv.T @ ev

    grad = wv * (A @ traj + b)

    return grad


def obs_grad_one_step(model, q, dq, ddq, field):
    ''' Compute the obstacle gradient for a single waypoint. '''
    n = q.shape[0]
    Js = model.sample_jacobians(q)
    dJs = model.sample_dJdt(q, dq)

    # Cartesian position, velocity, acceleration
    xs = model.sample_points(q)
    dxs = Js @ dq
    ddxs = Js @ ddq + dJs @ dq

    grad = np.zeros(n)
    eps = 1e-8

    num_pts = xs.shape[0]

    # numerical integration over the 5 points on the body
    for i in range(num_pts):
        x = xs[i, :]
        dx = dxs[i, :]
        ddx = ddxs[i, :]
        J = Js[i, :, :]

        obs_eps = 0.1
        c = field.cost(x, obs_eps)
        dc = field.cost_grad(x, obs_eps)

        dx_norm = np.linalg.norm(dx)
        if dx_norm < eps:
            continue

        dx_unit = dx / dx_norm
        A = np.eye(2) - np.outer(dx_unit, dx_unit)
        kappa = A @ ddx / dx_norm**2

        grad += dx_norm * J.T @ (A @ dc - c * kappa)

    return grad / num_pts


def obs_grad(model, traj, q0, qf, field, N):
    ''' Compute the obstacle gradient for the entire trajectory. '''
    n = q0.shape[0]

    # finite diff matrices
    Kv, ev = fd1(N, n, q0, qf)
    Ka, ea = fd2(N, n, q0, qf)

    # first and second derivatives of the trajectory
    dtraj = Kv @ traj + ev
    ddtraj = Ka @ traj + ea

    grad = np.zeros(N * n)

    for i in range(N):
        l = i*n
        u = (i+1)*n

        q = traj[l:u]
        dq = dtraj[l:u]
        ddq = ddtraj[l:u]

        grad[l:u] += obs_grad_one_step(model, q, dq, ddq, field)

    return grad


def main():
    np.set_printoptions(precision=3, suppress=True)

    model = models.ThreeInputModel(output_idx=[0, 1])

    circle = CircleField([3, 1], 0.5)
    floor = FloorField(0)
    field = ObstacleField([circle, floor])

    N = 20
    n = 3

    q0 = np.array([0, np.pi/4.0, -np.pi/4.0])
    qf = np.array([5, np.pi/4.0, -np.pi/4.0])
    traj0 = np.linspace(q0, qf, N + 2)[1:-1, :].flatten()
    traj = traj0

    Kv, ev = fd1(N, n, q0, qf)
    A = Kv.T @ Kv
    Ainv = np.linalg.inv(A)

    learn_rate = 0.01

    for i in range(100):
        mgrad = motion_grad(model, traj, q0, qf, N)
        ograd = obs_grad(model, traj, q0, qf, field, N)
        grad = mgrad + 10*ograd

        traj = traj - learn_rate * Ainv @ grad

    traj = np.concatenate((q0, traj, qf)).reshape((N + 2, n))

    points = np.array([model.sample_points(traj[i, :])[2:, :] for i in range(N + 1)])

    ax = plt.gca()
    ax.set_aspect('equal')
    plt.plot(points[:, 0, 0], points[:, 0, 1], 'o-', label='p0')
    plt.plot(points[:, 1, 0], points[:, 1, 1], 'o-', label='p1')
    plt.plot(points[:, 2, 0], points[:, 2, 1], 'o-', label='p2')

    ax = plt.gca()
    ax.add_patch(plt.Circle(circle.c, circle.r, color='k', fill=False))

    plt.legend()

    plt.grid()
    plt.show()



if __name__ == '__main__':
    main()
