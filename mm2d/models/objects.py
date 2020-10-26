import numpy as np


class InvertedPendulum:
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
