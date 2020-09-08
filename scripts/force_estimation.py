#!/usr/bin/env python

import numpy as np
import matplotlib.pyplot as plt

import IPython


# object
ko = 200  # spring constant

fn = 50  # normal force
mu = 1    # coefficient of friction
wo = mu * fn


def main():
    N = 1000
    t = np.linspace(0, 10, N)
    dt = t[1] - t[0]

    vd = 1  # desired velocity

    x = np.zeros(N)  # position of EE
    f = np.zeros(N)  # measured force
    xo = np.ones(N)  # position of object

    u = vd  # input velocity for robot

    for i in range(N-1):
        if x[i] > xo[i]:
            # contact
            f_elastic = ko * (x[i] - xo[i])

            # if elastic force overcomes friction force, the object moves
            # (which should result in lower elastic force next time)
            if f_elastic > wo:
                # these two formulae are equivalent:
                # xo[i+1] = xo[i] + (f[i+1] - wo) / ko
                xo[i] = x[i] - wo / ko

            f[i] = ko * (x[i] - xo[i])
        else:
            f[i] = 0

        xo[i+1] = xo[i]
        x[i+1] = x[i] + u * dt

    # to make the plot look nice
    f[-1] = f[-2]

    # calculate object velocity
    vo = np.zeros(N)
    vo[1:] = (xo[1:] - xo[:-1]) / dt
    moving = vo > 0

    # we only care about points when the robot and object are actually in
    # contact
    contact = f > 0

    # A = np.vstack((x - xo)).T
    A = (x - xo)[contact, None]
    b = f[contact]

    eps = 0
    params = np.linalg.solve(A.T @ A, A.T @ b)

    k_est = params[0]
    w_est = np.mean(f[moving])

    print(f'k = {k_est}\nw = {w_est}')

    IPython.embed()

    plt.plot(t, x, label='x')
    plt.plot(t, f, label='f')
    plt.plot(t, xo, label='xo')
    plt.plot(t, vo, label='vo')
    plt.grid()
    plt.legend()
    plt.show()


if __name__ == '__main__':
    main()
