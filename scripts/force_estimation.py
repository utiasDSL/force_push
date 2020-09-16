#!/usr/bin/env python

import numpy as np
import matplotlib.pyplot as plt

import IPython


# object
ko = 100  # spring constant

fn = 10  # normal force
mu = 1    # coefficient of friction
wo = mu * fn


def main():
    N = 1000
    t = np.linspace(0, 3, N)
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

    contact_idx = np.nonzero(f)[0][0]
    moving_idx = np.nonzero(vo)[0][0]

    print(f'k = {k_est}\nw = {w_est}')

    plt.plot(t, x, label='$x_{ee}$ (m)')
    plt.plot(t, xo, label='$x_{obj}$ (m)')
    plt.plot(t, f, label='$f_{ee}$ (N)')

    plt.plot([t[contact_idx], t[contact_idx]], [0, np.max(f)], '--', color='k')
    plt.plot([t[moving_idx], t[moving_idx]], [0, np.max(f)], '--', color='k')

    plt.xlabel('Time (s)')
    plt.title('One-dimensional push simulation')
    plt.grid()
    plt.legend()
    plt.show()


if __name__ == '__main__':
    main()
