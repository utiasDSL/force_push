#!/usr/bin/env python

import numpy as np
import matplotlib.pyplot as plt

import IPython


# object
ko = 100  # spring constant

fn = 100  # normal force
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

    for i in range(N - 1):
        x[i+1] = x[i] + u * dt

        if x[i+1] > xo[i]:
            # contact
            f[i+1] = ko * (x[i+1] - xo[i])

            # if elastic force is overcomes friction force, the object moves
            # (which should result in lower elastic force next time)
            if f[i+1] > wo:
                # these two formulae are equivalent:
                # xo[i+1] = xo[i] + (f[i+1] - wo) / ko
                xo[i+1] = x[i+1] - wo / ko
            else:
                xo[i+1] = xo[i]
        else:
            f[i+1] = 0

    plt.plot(t, x, label='x')
    plt.plot(t, f, label='f')
    plt.plot(t, xo, label='xo')
    plt.grid()
    plt.legend()
    plt.show()


if __name__ == '__main__':
    main()
