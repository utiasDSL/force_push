#!/usr/bin/env python

import numpy as np
import matplotlib.pyplot as plt


# object
xo = 1
ko = 100


def main():
    N = 1000
    t = np.linspace(0, 10, N)
    dt = t[1] - t[0]

    vd = 1
    xd = vd * t
    fd = 1

    x = np.zeros(N)
    f = np.zeros(N)
    u = vd

    kv = 1
    kp = 1
    kf = 1

    for i in range(N - 1):
        x[i + 1] = x[i] + u * dt

        if x[i+1] > xo:
            f[i + 1] = ko * (x[i + 1] - xo)
        else:
            f[i + 1] = 0

        df = fd - f[i+1]
        dx = xd[i+1] - x[i+1]

        u = vd + (kp*dx + kf*df) / kv

    plt.plot(t, x, label='x')
    plt.plot(t, f, label='f')
    plt.grid()
    plt.legend()
    plt.show()


if __name__ == '__main__':
    main()
