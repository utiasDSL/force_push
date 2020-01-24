import numpy as np
import matplotlib.pyplot as plt

DURATION = 10
DT = 0.01
STEPS = int(DURATION / DT)

# force model params
Kf = 1e3

# controller gains
K_pf = 0
K_if = 0.01

fd = 1


def main():
    I = 0
    p = 0

    ps = np.zeros(STEPS + 1)
    ts = np.zeros(STEPS + 1)
    fs = np.zeros(STEPS + 1)

    for i in xrange(STEPS):
        f = Kf * p if p > 0 else 0
        df = fd - f
        I += DT * df
        p = K_pf * df + K_if * I

        fs[i+1] = f
        ps[i+1] = p
        ts[i+1] = (i+1) * DT

    plt.plot(ts, ps, label='Position')
    plt.plot(ts, fs, label='Force')
    plt.xlabel('Time (s)')
    plt.legend()
    plt.grid()
    plt.show()


if __name__ == '__main__':
    main()
