import numpy as np
import scipy.integrate
import matplotlib.pyplot as plt
import IPython

# L = 1.0
# L2 = L * 0.5
#
#
# def dist(y, x):
#     return np.sqrt(x**2 + y**2)
#
#
# # numerical answer
# v1, _ = scipy.integrate.dblquad(dist, -L2, L2, -L2, L2)
#
# # (approximate) analytic answer
# v2 = L**3 * 1.14779 / 3
#
# print(f'Numerical answer = {v1}\n Analytic answer = {v2}')

class Circle:
    def __init__(self, c, r):
        self.c = np.array(c)
        self.r = r



# object
# ko = 100  # spring constant
#
# fn = 10  # normal force
# mu = 1    # coefficient of friction
# wo = mu * fn


def main():
    N = 1000
    t = np.linspace(0, 3, N)
    dt = t[1] - t[0]

    f = np.zeros(N)  # measured force

    # object is a circle
    r_o = 0.5
    p_o = np.zeros((N, 2))
    th_o = np.zeros(N)

    k_o = 1000  # spring constant
    fn = 100
    mu = 1
    w_o = fn * mu
    tau_o = 2 * r_o * w_o / 3

    # EE position and velocity
    p_ee = np.zeros((N, 2))
    p_ee[0, :] = np.array([-0.1, -0.5])
    v_ee = 1 * np.array([0, 1])

    for i in range(N-1):
        d_ee_o = np.linalg.norm(p_ee[i, :] - p_o[i, :])

        if d_ee_o < r_o:
            depth = r_o - d_ee_o
            # normal force is due to elasticity
            f_elastic_n = k_o * depth

            # tangential force is from friction
            f_elastic_t = mu * f_elastic_n

            # tangential force results in a moment
            tau_elastic = f_elastic_t * d_ee_o

            if f_elastic_n > w_o:
                direction = (p_o[i, :] - p_ee[i, :]) / d_ee_o
                p_o[i, :] = p_o[i, :] + (depth - w_o / k_o) * direction

            if tau_elastic > tau_o:
                # need to rotate in response to tangential force, when it
                # overcomes friction
                pass

            # recalculate now that object has moved
            d_ee_o = np.linalg.norm(p_ee[i, :] - p_o[i, :])
            f[i] = k_o * (r_o - d_ee_o)
        else:
            f[i] = 0

        p_o[i+1, :] = p_o[i, :]
        p_ee[i+1, :] = p_ee[i, :] + v_ee * dt

    plt.figure(1)
    plt.plot(t, f)
    plt.grid()

    plt.figure(2)
    plt.plot(p_ee[:, 0], p_ee[:, 1], label='EE')
    plt.plot(p_o[:, 0], p_o[:, 1], label='Object')
    plt.legend()
    plt.grid()
    plt.show()

    IPython.embed()


if __name__ == '__main__':
    main()
