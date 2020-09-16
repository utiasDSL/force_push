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
            # contact
            # TODO this calculation must be wrong
            # x = p_ee[i, 0]
            # rt = np.sqrt(r_o**2 - (p_ee[i, 0] - p_o[i, 0])**2)
            # y = p_o[i, 1] - rt
            # if y > p_ee[i, 1]:
            #     y = p_o[i, 1] + rt
            # p_contact = np.array([x, y])
            # d = np.linalg.norm(p_ee[i, :] - p_contact)

            # f_elastic = k_o * depth
            f_elastic_n = k_o * depth
            # tau_elastic = (p_o[i, 0] - p_ee[i, 0]) * f_elastic

            # theta = np.arctan2(p_o[i, 0] - p_ee[i, 0], p_o[i, 1] - p_ee[i, 1])
            # f_elastic_n = f_elastic * np.cos(theta)
            # f_elastic_t = f_elastic * np.sin(theta)

            if f_elastic_n > w_o:
                # direction = np.array([np.sin(theta), np.cos(theta)])
                direction = (p_o[i, :] - p_ee[i, :]) / d_ee_o
                p_o[i, :] = p_o[i, :] + (depth - w_o / k_o) * direction

            d_ee_o = np.linalg.norm(p_ee[i, :] - p_o[i, :])
            # recalculate
            # y = p_ee[i, 1]
            # x = p_o[i, 0] - np.sqrt(r_o**2 - (p_ee[i, 1] - p_o[i, 1])**2)
            # d = np.linalg.norm(p_ee[i, :] - [x, y])
            f[i] = k_o * (r_o - d_ee_o)
        else:
            f[i] = 0

        p_o[i+1, :] = p_o[i, :]
        p_ee[i+1, :] = p_ee[i, :] + v_ee * dt
        # if p_ee[i+1, 1] >= 0:
        #     break


        # if x[i] > xo[i]:
        #     # contact
        #     f_elastic = ko * (x[i] - xo[i])
        #
        #     # if elastic force overcomes friction force, the object moves
        #     # (which should result in lower elastic force next time)
        #     if f_elastic > wo:
        #         # these two formulae are equivalent:
        #         # xo[i+1] = xo[i] + (f[i+1] - wo) / ko
        #         xo[i] = x[i] - wo / ko
        #
        #     f[i] = ko * (x[i] - xo[i])
        # else:
        #     f[i] = 0
        #
        # xo[i+1] = xo[i]
        # x[i+1] = x[i] + u * dt

    # to make the plot look nice
    # f[-1] = f[-2]

    # # calculate object velocity
    # vo = np.zeros(N)
    # vo[1:] = (xo[1:] - xo[:-1]) / dt
    # moving = vo > 0
    #
    # # we only care about points when the robot and object are actually in
    # # contact
    # contact = f > 0
    #
    # # A = np.vstack((x - xo)).T
    # A = (x - xo)[contact, None]
    # b = f[contact]
    #
    # eps = 0
    # params = np.linalg.solve(A.T @ A, A.T @ b)
    #
    # k_est = params[0]
    # w_est = np.mean(f[moving])
    #
    # contact_idx = np.nonzero(f)[0][0]
    # moving_idx = np.nonzero(vo)[0][0]
    #
    # print(f'k = {k_est}\nw = {w_est}')

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
