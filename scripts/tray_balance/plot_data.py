import numpy as np
import matplotlib.pyplot as plt
import IPython


def load_data(filename):
    data = np.load(filename)
    ts = data["ts"]
    P_ew_ws = data["P_ew_ws"]
    P_tw_ws = data["P_tw_ws"]
    P_ew_wd = data["P_ew_wd"]

    err_norm = np.linalg.norm(P_ew_ws[:, :2] - P_ew_wd[:2], axis=1)
    nz_idx, = np.nonzero(err_norm < 1e-2)
    stop_idx = nz_idx[0]

    return ts[:stop_idx], P_ew_ws[:stop_idx, :], P_tw_ws[:stop_idx, :], P_ew_wd


def main():
    ts1, P_ew_ws1, P_tw_ws1, P_ew_wd1 = load_data("data_good.npz")
    ts2, P_ew_ws2, P_tw_ws2, P_ew_wd2 = load_data("data_unconstrainted.npz")
    ts3, P_ew_ws3, P_tw_ws3, P_ew_wd3 = load_data("data_tall_object.npz")

    print(f"tf1 = {ts1[-1]}")
    print(f"tf2 = {ts2[-1]}")
    print(f"tf3 = {ts3[-1]}")

    fig = plt.figure(figsize=(6, 2))
    plt.rcParams.update({'font.size': 8,
                         'text.usetex': True,
                         'legend.fontsize': 8})

    # IPython.embed(V

    plt.subplot(131)
    plt.plot(P_ew_ws1[:, 0], P_ew_ws1[:, 1], label="EE")
    plt.plot(P_tw_ws1[1:, 0], P_tw_ws1[1:, 1], label="Tray")
    plt.plot(P_ew_ws1[0, 0], P_ew_ws1[0, 1], "o", color="g", label="Start")
    plt.plot(P_ew_wd1[0], P_ew_wd1[1], "o", color="r", label="End")
    plt.xlabel("x (m)")
    plt.ylabel("y (m)")

    plt.subplot(132)
    plt.plot(P_ew_ws2[:, 0], P_ew_ws2[:, 1], label="EE")
    plt.plot(P_tw_ws2[1:, 0], P_tw_ws2[1:, 1], label="Tray")
    plt.plot(P_ew_ws2[0, 0], P_ew_ws2[0, 1], "o", color="g", label="Start")
    plt.plot(P_ew_wd2[0], P_ew_wd2[1], "o", color="r", label="End")
    plt.xlabel("x (m)")
    plt.legend()

    plt.subplot(133)
    plt.plot(P_ew_ws3[:, 0], P_ew_ws3[:, 1], label="EE")
    plt.plot(P_tw_ws3[1:, 0], P_tw_ws3[1:, 1], label="Tray")
    plt.plot(P_ew_ws3[0, 0], P_ew_ws3[0, 1], "o", color="g", label="Start")
    plt.plot(P_ew_wd3[0], P_ew_wd3[1], "o", color="r", label="End")
    plt.xlabel("x (m)")

    fig.tight_layout(pad=0.5)
    fig.savefig('tray_balance.pdf')
    plt.show()


if __name__ == "__main__":
    main()
