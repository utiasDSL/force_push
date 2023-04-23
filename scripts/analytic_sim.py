"""Simulation of pushing based on quasistatic model and my control law."""
import numpy as np
import matplotlib.pyplot as plt
import time
from mmpush import *

import IPython


def simulate_pushing(motion, slider, path, speed, kθ, ky, x0, duration, timestep):
    """Simulate pushing a slider with single-point contact."""
    x = x0.copy()
    xs = [x0.copy()]
    us = []
    ts = [0]

    success = True

    t = 0
    while t < duration:
        # compute required quantities
        φ = x[2]
        s = x[3]
        C_wo = rot2d(φ)
        f = C_wo.T @ x[4:]
        nc = slider.contact_normal(s)
        # TODO catch error for sliding off
        try:
            r_co_o = slider.contact_point(s)
            r_cψ_o = r_co_o - slider.cof
        except ValueError:
            print(f"Contact point left the slider at time = {t}!")
            success = False
            break
        # W = np.array([[1, 0], [0, 1], [-r_cψ_o[1], r_cψ_o[0]]])

        # term to correct deviations from desired line
        # this is simpler than pure pursuit!
        r_ow_w = x[:2]
        r_cw_w = r_ow_w + C_wo @ r_co_o
        Δ = path.compute_travel_direction(r_cw_w)
        θy = ky * path.compute_lateral_offset(r_cw_w)
        # print(path.compute_lateral_offset(r_cw_w))

        # angle-based control law
        θd = signed_angle(Δ, C_wo @ unit(f))
        θv = (1 + kθ) * θd + θy
        vp = speed * C_wo.T @ rot2d(θv) @ Δ

        # equations of motion
        try:
            Vψ, f, α = motion.solve(vp, r_cψ_o, nc)
        except ValueError as e:
            print(e)
            success = False
            break

        # solved velocity is about the CoF, move back to the centroid of the
        # object
        # fmt: off
        Vo = np.array([
            [1, 0, -slider.cof[1]],
            [0, 1,  slider.cof[0]],
            [0, 0, 1]]) @ Vψ
        # fmt: on

        # update state
        x[:2] += timestep * C_wo @ Vo[:2]
        x[2] = wrap_to_pi(x[2] + timestep * Vo[2])
        x[3] += timestep * slider.s_dot(α)
        x[4:] = C_wo @ f

        t += timestep

        us.append(vp)
        xs.append(x.copy())
        ts.append(t)

    return success, np.array(ts), np.array(xs), np.array(us)


def make_line(a, b, color="k"):
    return plt.Line2D([a[0], b[0]], [a[1], b[1]], color=color, linewidth=1)


def update_line(line, a, b):
    line.set_xdata([a[0], b[0]])
    line.set_ydata([a[1], b[1]])


def playback_simulation(xs, us, slider, path, sleep, step=1):
    plt.ion()
    fig = plt.figure()
    ax = plt.gca()

    ax.set_xlabel("x (m)")
    ax.set_ylabel("y (m)")
    ax.set_xlim([-5, 15])
    ax.set_ylim([-10, 10])
    ax.grid()

    ax.set_aspect("equal")

    x = xs[0, :]
    φ = x[2]
    s = x[3]
    C_wb = rot2d(φ)
    f = x[4:]  # world frame
    r_co_o = slider.contact_point(s)
    r_cw_w = x[:2] + C_wb @ r_co_o
    vp = C_wb @ us[0, :]

    if type(slider) is CircleSlider:
        patch = plt.Circle(
            xs[0, :2],
            radius=slider.r,
        )
    elif type(slider) is QuadSlider:
        patch = plt.Rectangle(
            xs[0, :2] - [slider.hx, slider.hy],
            width=2 * slider.hx,
            height=2 * slider.hy,
            angle=np.rad2deg(φ),
            rotation_point="center",
        )
    ax.add_patch(patch)

    push_line = make_line(r_cw_w, r_cw_w + unit(vp), color="r")
    ax.add_line(push_line)

    force_line = make_line(r_cw_w, r_cw_w + unit(f), color="b")
    ax.add_line(force_line)

    c = path.compute_closest_point(r_cw_w)
    deviation_line = make_line(r_cw_w, c, color="g")
    ax.add_line(deviation_line)

    Δ = path.compute_travel_direction(r_cw_w)
    des_line = make_line(r_cw_w, r_cw_w + Δ, color="k")
    ax.add_line(des_line)

    for i in range(1, len(us), step):
        x = xs[i, :]

        φ = x[2]
        s = x[3]
        C_wb = rot2d(φ)
        f = x[4:]  # world frame
        r_co_o = slider.contact_point(s)
        r_cw_w = x[:2] + C_wb @ r_co_o
        vp = C_wb @ us[i, :]

        Δ = path.compute_travel_direction(r_cw_w)
        c = path.compute_closest_point(r_cw_w)

        if type(slider) is CircleSlider:
            patch.center = x[:2]
        elif type(slider) is QuadSlider:
            patch.xy = x[:2] - [slider.hx, slider.hy]
            patch.angle = np.rad2deg(φ)
        update_line(push_line, r_cw_w, r_cw_w + unit(vp))
        update_line(force_line, r_cw_w, r_cw_w + unit(f))
        update_line(des_line, r_cw_w, r_cw_w + Δ)
        update_line(deviation_line, r_cw_w, c)

        fig.canvas.draw()
        fig.canvas.flush_events()
        time.sleep(sleep)
    plt.ioff()


def simulate_many(
    motion, slider, path, speed, kθ, ky, duration, timestep, y0s, θ0s, sθs, μs
):
    """Simulate many pushes, one for each value of x0."""
    x0s = []

    all_ts = []
    all_xs = []
    successes = []
    all_μs = []

    for μ in μs:
        motion.μ = μ
        for y0 in y0s:
            for θ0 in θ0s:
                for s0 in sθs:
                    x0 = np.array([0., y0, θ0, s0, 1, 0])
                    x0s.append(x0)
                    success, ts, xs, us = simulate_pushing(
                        motion, slider, path, speed, kθ, ky, x0, duration, timestep
                    )
                    successes.append(success)
                    all_ts.append(ts)
                    all_xs.append(xs)
                    all_μs.append(μ)
    return successes, all_ts, all_xs, all_μs


def test_circle_slider():
    direction = np.array([1, 0])
    path = StraightPath(direction)
    speed = 0.5

    f_max = 5
    τ_max = 2.5
    μ = 0.1

    # control gains
    kθ = 0.1
    ky = 0.1

    # x = (x, y, θ, s, f_x, f_y)
    x0 = np.array([0.0, 0.1, 0, 0, 1, 0])

    motion = QPPusherSliderMotion(f_max, τ_max, μ)
    slider = CircleSlider(0.5)

    duration = 100
    timestep = 0.01

    successes, ts, xs, us = simulate_pushing(
        motion, slider, path, speed, kθ, ky, x0, duration, timestep
    )
    playback_simulation(xs, us, slider, path, sleep=0.000)


def test_quad_slider():
    direction = np.array([1, 0])
    path = StraightPath(direction)
    speed = 0.5

    f_max = 5
    τ_max = 2.5
    μ = 0.2

    # control gains
    kθ = 0.1
    ky = 0.1

    # x = (x, y, θ, s, f_x, f_y)
    x0 = np.array([0.0, 0.4, 0, 0, 1, 0])

    motion = QPPusherSliderMotion(f_max, τ_max, μ)
    slider = QuadSlider(0.5, 0.5, cof=[0, 0])

    duration = 10
    timestep = 0.01

    successes, ts, xs, us = simulate_pushing(
        motion, slider, path, speed, kθ, ky, x0, duration, timestep
    )
    playback_simulation(xs, us, slider, path, sleep=0.001)


def main():
    # test_circle_slider()
    # return

    direction = np.array([1, 0])
    path = StraightPath(direction)

    speed = 0.5

    f_max = 5
    τ_max = 2.5
    # τ_max = 1
    μ = 0

    # control gains
    # kθ = 1.0
    # ky = 1.0
    kθ = 0.1
    ky = 0.01

    # x = (x, y, θ, s, f_x, f_y)

    motion = QPPusherSliderMotion(f_max, τ_max, μ)

    duration = 100
    timestep = 0.1

    y0s = [-0.2, 0, 0.2]
    θ0s = [-0.2, 0, 0.2]
    s0s = [-0.2, 0, 0.2]
    μs = [0, 0.25, 0.5]

    # slider = QuadSlider(0.5, 0.5, cof=[0., 0.])
    # successes, ts, xs, μs1 = simulate_many(motion, slider, path, speed, kθ, ky, duration, timestep, y0s, θ0s, s0s, μs)
    # n = len(ts)
    #
    # plt.figure()
    # for i in range(n):
    #     plt.plot(xs[i][:, 0], xs[i][:, 1], color="b", alpha=0.1)
    # plt.xlabel("x [m]")
    # plt.ylabel("y [m]")
    # plt.title("Square slider")
    # plt.grid()
    #
    # slider = QuadSlider(0.5, 0.5, cof=[-0.1, -0.1])
    # successes, ts, xs, μs2 = simulate_many(motion, slider, path, speed, kθ, ky, duration, timestep, y0s, θ0s, s0s, μs)
    # n = len(ts)
    #
    # plt.figure()
    # for i in range(n):
    #     plt.plot(xs[i][:, 0], xs[i][:, 1], color="r", alpha=0.1)
    # plt.xlabel("x [m]")
    # plt.ylabel("y [m]")
    # plt.title("Square slider, cof offset")
    # plt.grid()

    slider = CircleSlider(0.5)
    successes, ts, xs, μs3 = simulate_many(motion, slider, path, speed, kθ, ky, duration, timestep, y0s, θ0s, s0s, μs)
    n = len(ts)

    plt.figure()
    for i in range(n):
        plt.plot(xs[i][:, 0], xs[i][:, 1], color="g", alpha=0.1)
        if not successes[i]:
            print(f"circle failed with x0 = {xs[i][0, :]}, μ = {μs3[i]}")
    plt.xlabel("x [m]")
    plt.ylabel("y [m]")
    plt.title("Circle slider")
    plt.grid()

    plt.show()

    # n = 11
    # y_values = np.linspace(-1.5, 1.5, n)
    # s_values = np.linspace(-0.4, 0.4, n)
    # convergence = np.zeros((n, n))
    # for i, y in enumerate(y_values):
    #     for j, s in enumerate(s_values):
    #         x0 = np.array([0, y, 0, s, 1, 0])
    #         ts, xs = simulate_pushing(
    #             motion, slider, path, speed, kθ, ky, x0, duration, timestep
    #         )
    #         print(f"s_max = {np.max(xs[:, 3])}")
    #         if np.max(xs[:, 3]) > 0.5:
    #             continue
    #         y_final = xs[-1, 1]
    #         θf_final = np.arctan2(xs[-1, 5], xs[-1, 4])
    #         print(f"y = {y_final}")
    #         print(f"θf = {θf_final}")
    #         # IPython.embed()
    #         # return
    #         if np.abs(y_final) < 1e-3 and np.abs(θf_final) < 1e-3:
    #             convergence[i, j] = 1
    #
    # # TODO can we do this as a non-3D grid?
    # x, y = np.meshgrid(y_values, s_values)
    # fig = plt.figure()
    # ax = fig.add_subplot(projection="3d")
    # ax.scatter(x, y, convergence)
    # plt.grid()
    # plt.show()


if __name__ == "__main__":
    main()
