"""Basic functions for simulating and viewing the pusher-slider system."""
import numpy as np
import matplotlib.pyplot as plt
import time

from force_push import util
from force_push.slider import CircleSlider, QuadSlider


def simulate_pushing2(motion, slider, path, speed, kθ, ky, x0, duration, timestep, ki_θ=0, ki_y=0):
    """Simulate pushing a slider with single-point contact. [EXPERIMENTAL VERSION]"""
    x = x0.copy()
    xs = [x0.copy()]
    us = []
    ts = [0]

    success = True
    yc_int = 0
    θd_int = 0

    t = 0
    while t < duration:
        # compute required quantities
        φ = x[2]
        s = x[3]
        C_wo = util.rot2d(φ)
        f_w = x[4:]
        nc = slider.contact_normal(s)

        try:
            r_co_o = slider.contact_point(s)
            r_cψ_o = r_co_o - slider.cof
        except ValueError:
            print(f"Contact point left the slider at time = {t}!")
            success = False
            break

        # term to correct deviations from desired line
        # this is simpler than pure pursuit!
        r_ow_w = x[:2]
        r_cw_w = r_ow_w + C_wo @ r_co_o
        Δ = path.compute_travel_direction(r_cw_w)
        yc = path.compute_lateral_offset(r_cw_w)
        yc_int += timestep * yc
        θy = ky * yc + ki_y * yc_int

        # angle-based control law
        θd = util.signed_angle(Δ, util.unit(f_w))
        θd_int += timestep * θd
        θp = (1 + kθ) * θd + ki_θ * θd_int + θy

        # if np.abs(θp) > 0.5 * np.pi:
        #     print("Pusher is going backward.")

        vp_w = speed * util.rot2d(θp) @ Δ
        vp = C_wo.T @ vp_w

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
        x[2] = util.wrap_to_pi(x[2] + timestep * Vo[2])
        x[3] += timestep * slider.s_dot(α)
        x[4:] = C_wo @ f

        t += timestep

        us.append(vp)
        xs.append(x.copy())
        ts.append(t)

    return success, np.array(ts), np.array(xs), np.array(us)


def simulate_pushing(motion, slider, path, speed, kθ, ky, x0, duration, timestep):
    """Simulate pushing a slider with single-point contact."""
    x = x0.copy()
    xs = [x0.copy()]
    us = []
    ts = [0]

    success = True
    # θp = 0

    t = 0
    while t < duration:
        # compute required quantities
        φ = x[2]
        s = x[3]
        C_wo = util.rot2d(φ)
        f_w = x[4:]
        nc = slider.contact_normal(s)

        try:
            r_co_o = slider.contact_point(s)
            r_cψ_o = r_co_o - slider.cof
        except ValueError:
            print(f"Contact point left the slider at time = {t}!")
            success = False
            break

        # term to correct deviations from desired line
        # this is simpler than pure pursuit!
        r_ow_w = x[:2]
        r_cw_w = r_ow_w + C_wo @ r_co_o
        Δ = path.compute_travel_direction(r_cw_w)
        θy = ky * path.compute_lateral_offset(r_cw_w)

        # angle-based control law
        θd = util.signed_angle(Δ, util.unit(f_w))
        # θp_last = θp
        θp = (1 + kθ) * θd + θy

        # r = 0.01
        # if θp - θp_last > r:
        #     θp = θp_last + r
        # elif θp - θp_last < -r:
        #     θp = θp_last - r
        if np.abs(θp) > 0.5 * np.pi:
            print("Pusher is going backward.")

        vp_w = speed * util.rot2d(θp) @ Δ

        # avoid going backward (this is pathological anyway)
        # if vp_w[0] < 0:
        #     vp_w[0] = 0
        #     vp_w = speed * unit(vp_w)

        vp = C_wo.T @ vp_w

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
        x[2] = util.wrap_to_pi(x[2] + timestep * Vo[2])
        x[3] += timestep * slider.s_dot(α)
        x[4:] = C_wo @ f

        t += timestep

        us.append(vp)
        xs.append(x.copy())
        ts.append(t)

    return success, np.array(ts), np.array(xs), np.array(us)


def make_line(a, b, color="k"):
    return plt.Line2D([a[0], b[0]], [a[1], b[1]], color=color, linewidth=1, solid_capstyle="round")


def update_line(line, a, b):
    line.set_xdata([a[0], b[0]])
    line.set_ydata([a[1], b[1]])


def playback_simulation(xs, us, slider, path, sleep, step=1):
    """Playback a trajectory of pushing."""
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
    C_wb = util.rot2d(φ)
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

    push_line = make_line(r_cw_w, r_cw_w + util.unit(vp), color="r")
    ax.add_line(push_line)

    force_line = make_line(r_cw_w, r_cw_w + util.unit(f), color="b")
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
        C_wb = util.rot2d(φ)
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
        update_line(push_line, r_cw_w, r_cw_w + util.unit(vp))
        update_line(force_line, r_cw_w, r_cw_w + util.unit(f))
        update_line(des_line, r_cw_w, r_cw_w + Δ)
        update_line(deviation_line, r_cw_w, c)

        fig.canvas.draw()
        fig.canvas.flush_events()
        time.sleep(sleep)
    plt.ioff()
