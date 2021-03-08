#!/usr/bin/env python
import numpy as np

from mm2d import util


class TrayRenderer(object):
    def __init__(self, radius, p_te_e, p_c1e_e, p_c2e_e, P_ew_w):
        self.p_lt_t = np.array([-radius, 0])
        self.p_rt_t = np.array([radius, 0])
        self.p_te_e = p_te_e
        self.p_c1e_e = p_c1e_e
        self.p_c2e_e = p_c2e_e
        self.P_ew_w = P_ew_w

    def set_state(self, P_ew_w):
        self.P_ew_w = P_ew_w

    def render(self, ax):
        p_ew_w, θ_ew = self.P_ew_w[:2], self.P_ew_w[2]
        R_we = util.rotation_matrix(θ_ew)

        p_tw_w = p_ew_w + R_we @ self.p_te_e

        # sides
        p_lw_w = p_tw_w + R_we @ self.p_lt_t
        p_rw_w = p_tw_w + R_we @ self.p_rt_t

        # contact points
        p_c1w_w = p_ew_w + R_we @ self.p_c1e_e
        p_c2w_w = p_ew_w + R_we @ self.p_c2e_e

        self.tray, = ax.plot([p_lw_w[0], p_rw_w[0]], [p_lw_w[1], p_rw_w[1]], color='k')
        self.com, = ax.plot(p_tw_w[0], p_tw_w[1], 'o', color='k')
        self.contacts, = ax.plot([p_c1w_w[0], p_c2w_w[0]], [p_c1w_w[1], p_c2w_w[1]], 'o', color='r')

    def update_render(self):
        p_ew_w, θ_ew = self.P_ew_w[:2], self.P_ew_w[2]
        R_we = util.rotation_matrix(θ_ew)

        p_tw_w = p_ew_w + R_we @ self.p_te_e

        # sides
        p_lw_w = p_tw_w + R_we @ self.p_lt_t
        p_rw_w = p_tw_w + R_we @ self.p_rt_t

        # contact points
        p_c1w_w = p_ew_w + R_we @ self.p_c1e_e
        p_c2w_w = p_ew_w + R_we @ self.p_c2e_e

        self.tray.set_xdata([p_lw_w[0], p_rw_w[0]])
        self.tray.set_ydata([p_lw_w[1], p_rw_w[1]])

        self.com.set_xdata([p_tw_w[0]])
        self.com.set_ydata([p_tw_w[1]])

        self.contacts.set_xdata([p_c1w_w[0], p_c2w_w[0]])
        self.contacts.set_ydata([p_c1w_w[1], p_c2w_w[1]])
