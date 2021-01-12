#!/usr/bin/env python
import numpy as np

from mm2d import util


class TrayRenderer(object):
    def __init__(self, radius, e_p_t, e_p_1, e_p_2, pe):
        self.t_p_l = np.array([-radius, 0])
        self.t_p_r = np.array([radius, 0])
        self.e_p_t = e_p_t
        self.e_p_1 = e_p_1
        self.e_p_2 = e_p_2
        self.pe = pe

    def set_state(self, pe):
        self.pe = pe

    def render(self, ax):
        θ = self.pe[2]
        R = util.rotation_matrix(θ)

        w_p_t = self.pe[:2] + R.dot(self.e_p_t)

        # sides
        p_left = w_p_t + R.dot(self.t_p_l)
        p_right = w_p_t + R.dot(self.t_p_r)

        # contact points
        p1 = self.pe[:2] + R.dot(self.e_p_1)
        p2 = self.pe[:2] + R.dot(self.e_p_2)

        self.tray, = ax.plot([p_left[0], p_right[0]], [p_left[1], p_right[1]], color='k')
        self.com, = ax.plot(w_p_t[0], w_p_t[1], 'o', color='k')
        self.contacts, = ax.plot([p1[0], p2[0]], [p1[1], p2[1]], 'o', color='r')

    def update_render(self):
        θ = self.pe[2]
        R = util.rotation_matrix(θ)

        w_p_t = self.pe[:2] + R.dot(self.e_p_t)

        # sides
        p_left = w_p_t + R.dot(self.t_p_l)
        p_right = w_p_t + R.dot(self.t_p_r)

        # contact points
        p1 = self.pe[:2] + R.dot(self.e_p_1)
        p2 = self.pe[:2] + R.dot(self.e_p_2)

        self.tray.set_xdata([p_left[0], p_right[0]])
        self.tray.set_ydata([p_left[1], p_right[1]])

        self.com.set_xdata([w_p_t[0]])
        self.com.set_ydata([w_p_t[1]])

        self.contacts.set_xdata([p1[0], p2[0]])
        self.contacts.set_ydata([p1[1], p2[1]])
