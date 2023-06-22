import numpy as np

from force_push import util


class Controller:
    """Force angle-based pushing controller."""

    def __init__(self, speed, kθ, ky, path, ki_θ=0, ki_y=0, lookahead=0):
        self.speed = speed
        self.kθ = kθ
        self.ky = ky
        self.path = path

        self.ki_θ = ki_θ
        self.ki_y = ki_y
        self.lookahead = lookahead

        self.yc_int = 0
        self.θd_int = 0

        # force thresholds
        self.ft_max = 10
        self.ft_min = 1

        self.k_div = 0.01

        self.first_contact = False
        self.θp_last = 0
        self.inc_sign = 1
        self.inc = 0.1

    def update(self, position, force, dt=0):
        """Compute a new pushing velocity based on contact position and force
        (all expressed in the world frame)."""
        assert len(position) == 2
        assert len(force) == 2

        pathdir, yc = self.path.compute_direction_and_offset(
            position, lookahead=self.lookahead
        )
        f_norm = np.linalg.norm(force)

        # Bail if we haven't ever made contact yet
        if not self.first_contact:
            if f_norm < self.ft_min:
                return self.speed * pathdir
            self.first_contact = True

        θd = util.signed_angle(pathdir, util.unit(force))

        # integrators
        self.yc_int += dt * yc
        self.θd_int += dt * θd

        # term to diverge if force magnitude is above ft_max
        f_div = max(f_norm - self.ft_max, 0)

        # pushing angle
        if f_norm < self.ft_min:
            # if we've lost contact, try to recover by circling back
            # TODO I'd like to actually seek the previous contact point
            θp = self.θp_last + self.inc_sign * self.inc
        else:
            θp = (
                (1 + self.kθ + self.k_div * f_div) * θd
                + self.ky * yc
                + self.ki_θ * self.θd_int
                + self.ki_y * self.yc_int
            )
            direction = np.array([np.cos(θp), np.sin(θp)])
            self.inc_sign = np.sign(util.signed_angle(pathdir, direction))
        self.θp_last = θp

        # pushing velocity
        return self.speed * util.rot2d(θp) @ pathdir
