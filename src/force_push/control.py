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

    def update(self, position, force, dt=0):
        """Compute a new pushing velocity based on contact position and force
        (all expressed in the world frame)."""
        assert len(position) == 2
        assert len(force) == 2

        Δ, yc = self.path.compute_direction_and_offset(
            position, lookahead=self.lookahead
        )
        θd = util.signed_angle(Δ, util.unit(force))

        # integrators
        self.yc_int += dt * yc
        self.θd_int += dt * θd

        # pushing angle
        θp = (
            (1 + self.kθ) * θd
            + self.ky * yc
            + self.ki_θ * self.θd_int
            + self.ki_y * self.yc_int
        )

        # pushing velocity
        return self.speed * util.rot2d(θp) @ Δ
