import numpy as np

from force_push import util


class Controller:
    """Force angle-based pushing controller."""

    def __init__(
        self,
        speed,
        kθ,
        ky,
        path,
        ki_θ=0,
        ki_y=0,
        lookahead=0,
        corridor_radius=np.inf,
        force_min=1,
        force_max=50,
        con_inc=0.3,
        div_inc=0.3,
    ):
        """Force-based pushing controller.

        Parameters:
            speed: linear pushing speed
            kθ: gain for stable pushing
            ky: gain for path tracking
            path: path to track
            ki_θ: integral gain for stable pushing
            ki_y: integral gain for path tracking
            lookahead: distance to lookahead on the path when determining
                direction and offset
            force_min: contact requires a force of at least this much
            force_max: diverge if force exceeds this much
            con_inc: increment to push angle to converge back to previous point
            div_inc: increment to push angle to diverge when force is too high
        """
        self.speed = speed
        self.kθ = kθ
        self.ky = ky
        self.path = path

        self.ki_θ = ki_θ
        self.ki_y = ki_y
        self.lookahead = lookahead

        # distance from center of corridor the edges
        # if infinite, then the corridor is just open space
        # to be used with hallways (won't work if there aren't actually walls
        # present)
        self.corridor_radius = corridor_radius

        # force thresholds
        # self.force_max = 10
        self.force_min = force_min
        self.force_max = force_max

        # convergence and divergence increment
        self.con_inc = con_inc
        self.div_inc = div_inc

        # variables
        self.first_contact = False
        self.yc_int = 0
        self.θd_int = 0
        self.θp = 0
        self.inc_sign = 1

    def reset(self):
        """Reset the controller to its initial state."""
        self.first_contact = False
        self.yc_int = 0
        self.θd_int = 0
        self.θp = 0
        self.inc_sign = 1

    def update(self, position, force, dt=0):
        """Compute a new pushing velocity based on contact position and force
        (all expressed in the world frame)."""
        assert len(position) == 2
        assert len(force) == 2

        pathdir, yc = self.path.compute_direction_and_offset(
            position, lookahead=self.lookahead
        )
        f_norm = np.linalg.norm(force)

        # bail if we haven't ever made contact yet
        if not self.first_contact:
            if f_norm < self.force_min:
                return self.speed * pathdir
            self.first_contact = True

        θd = util.signed_angle(pathdir, util.unit(force))

        # integrators
        self.yc_int += dt * yc
        self.θd_int += dt * θd

        # pushing angle
        if f_norm < self.force_min:
            # if we've lost contact, try to recover by circling back
            θp = self.θp - self.inc_sign * self.con_inc
        elif f_norm > self.force_max:
            # diverge from the path if force is too high

            # TODO
            # θp = self.θp + self.inc_sign * self.inc
            θp = self.θp + np.sign(θd) * self.div_inc
        else:
            θp = (
                (1 + self.kθ) * θd
                + self.ky * yc
                + self.ki_θ * self.θd_int
                + self.ki_y * self.yc_int
            )
            self.inc_sign = np.sign(θd)
        self.θp = util.wrap_to_pi(θp)

        # pushing velocity
        pushdir = util.rot2d(self.θp) @ pathdir

        # avoid the walls of the corridor
        tangent, off = self.path.compute_direction_and_offset(position, lookahead=0)
        if np.abs(off) >= self.corridor_radius:
            R = util.rot2d(np.pi / 2)
            perp = R @ tangent
            print("correction!")
            if off > 0 and perp @ pushdir > 0:
                pushdir = util.unit(pushdir - (perp @ pushdir) * perp)
            elif off < 0 and perp @ pushdir < 0:
                pushdir = util.unit(pushdir - (perp @ pushdir) * perp)

        return self.speed * pushdir
