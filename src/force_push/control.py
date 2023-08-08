import numpy as np
from qpsolvers import solve_qp

from force_push import util


class RobotController:
    """Controller for the mobile base.

    Designed to achieve commanded EE linear velocity while using base rotation
    to avoid obstacles and otherwise align the robot with the path.
    """

    def __init__(self, r_cb_b, lb, ub, obstacles=None, min_dist=0.75, solver="proxqp"):
        """Initialize the controller.

        Parameters:
            r_cb_b: 2D position of the contact point w.r.t. the base frame origin
            lb: Lower bound on velocity (v_x, v_y, ω)
            ub: Upper bound on velocity
            obstacles: list of LineSegments to be avoided (optional)
            min_dist: Minimum distance to maintain from obstacles.
            solver: the QP solver to use (default: 'proxqp')
        """
        self.r_cb_b = r_cb_b
        self.lb = lb
        self.ub = ub
        self.obstacles = obstacles if obstacles is not None else []
        self.min_dist = min_dist
        self.solver = solver

    def _compute_obstacle_constraint(self, r_bw_w):
        # no obstacles, no constraint
        n = len(self.obstacles)
        if n == 0:
            return None, None

        # find nearby obstacles
        normals = []
        for obstacle in self.obstacles:
            closest, dist = obstacle.closest_point_and_distance(r_bw_w)
            if dist <= self.min_dist:
                normals.append(util.unit(closest - r_bw_w))

        # no nearby obstacles
        if len(normals) == 0:
            return None, None

        G = np.atleast_2d(normals)
        G = np.hstack((G, np.zeros((G.shape[0], 1))))
        h = np.zeros(G.shape[0])
        return G, h

    def update(self, r_bw_w, C_wb, V_ee_d):
        """Compute new controller input u.

        Parameters:
            r_bw_w: 2D position of base in the world frame
            C_wb: rotation matrix from base frame to world frame
            V_ee_d: desired EE velocity (v_x, v_y, ω)
        """
        S = np.array([[0, -1], [1, 0]])
        J = np.hstack((np.eye(2), (S @ C_wb @ self.r_cb_b)[:, None]))

        P = np.diag([0.1, 0.1, 1])
        q = np.array([0, 0, -V_ee_d[2]])
        G, h = self._compute_obstacle_constraint(r_bw_w)
        A = J
        b = V_ee_d[:2]

        u = solve_qp(
            P=P, q=q, A=A, b=b, G=G, h=h, lb=self.lb, ub=self.ub, solver=self.solver
        )
        return u


class PushController:
    """Task-space force angle-based pushing controller."""

    def __init__(
        self,
        speed,
        kθ,
        ky,
        path,
        ki_θ=0,
        ki_y=0,
        corridor_radius=np.inf,
        force_min=1,
        force_max=50,
        con_inc=0.3,
        div_inc=0.3,
        obstacles=None,
        min_dist=0.1,
    ):
        """Force-based pushing controller.

        Parameters:
            speed: linear pushing speed
            kθ: gain for stable pushing
            ky: gain for path tracking
            path: path to track
            ki_θ: integral gain for stable pushing
            ki_y: integral gain for path tracking
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

        # distance from center of corridor the edges
        # if infinite, then the corridor is just open space
        # to be used with hallways (won't work if there aren't actually walls
        # present)
        self.corridor_radius = corridor_radius

        # force thresholds
        self.force_min = force_min
        self.force_max = force_max

        # convergence and divergence increment
        self.con_inc = con_inc
        self.div_inc = div_inc

        # obstacles
        self.obstacles = obstacles if obstacles is not None else []
        self.min_dist = min_dist

        # variables
        self.reset()
        # self.first_contact = False
        # self.yc_int = 0
        # self.θd_int = 0
        # self.θp = 0
        # self.inc_sign = 1
        # self.diverge = False

    def reset(self):
        """Reset the controller to its initial state."""
        self.first_contact = False
        self.yc_int = 0
        self.θd_int = 0
        self.θp = 0
        self.inc_sign = 1
        self.diverge = False

    def update(self, position, force, dt=0):
        """Compute a new pushing velocity based on contact position and force
        (all expressed in the world frame)."""
        assert len(position) == 2
        assert len(force) == 2

        pathdir, yc = self.path.compute_direction_and_offset(position)
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

        speed = self.speed

        print(f"f_norm = {f_norm}")

        # pushing angle
        if f_norm < self.force_min:
            # if we've lost contact, try to recover by circling back
            print("converge!")
            θp = self.θp - self.inc_sign * self.con_inc
            self.diverge = False
        elif f_norm > self.force_max or self.diverge:
            # diverge from the path if force is too high
            print("diverge!")
            # θp = self.θp + self.inc_sign * self.div_inc
            if not self.diverge:
                # rev = self.θp + self.inc_sign * np.pi
                self.div_init = self.θp
                # self.div_min = min(self.θp, rev)
                # self.div_max = max(self.θp, rev)
            # # θp = self.θp + np.sign(θd) * self.div_inc
            # # TODO may not play well with wrap to pi
            # θp = self.θp + self.inc_sign * self.div_inc
            # θp = min(max(self.div_min, θp), self.div_max)

            θp = self.θp + self.inc_sign * self.div_inc

            # limit divergence to no more than 180 degrees (i.e., going backward)
            div_delta = util.wrap_to_pi(θp - self.div_init)
            if div_delta > np.pi:
                θp = self.div_init + np.pi

            # if self.diverge:
            #     # already diverging: keep doing what we're doing
            #     θp = self.θp
            # else:
            #     # θp = self.θp + np.sign(θd) * self.div_inc
            #     θp = self.θp + np.sign(θd) * 0.75 * np.pi
            self.diverge = True
        else:
            # self.diverge = False
            θp = (
                (1 + self.kθ) * θd
                + self.ky * yc
                + self.ki_θ * self.θd_int
                + self.ki_y * self.yc_int
            )
            self.inc_sign = np.sign(θd)

        self.θp = util.wrap_to_pi(θp)
        print(f"θp = {θp}")

        # pushing velocity
        pushdir = util.rot2d(self.θp) @ pathdir

        # avoid the walls of the corridor
        # if np.abs(yc) >= self.corridor_radius:
        #     R = util.rot2d(np.pi / 2)
        #     perp = R @ pathdir
        #     print("correction!")
        #     if off > 0 and perp @ pushdir > 0:
        #         pushdir = util.unit(pushdir - (perp @ pushdir) * perp)
        #     elif off < 0 and perp @ pushdir < 0:
        #         pushdir = util.unit(pushdir - (perp @ pushdir) * perp)

        R = util.rot2d(np.pi / 2)
        for obstacle in self.obstacles:
            closest, dist = obstacle.closest_point_and_distance(position)
            if dist <= self.min_dist:
                print("correction!")
                normal = util.unit(closest - position)
                if pushdir @ normal > 0:
                    perp = R @ normal
                    pushdir = np.sign(pushdir @ perp) * perp

        return speed * pushdir
