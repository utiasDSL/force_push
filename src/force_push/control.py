import numpy as np
from qpsolvers import solve_qp
from scipy.linalg import block_diag

from force_push import util


class AdmittanceController:
    """Admittance controller to modify velocity command when force is high.

    Parameters
    ----------
    kf : float
        Gain that maps force to velocity
    force_max : float
        Only modify the commands when the force is above this magnitude.
    """

    def __init__(self, kf, force_max):
        self.kf = kf
        self.force_max = force_max

    def update(self, force, v_cmd):
        """Compute a new control command.

        Parameters
        ----------
        force :
            The sensed contact force.
        v_cmd :
            The nominal velocity command.

        Returns
        -------
        :
            The velocity command modified to comply with the sensed force.
        """
        f_norm = np.linalg.norm(force)
        if f_norm > self.force_max:
            vf = -self.kf * (f_norm - self.force_max) * util.unit(force)
            v_cmd = v_cmd + vf
        return v_cmd


class RobotController:
    """Controller for the mobile base.

    Designed to achieve commanded EE linear velocity while using base rotation
    to avoid obstacles and otherwise align the robot with the path.

    Parameters
    ----------
    r_cb_b :
        2D position of the contact point w.r.t. the base frame origin.
    lb :
        Lower bound on velocity (v_x, v_y, ω).
    ub :
        Upper bound on velocity.
    obstacles :
        List of LineSegments to be avoided.
    min_dist : float
        Minimum distance to maintain from obstacles.
    vel_weight : float
        Weight on velocity norm in controller.
    acc_weight : float
        Weight on acceleration norm in the controller.
    solver : str
        The QP solver to use.
    """

    def __init__(
        self,
        r_cb_b,
        lb,
        ub,
        vel_weight=1,
        acc_weight=0,
        obstacles=None,
        min_dist=0.75,
        solver="proxqp",
    ):
        self.r_cb_b = r_cb_b
        self.lb = np.append(lb, 0)
        self.ub = np.append(ub, 1)
        self.obstacles = obstacles if obstacles is not None else []
        self.min_dist = min_dist
        self.wv = vel_weight
        self.wa = acc_weight
        self.solver = solver

    def _compute_obstacle_constraint(self, r_bw_w):
        """Build the matrices requires to enforce obstacle avoidance."""
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
        G = np.hstack((G, np.zeros((G.shape[0], 2))))
        h = np.zeros(G.shape[0])
        return G, h

    def update(self, r_bw_w, C_wb, V_ee_d, u_last=None):
        """Compute new controller input.

        Parameters
        ----------
        r_bw_w :
            2D position of base in the world frame.
        C_wb :
            2x2 rotation matrix from base frame to world frame.
        V_ee_d :
            Desired EE velocity (v_x, v_y, ω).

        Returns
        -------
        :
            Joint velocity command.
        """
        if u_last is None:
            u_last = np.zeros(3)

        S = np.array([[0, -1], [1, 0]])
        J = np.hstack((np.eye(2), (S @ C_wb @ self.r_cb_b)[:, None]))

        # acceleration cost
        Pa = self.wa * np.diag([1, 1, 1, 0])
        qa = self.wa * np.append(-u_last, 0)

        # velocity cost
        Pv = self.wv * np.diag([1, 1, 1, 0])
        qv = self.wv * np.array([0, 0, -V_ee_d[2], 0])

        # tracking cost (if we want it to be a cost rather than constraint)
        # Pt = block_diag(J.T @ J, [[0]])
        # qt = np.append(-V_ee_d[:2] @ J, 0)

        # speed scaling cost
        Pα = np.diag([0, 0, 0, 1])
        qα = np.array([0, 0, 0, -1])

        P = Pa + Pv + Pα
        q = qa + qv + qα
        G, h = self._compute_obstacle_constraint(r_bw_w)

        # with speed scaling
        # A = np.hstack((J, -V_ee_d[:2, None]))
        # b = np.zeros(2)

        # without speed scaling
        A = np.hstack((J, np.zeros((2, 1))))
        b = V_ee_d[:2]

        x = solve_qp(
            P=P, q=q, A=A, b=b, G=G, h=h, lb=self.lb, ub=self.ub, solver=self.solver
        )
        u, α = x[:3], x[3]
        # print(f"α = {α}")
        return u


class PushController:
    """Task-space force angle-based pushing controller.

    Parameters
    ----------
    speed : float
        Linear pushing speed.
    kθ : float
        Gain for stable pushing.
    ky : float
        Gain for path tracking.
    path :
        Path to track.
    ki_θ : float
        Integral gain for stable pushing.
    ki_y : float
        Integral gain for path tracking.
    force_min : float
        Contact requires a force of at least this much.
    force_max : float
        DEPRECATED. Diverge if force exceeds this much.
    con_inc : float
        Increment to push angle to converge back to previous point.
    div_inc : float
        DEPRECATED. Increment to push angle to diverge when force is too high.
    obstacles :
        List of obstacles to avoid with the EE.
    min_dist :
        Minimum distance to maintain from the obstacles.
    """

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
        self.div_max = np.pi

        # obstacles
        self.obstacles = obstacles if obstacles is not None else []
        self.min_dist = min_dist

        # variables
        self.reset()

    def reset(self):
        """Reset the controller to its initial state."""
        self.first_contact = False
        self.yc_int = 0
        self.θd_int = 0
        self.θp = 0
        self.inc_sign = 1
        self.diverge = False
        self.converge = False

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

        # pushing angle
        if f_norm < self.force_min:
            # if we've lost contact, try to recover by circling back
            print("converge!")

            if not self.converge:
                self.con_init = self.θp
                self.con = 0

            # we keep the "convergence angle" separate from θp so we can
            # directly check its magnitude without worrying about wrapping to
            # pi and all that
            self.con += self.con_inc
            if self.con > self.div_max:
                self.con = self.div_max
            θp = self.con_init - self.inc_sign * self.con

            # θp = self.θp - self.inc_sign * self.con_inc
            #
            # # limit convergence to no more than div_max
            # con_delta = util.wrap_to_pi(self.con_init - θp)
            # if abs(con_delta) > self.div_max:
            #     θp = self.con_init - self.div_max

            self.diverge = False
            self.converge = True
        elif f_norm > self.force_max or self.diverge:
            # diverge from the path if force is too high
            print("diverge!")
            if not self.diverge:
                self.div_init = self.θp
                self.div = 0
            self.div += self.div_inc
            if self.div > self.div_max:
                self.div = self.div_max
            θp = self.div_init + self.inc_sign * self.div

            # θp = self.θp + self.inc_sign * self.div_inc
            #
            # # limit divergence to no more than div_max
            # div_delta = util.wrap_to_pi(θp - self.div_init)
            # if abs(div_delta) > self.div_max:
            #     θp = self.div_init + self.div_max

            # if self.diverge:
            #     # already diverging: keep doing what we're doing
            #     θp = self.θp
            # else:
            #     # θp = self.θp + np.sign(θd) * self.div_inc
            #     θp = self.θp + self.inc_sign * 0.5 * np.pi
            self.diverge = True
            self.converge = False
        else:
            θp = (
                (1 + self.kθ) * θd
                + self.ky * yc
                + self.ki_θ * self.θd_int
                + self.ki_y * self.yc_int
            )
            self.inc_sign = np.sign(θp)  # NOTE
            self.converge = False

        self.θp = util.wrap_to_pi(θp)

        # pushing velocity
        pushdir = util.rot2d(self.θp) @ pathdir

        # avoid the obstacles
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
