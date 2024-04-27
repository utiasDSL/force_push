import numpy as np
from qpsolvers import solve_qp
from scipy.linalg import block_diag

from force_push import util

import time


class AdmittanceController:
    """Admittance controller to modify velocity command when force is high.

    Parameters
    ----------
    kf : float
        Gain that maps force to velocity
    force_max : float
        Only modify the commands when the force is above this magnitude.
    """

    def __init__(self, kf, force_max, vel_max=np.inf):
        self.kf = kf
        self.force_max = force_max
        self.vel_max = vel_max

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
            print("admittance!")
        if np.linalg.norm(v_cmd) > self.vel_max:
            v_cmd = self.vel_max * util.unit(v_cmd)
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
            info = obstacle.closest_point_info(r_bw_w)
            if info.deviation <= self.min_dist:
                normals.append(util.unit(info.point - r_bw_w))

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
    con_inc : float
        Increment to push angle to converge back to previous point.
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
        force_min=1,
        con_inc=0.3,
        obstacles=None,
        min_dist=0.1,
    ):
        self.speed = speed
        self.kθ = kθ
        self.ky = ky
        self.path = path

        self.ki_θ = ki_θ
        self.ki_y = ki_y

        # force thresholds
        self.force_min = force_min

        # convergence increment
        self.con_inc = con_inc
        self.div_max = np.pi

        # obstacles
        self.obstacles = obstacles if obstacles is not None else []
        self.min_dist = min_dist

        # variables
        self.reset()

    def reset(self):
        """Reset the controller to its initial state."""
        self.first_contact = False
        self.offset_int = 0
        self.θf_int = 0
        self.θp = 0
        self.converge = False

        # this is the max achieved distance from the start of the path; we
        # don't want to steer toward points that are closer to the start than
        # this
        self.dist_from_start = 0

    def update(self, position, force, dt=0):
        """Compute a new pushing velocity.

        Parameters
        ----------
        position : np.ndarray, shape (2,)
            2D position of the contact point in the world frame.
        force : np.ndarray, shape (2,)
            2D contact force expressed in the world frame.

        Returns
        -------
        : np.ndarray, shape (2,)
            The 2D pushing velocity in the world frame.
        """
        assert len(position) == 2
        assert len(force) == 2

        info = self.path.compute_closest_point_info(
            position, min_dist_from_start=self.dist_from_start
        )
        # assert np.isclose(self.dist_from_start, 0)
        self.dist_from_start = max(info.distance_from_start, self.dist_from_start)
        pathdir, offset = info.direction, info.offset
        f_norm = np.linalg.norm(force)

        # bail if we haven't ever made contact yet
        if not self.first_contact:
            if f_norm < self.force_min:
                return self.speed * pathdir
            self.first_contact = True

        θf = util.signed_angle(pathdir, util.unit(force))

        # pushing angle
        if f_norm < self.force_min:
            # if we've lost contact, try to recover by circling back
            print("converge!")

            # converge to the open loop angle (i.e., back toward the path)
            # over time
            θ_target = -self.ky * offset
            Δθ = util.wrap_to_pi(θ_target - self.θp)
            if np.abs(Δθ) > self.con_inc:
                Δθ = np.sign(Δθ) * self.con_inc
            θp = self.θp + Δθ
        else:
            # relative to pathdir
            θp = (1 + self.kθ) * θf + self.ky * offset

        self.θp = util.wrap_to_pi(θp)

        # pushing direction
        pushdir = util.rot2d(self.θp) @ pathdir

        # avoid the obstacles
        R = util.rot2d(np.pi / 2)
        for obstacle in self.obstacles:
            info = obstacle.closest_point_info(position)
            if info.deviation <= self.min_dist:
                normal = util.unit(info.point - position)
                if pushdir @ normal > 0:
                    print("correction!")
                    perp = R @ normal
                    pushdir0 = pushdir
                    pushdir = np.sign(pushdir @ perp) * perp

        return self.speed * pushdir


class DipolePushController:
    """Dipole-based pushing controller from Igarashi et al. (2010).

    See https://doi.org/10.1109/ROBOT.2010.5509483.

    This controller requires measurements of the slider's position.

    Parameters
    ----------
    speed : float, non-negative
        Linear pushing speed.
    path :
        Path to track.
    """

    def __init__(self, speed, path, lookahead_dist):
        assert speed > 0
        assert lookahead_dist >= 0

        self.speed = speed
        self.path = path
        self.lookahead_dist = lookahead_dist

        self.reset()

    def reset(self):
        """Reset the controller to its initial state."""
        self.dist_from_start = 0

    def update(self, contact_position, slider_position):
        """Compute a new pushing velocity.

        Parameters
        ----------
        contact_position : np.ndarray, shape (2,)
            2D position of the contact point in the world frame.
        slider_position : np.ndarray, shape (2,)
            2D position of the slider in the world frame.

        Returns
        -------
        : np.ndarray, shape (2,)
            The 2D pushing velocity in the world frame.
        """
        assert len(contact_position) == 2
        assert len(slider_position) == 2

        # we can use the slider position here because this controller assumes
        # it is available
        info = self.path.compute_closest_point_info(
            slider_position, min_dist_from_start=self.dist_from_start
        )
        self.dist_from_start = max(info.distance_from_start, self.dist_from_start)

        # target point to steer toward is `self.lookahead_dist` ahead of the
        # closest point on the path
        target = self.path.point_at_distance(
            info.distance_from_start + self.lookahead_dist
        )
        targetdir = util.unit(target - slider_position)

        # orthogonal axis of local path frame
        R = util.rot2d(np.pi / 2)
        orthdir = R @ targetdir

        q = contact_position - slider_position
        θ = util.signed_angle(targetdir, util.unit(q))

        # see Fig. 4 of Igarashi et al. (2010)
        pushdir = util.unit(np.cos(2 * θ) * targetdir + np.sin(2 * θ) * orthdir)

        return self.speed * pushdir
