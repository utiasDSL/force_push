import numpy as np
import matplotlib.pyplot as plt


# class Wall(object):
#     def __init__(self, x):
#         self.x = x  # location
#
#     def apply_force(self, p, dp):
#         ''' Apply force based on end effector position p. '''
#         if p[0] > self.x:
#             dx = p[0] - self.x
#             f = np.array([SPRING_CONST * dx + DAMPER_CONST * dp[0], 0])
#         else:
#             f = np.zeros(2)
#         return f + F_SIGMA * np.random.randn(f.shape[0]) + F_MEAN
#
#     def draw(self, ax):
#         ax.plot([self.x, self.x], [0, 2], color='k')


# def unit(a):
#     return a / np.linalg.norm(a)



class Circle(object):
    def __init__(self, r, k, f_fric=10):
        self.r = r
        self.k = k
        self.f_fric = f_fric

    def calc_line_segment_force(self, pc, p1, p2):
        ''' Calculate interaction force between circle centered at pc and a
            line segment with endpoints p1 and p2. '''
        p_closest, _ = dist_to_line_segment(pc, p1, p2)
        f, movement = self.calc_point_force(pc, p_closest)
        return f, movement

    def calc_point_force(self, pc, p):
        ''' Calculate interaction force between circle centered at pc and a
            point p. '''
        a = pc - p[:2]
        d = np.linalg.norm(a)
        direction = a / d

        # force is normal to surface -- we do not account for friction
        if d < self.r:
            dx = (self.r - d) * direction
            f = self.k * dx
        else:
            f = np.zeros(2)

        return f

    def apply_force(self, f):
        ''' Apply force f to produce movement, which lowers f so that
            equilbrium with friction is maintained. '''
        if f @ f > self.f_fric**2:
            dx = f / self.k
            f_new = self.f_fric * f / np.linalg.norm(f)
            dx_new = f_new / self.k
            movement = dx - dx_new
        else:
            movement = np.zeros(2)
            f_new = f
        return f_new, movement
