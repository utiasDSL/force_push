import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import fsolve
import IPython


def main():
    a = 0.2
    w = 0.5
    l1 = 0.5
    l2 = 0.5

    # origin of base frame
    pb = np.array([0, 0])
    th = 0

    # end points of front of base
    pbl = pb + np.array([a, 0.5 * w])
    pbr = pb + np.array([a, -0.5 * w])

    q1 = -0.5 * np.pi
    q2 = 0.7 * np.pi

    p1 = pb + l1 * np.array([np.cos(q1), np.sin(q1)])
    pe = p1 + l2 * np.array([np.cos(q1+q2), np.sin(q1+q2)])

    # solve for base contact position pc and object center po
    ro = 0.25  # known object radius

    xc = pb[0] + a

    def equations(args):
        yc, xo, yo = args

        # base contact point is on the circle
        eq1 = (xo - xc)**2 + (yo - yc)**2 - ro**2

        # EE position is on the circle
        eq2 = (xo - pe[0])**2 + (yo - pe[1])**2 - ro**2

        # pc to center of circle is perpendicular to front of base
        eq3 = (xc - xo) * (pbl[0] - pbr[0]) + (yc - yo) * (pbl[1] - pbr[1])
        return [eq1, eq2, eq3]

    # TODO there is likely an analytic solution to this problem
    yc, xo, yo = fsolve(equations, (pb[1], xc + ro, pb[1]))
    pc = np.array([xc, yc])
    po = np.array([xo, yo])
    print(f'{xo} {yo}')

    delta = pbl - pbr
    alpha = np.sqrt((delta[1]*ro)**2/(delta[0]**2 + delta[1]**2))

    # only one of these should be possible if the end effector is located in
    # front of the base
    if ro**2 - (xc + alpha - pe[0])**2 >= 0:
        xo = xc + alpha
    else:
        xo = xc - alpha

    # could be plus or minus
    yo = pe[1] + np.sqrt(ro**2 - (xo - pe[0])**2)

    IPython.embed()


    # find the point at the tip of the velocity cone
    A = np.vstack((po - pc, po - pe))
    b = np.array([(po - pc) @ pc, (po - pe) @ pe])
    pivot = np.linalg.solve(A, b)

    plt.plot([pb[0], pbl[0], pbr[0], p1[0], pe[0]], [pb[1], pbl[1], pbr[1], p1[1], pe[1]], 'o', color='k')
    plt.plot([pbl[0], pbr[0]], [pbl[1], pbr[1]], '-', color='k')
    plt.plot([pb[0], p1[0], pe[0]], [pb[1], p1[1], pe[1]], '-', color='b')
    plt.plot([xo], [yo], 'o', color='r')
    plt.plot([xc], [yc], 'o', color='g')
    plt.plot([pivot[0]], [pivot[1]], 'o', color='y')
    ax = plt.gca()
    ax.add_patch(plt.Circle((xo, yo), ro, color='r', fill=False))
    plt.xlim([-1, 3])
    plt.ylim([-2, 2])
    plt.grid()
    plt.show()


if __name__ == '__main__':
    main()
