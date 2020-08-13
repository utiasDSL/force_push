import numpy as np
import sympy
import IPython


def to_np(M):
    return np.array(M).astype(np.float64)


def rot2(a):
    s = np.sin(a)
    c = np.cos(a)
    return np.array([[c, -s], [s, c]])


def sys3in2out():
    a, u1, u2, u3 = sympy.symbols('a,u1,u2,u3')
    c = sympy.cos(a)
    s = sympy.sin(a)

    J = sympy.Matrix([[1, 2, 3], [4, 5, 6]])
    P = sympy.Matrix([[c, -s], [s, c]])
    u = sympy.Matrix([u1, u2, u3])
    v = sympy.Matrix([1, 2])

    PJ = P @ J
    C = sympy.Matrix([[0, PJ[0, 1], PJ[0, 2]], [PJ[1, 0], 0, 0]])

    A = sympy.Matrix([J, C])
    b = sympy.Matrix([v, 0, 0])

    eqns = A @ u - b

    solns = sympy.solve(eqns, [a, u1, u2, u3])

    J = to_np(J)

    for soln in solns:
        soln = to_np(soln)
        a = soln[0]
        u = soln[1:]
        P = rot2(a)

        print('a = {}\nu = u{}\ncost = {}'.format(a, u, u @ u))
        IPython.embed()


def sys4in2out():
    a, u1, u2, u3, u4 = sympy.symbols('a,u1,u2,u3,u4')
    c = sympy.cos(a)
    s = sympy.sin(a)

    J = sympy.Matrix([[1, 2, 3, 4], [5, 6, 7, 8]])
    P = sympy.Matrix([[c, -s], [s, c]])
    u = sympy.Matrix([u1, u2, u3, u4])
    v = sympy.Matrix([1, 2])

    PJ = P @ J
    C = sympy.Matrix([[0, PJ[0, 1], PJ[0, 2], PJ[0, 3]], [PJ[1, 0], 0, 0, 0]])

    A = sympy.Matrix([J, C])
    b = sympy.Matrix([v, 0, 0])

    eqns = A @ u - b

    solns = sympy.solve(eqns, [a, u1, u2, u3])

    IPython.embed()


sys3in2out()
# sys4in2out()
