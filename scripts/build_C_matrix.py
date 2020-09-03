import numpy as np
import sympy
import IPython


# J11, J12, J21, J22 = sympy.symbols('J11,J12,J21,J22')
#
# J = sympy.Matrix([[J11, J12], [J21, J22]])
#
# P = sympy.Matrix([[1, 0], [0, 0]])
# Q = sympy.Matrix([[0, 0], [0, 1]])

P11, P12, P21, P22 = sympy.symbols('P11,P12,P21,P22')
Q11, Q12, Q21, Q22 = sympy.symbols('Q11,Q12,Q21,Q22')
J11, J12, J21, J22 = sympy.symbols('J11,J12,J21,J22')

P = sympy.Matrix([[P11, P12], [P21, P22]])
Q = sympy.Matrix([[Q11, Q12], [Q21, Q22]])
J = sympy.Matrix([[J11, J12], [J21, J22]])

C = sympy.Matrix([[0, J12], [J21, 0]])

system = sympy.Eq(P @ J, C)
res = sympy.solve(system, (P11, P12, P21, P22))

IPython.embed()
