import numpy as np
import matplotlib.pyplot as plt
import jax.numpy as jnp
import jax
from functools import partial
from scipy import optimize
import IPython


MASS = 1.0
SIDE = 1.0
MOMENT_INERTIA = MASS*SIDE**2/6.0
M = np.diag([MASS, MASS, MOMENT_INERTIA])


P1 = np.array([1, 1, 0])

pe1 = P1[:2] + np.array([-0.5*SIDE, 0.25*SIDE])
ve = np.array([0.1, 0])
pe2 = pe1 + ve
pe3 = pe2 + ve

def square_pts(P):
    p = P[:2]
    theta = P[2]
    R = np.array([[np.cos(theta), -np.sin(theta)],
                  [np.sin(theta),  np.cos(theta)]])
    square0 = 0.5*np.array([[-SIDE, SIDE, SIDE, -SIDE, -SIDE],
                            [SIDE, SIDE, -SIDE, -SIDE, SIDE]])
    return p[:, None] + R.dot(square0)


def objective(var, P):
    return 0.5*jnp.dot(jnp.dot(P-var, M), P-var)


def eq_constraint(var, pe, embed=False):
    pc = var[:2]
    theta = var[2]
    R = jnp.array([[jnp.cos(theta), -jnp.sin(theta)],
                   [jnp.sin(theta),  jnp.cos(theta)]]).T
    pb2 = 0.5*jnp.array([-SIDE, SIDE])
    pb1 = 0.5*jnp.array([-SIDE, -SIDE])
    pb21 = pb2 - pb1
    x = jnp.dot(R, pe - pc) - pb1
    eq = jnp.dot(x, pb21) - SIDE*jnp.linalg.norm(x)
    if embed:
        IPython.embed()
    return eq


# obj_fun = partial(objective, P1)
# obj_jac = jax.jit(jax.jacfwd(obj_fun))
# eq_fun = partial(eq_constraint, pe2)
# eq_jac = jax.jit(jax.jacfwd(eq_fun))
# eq_constr = {
#         'type': 'eq',
#         'fun': eq_fun,
#         'jac': eq_jac
# }
#
# res = optimize.minimize(obj_fun, P1, jac=obj_jac, constraints=eq_constr, method='slsqp')
# P2 = res.x

obj_jac = jax.jit(jax.jacfwd(objective, argnums=0))
eq_jac = jax.jit(jax.jacfwd(eq_constraint, argnums=0))


def solve_opt(pe, Pc):
    obj_args = (Pc,)
    eq_args = (pe,)
    eq_constr = {
            'type': 'eq',
            'fun': eq_constraint,
            'jac': eq_jac,
            'args': eq_args
    }
    res = optimize.minimize(objective, Pc, args=obj_args, jac=obj_jac,
                            constraints=eq_constr, method='slsqp')
    return res.x


P2 = solve_opt(pe2, P1)
P3 = solve_opt(pe3, P2)

# IPython.embed()

square1 = square_pts(P1)
square2 = square_pts(P2)
square3 = square_pts(P3)

plt.plot(square1[0, :], square1[1, :], label='1')
plt.plot(square2[0, :], square2[1, :], label='2')
plt.plot(square3[0, :], square3[1, :], label='3')
plt.plot([pe1[0], pe2[0], pe3[0]], [pe1[1], pe2[1], pe3[1]], color='k')
plt.xlim([-1, 3])
plt.ylim([-1, 3])
plt.legend()
plt.grid()
plt.show()
