import numpy as np
import matplotlib.pyplot as plt
import jax.numpy as jnp
import jax
from scipy import optimize
import time
import IPython


MASS = 1.0
SIDE = 1.0
MOMENT_INERTIA = MASS*SIDE**2/6.0
M = np.diag([MASS, MASS, MOMENT_INERTIA])


P1 = np.array([1, 1, 0])

DT = 0.1
pe1 = P1[:2] + np.array([-0.5*SIDE, 0.25*SIDE])
ve = np.array([1., 0.])
pe2 = pe1 + DT*ve
pe3 = pe2 + DT*ve


def square_pts(P):
    p = P[:2]
    theta = P[2]
    R = np.array([[np.cos(theta), -np.sin(theta)],
                  [np.sin(theta),  np.cos(theta)]])
    square0 = 0.5*np.array([[-SIDE, SIDE, SIDE, -SIDE, -SIDE],
                            [SIDE, SIDE, -SIDE, -SIDE, SIDE]])
    return p[:, None] + R.dot(square0)


def square_obj(Pc, Pc0):
    # Pc is the opt variable
    e = Pc0 - Pc
    return 0.5*jnp.dot(jnp.dot(e, M), e)


def square_constr(Pc, pe):
    # Pc is the opt var, pe is the new position of the EE
    pc = Pc[:2]
    theta = Pc[2]
    R = jnp.array([[jnp.cos(theta), -jnp.sin(theta)],
                   [jnp.sin(theta),  jnp.cos(theta)]]).T
    pb2 = 0.5*jnp.array([-SIDE, SIDE])
    pb1 = 0.5*jnp.array([-SIDE, -SIDE])
    pb21 = pb2 - pb1
    x = jnp.dot(R, pe - pc) - pb1
    eq = jnp.dot(x, pb21) - SIDE*jnp.linalg.norm(x)
    return eq


def square_lagrangian(var, Pc0, pe0):
    # var is the full set of opt variables now: pc_k+1 (3), λ (1), ve (2)
    Pc = var[:3]
    λ = var[3]
    ve = var[4:]
    pe = pe0 + DT*ve
    L = square_obj(Pc, Pc0) + λ*square_constr(Pc, pe)
    return L


def control_obj(var, pg):
    pc = var[:2]
    e = pg - pc
    return 0.5*jnp.dot(e, e)


def radial_constr(var, limit):
    ve = var[4:]
    return limit**2 - jnp.dot(ve, ve)


# new equality constraint is the derivative of the Lagrangian
eq_fun_full = jax.jit(jax.grad(square_lagrangian, argnums=0))
def eq_fun(var, Pc0, pe0):
    return eq_fun_full(var, Pc0, pe0)[:4]

eq_jac = jax.jit(jax.jacfwd(eq_fun, argnums=0))
obj_jac = jax.jit(jax.jacfwd(control_obj, argnums=0))
ineq_jac = jax.jit(jax.jacfwd(radial_constr, argnums=0))


def solve_opt(Pc0, pe0, pg):
    guess = np.concatenate((Pc0, np.array([0, 1, 0])))
    obj_args = (pg,)
    eq_args = (Pc0, pe0)
    ineq_args = (1.0,)
    eq_constr = [{
        'type': 'eq',
        'fun': eq_fun,
        'jac': eq_jac,
        'args': eq_args
    }, {
        'type': 'ineq',
        'fun': radial_constr,
        'jac': ineq_jac,
        'args': ineq_args
    }]
    res = optimize.minimize(control_obj, guess, args=obj_args, jac=obj_jac,
                            constraints=eq_constr, method='slsqp')
    return res.x


Pc0 = np.array([1, 1, 0])
pe0 = Pc0[:2] + np.array([-0.5*SIDE, 0*SIDE])
pg = Pc0[:2] + np.array([1, 0])

# var1 = solve_opt(Pc0, pe0, pg)
# Pc1 = var1[:3]
# ve0 = var1[4:]
# pe1 = pe0 + DT*ve0
# var2 = solve_opt(Pc1, pe1, pg)
# Pc2 = var2[:3]
# ve1 = var2[4:]
# pe2 = pe1 + DT*ve1
Pc = Pc0
pe = pe0

N = 10
pes = np.zeros((N, 2))

for i in range(N):
    var = solve_opt(Pc, pe, pg)
    Pc = var[:3]
    ve = var[4:]
    pe = pe + DT*ve
    pes[i, :] = pe

    square = square_pts(Pc)
    plt.plot(square[0, :], square[1, :], label=str(i))


# square0 = square_pts(Pc0)
# square1 = square_pts(Pc1)
# square2 = square_pts(Pc2)
#
# plt.plot(square0[0, :], square0[1, :], label='0')
# plt.plot(square1[0, :], square1[1, :], label='1')
# plt.plot(square2[0, :], square2[1, :], label='2')
# plt.plot([pe0[0], pe1[0], pe2[0]], [pe0[1], pe1[1], pe2[1]], color='k')
plt.plot(pes[:, 0], pes[:, 1], color='k')
plt.xlim([-1, 3])
plt.ylim([-1, 3])
plt.legend()
plt.grid()
plt.show()

# IPython.embed()
# import sys; sys.exit()

# obj_jac = jax.jit(jax.jacfwd(square_obj, argnums=0))
# eq_jac = jax.jit(jax.jacfwd(square_constr, argnums=0))
#
#
# def solve_opt(pe, Pc):
#     obj_args = (Pc,)
#     eq_args = (pe,)
#     eq_constr = {
#             'type': 'eq',
#             'fun': eq_constraint,
#             'jac': eq_jac,
#             'args': eq_args
#     }
#     res = optimize.minimize(objective, Pc, args=obj_args, jac=obj_jac,
#                             constraints=eq_constr, method='slsqp')
#     return res.x
#
#
# P2 = solve_opt(pe2, P1)
# P3 = solve_opt(pe3, P2)
#
# # IPython.embed()
#
# square1 = square_pts(P1)
# square2 = square_pts(P2)
# square3 = square_pts(P3)
#
# plt.plot(square1[0, :], square1[1, :], label='1')
# plt.plot(square2[0, :], square2[1, :], label='2')
# plt.plot(square3[0, :], square3[1, :], label='3')
# plt.plot([pe1[0], pe2[0], pe3[0]], [pe1[1], pe2[1], pe3[1]], color='k')
# plt.xlim([-1, 3])
# plt.ylim([-1, 3])
# plt.legend()
# plt.grid()
# plt.show()
