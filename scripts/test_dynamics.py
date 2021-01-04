import numpy as np
import jax
import jax.numpy as jnp
import sympy as sym
from functools import partial
import IPython

Mb = 3
M1 = 2
M2 = 1

LX = 0
LY = 0
L1 = 1
L2 = 0.5

I1 = M1 * L1**2 / 12
I2 = M2 * L2**2 / 12

G = 9.8


def symbolic_dynamics(time):
    t = sym.symbols('t')
    q = sym.Matrix([sym.Function('q1')(t), sym.Function('q2')(t), sym.Function('q3')(t)])
    dq = q.diff(t)

    x1 = q[0] + LX + 0.5*L1*sym.cos(q[1])
    y1 = LY + 0.5*L1*sym.sin(q[1])
    x2 = q[0] + LX + L1*sym.cos(q[1]) + 0.5*L2*sym.cos(q[1]+q[2])
    y2 = LY + L1*sym.sin(q[1]) + 0.5*L2*sym.sin(q[1]+q[2])

    dx1 = x1.diff(t)
    dy1 = y1.diff(t)
    dx2 = x2.diff(t)
    dy2 = y2.diff(t)

    # Potential energy
    Pb = 0
    P1 = M1*G*y1
    P2 = M2*G*y2
    P = Pb + P1 + P2

    # Kinetic energy
    Kb = 0.5*Mb*dq[0]**2
    K1 = 0.5*M1*(dx1**2+dy1**2) + 0.5*I1*dq[1]**2
    K2 = 0.5*M2*(dx2**2+dy2**2) + 0.5*I2*(dq[1]+dq[2])**2
    K = Kb + K1 + K2

    # Lagrangian
    L = K - P

    # Generalized forces
    tau = L.diff(dq).diff(t) - L.diff(q)

    return tau.subs({
        q[0]: sym.sin(t),
        q[1]: t,
        q[2]: t*t,
        t: time}).doit()


def configuration(t, np=np):
    ''' Define joint configuration as function of time. '''
    q = np.array([np.sin(t), t, t*t])
    return q


def calc_mass_matrix(q, np=np):
    xb, θ1, θ2 = q
    θ12 = θ1 + θ2

    m11 = Mb + M1 + M2
    m12 = -(0.5*M1+M2)*L1*np.sin(θ1) - 0.5*M2*L2*np.sin(θ12)
    m13 = -0.5*M2*L2*np.sin(θ12)

    m22 = (0.25*M1+M2)*L1**2 + 0.25*M2*L2**2 + M2*L1*L2*np.cos(θ2) + I1 + I2
    m23 = 0.5*M2*L2*(0.5*L2+L1*np.cos(θ2)) + I2

    m33 = 0.25*M2*L2**2 + I2

    M = np.array([
        [m11, m12, m13],
        [m12, m22, m23],
        [m13, m23, m33]])
    return M


def calc_christoffel(q, np=np):
    xb, θ1, θ2 = q
    θ12 = θ1 + θ2

    # Partial derivations of mass matrix
    dMdxb = np.zeros((3, 3))

    dMdθ1_12 = -0.5*M1*L1*np.cos(θ1) - M2*L1*np.cos(θ1) - 0.5*M2*L2*np.cos(θ12)
    dMdθ1_13 = -0.5*M2*L2*np.cos(θ12)
    dMdθ1 = np.array([
        [0, dMdθ1_12, dMdθ1_13],
        [dMdθ1_12, 0, 0],
        [dMdθ1_13, 0, 0]])

    dMdθ2_12 = -0.5*M2*L2*np.cos(θ12)
    dMdθ2_13 = -0.5*M2*L2*np.cos(θ12)
    dMdθ2_22 = -M2*L1*L2*np.sin(θ2)
    dMdθ2_23 = -0.5*M2*L1*L2*np.sin(θ2)
    dMdθ2 = np.array([
        [0,        dMdθ2_12, dMdθ2_13],
        [dMdθ2_12, dMdθ2_22, dMdθ2_23],
        [dMdθ2_13, dMdθ2_23, 0]])

    dMdq = np.zeros((3, 3, 3))
    dMdq[:, :, 0] = dMdxb
    dMdq[:, :, 1] = dMdθ1
    dMdq[:, :, 2] = dMdθ2

    Γ = dMdq - 0.5*dMdq.T

    # Construct matrix of Christoffel symbols
    # Γ = np.zeros((3, 3, 3))
    # Γ[:, :, 0] += dMdq[0, :, :]
    # Γ[:, :, 1] += dMdq[1, :, :]
    # Γ[:, :, 2] += dMdq[2, :, :]

    # Γ[0, :, :] -= 0.5*dMdq[0, :, :]
    # Γ[1, :, :] -= 0.5*dMdq[1, :, :]
    # Γ[2, :, :] -= 0.5*dMdq[2, :, :]

    # for i in range(3):
    #     for j in range(3):
    #         for k in range(3):
    #             Γ[i, j, k] = 0.5*(dMdq[k, j, i] + dMdq[i, k, j] - dMdq[i, j, k])  # mine
    #             # Γ[i, j, k] = dMdq[k, j, i] - 0.5*dMdq[i, j, k]  # mine
    #             # Γ[i, j, k] = 0.5*(dMdq[k, j, i] + dMdq[k, i, j] - dMdq[i, j, k])  # Spong
    return Γ


def calc_gravity_vector(q):
    xb, θ1, θ2 = q
    θ12 = θ1 + θ2
    return np.array([0,
                     (0.5*M1+M2)*G*L1*np.cos(θ1) + 0.5*M2*L2*G*np.cos(θ12),
                     0.5*M2*L2*G*np.cos(θ12)])


def manual_dynamics_mat(q, dq, ddq):
    M = calc_mass_matrix(q)
    Γ = calc_christoffel(q)
    g = calc_gravity_vector(q)

    return M @ ddq + dq @ Γ @ dq + g


def potential_energy(q, np=np):
    y1 = LY + 0.5*L1*np.sin(q[1])
    y2 = LY + L1*np.sin(q[1]) + 0.5*L2*np.sin(q[1]+q[2])

    Pb = 0
    P1 = M1*G*y1
    P2 = M2*G*y2
    P = Pb + P1 + P2

    return P


def kinetic_energy(q, dq, np=np):
    # x1 = q[0] + LX + 0.5*L1*np.cos(q[1])
    # y1 = LY + 0.5*L1*np.sin(q[1])
    # x2 = q[0] + LX + L1*np.cos(q[1]) + 0.5*L2*np.cos(q[1]+q[2])
    # y2 = LY + L1*np.sin(q[1]) + 0.5*L2*np.sin(q[1]+q[2])

    # TODO can we auto-diff these?
    # TODO this should be done with Jacobian calculations (i.e. linear in dq)
    dx1 = dq[0] - 0.5*L1*dq[1]*np.sin(q[1])
    dy1 = 0.5*L1*dq[1]*np.cos(q[1])
    dx2 = dq[0] - L1*dq[1]*np.sin(q[1]) - 0.5*L2*(dq[1]+dq[2])*np.sin(q[1]+q[2])
    dy2 = L1*dq[1]*np.cos(q[1]) + 0.5*L2*(dq[1]+dq[2])*np.cos(q[1]+q[2])

    Kb = 0.5*Mb*dq[0]**2
    K1 = 0.5*M1*(dx1**2+dy1**2) + 0.5*I1*dq[1]**2
    K2 = 0.5*M2*(dx2**2+dy2**2) + 0.5*I2*(dq[1]+dq[2])**2

    return Kb + K1 + K2


def lagrangian(q, dq, np=np):
    K = kinetic_energy(q, dq, np=np)
    P = potential_energy(q, np=np)
    return K - P


def auto_diff_dynamics():
    q_func = partial(configuration, np=jnp)
    dq_func = jax.jacfwd(partial(configuration, np=jnp))

    # diff Lagrangian w.r.t. q
    dLdq_func = jax.grad(partial(lagrangian, np=jnp), argnums=0)

    # diff Lagrangian w.r.t. dq
    dLddq_func = jax.grad(partial(lagrangian, np=jnp), argnums=1)

    # compose to make dLdq a function of time t
    def dLddq_func_t(t):
        return dLddq_func(q_func(t), dq_func(t))

    # diff dLdq w.r.t. t
    ddt_dLddq_func_t = jax.jacfwd(dLddq_func_t)

    # generalized forces expressed as a function of time
    def tau_func(t):
        q = q_func(t)
        dq = dq_func(t)
        return ddt_dLddq_func_t(t) - dLdq_func(q, dq)

    return jax.jit(tau_func)


def fkb_ad(q, np=np):
    return np.array([q[0], 0, 0])

def fk1_ad(q, np=np):
    return np.array([
        q[0] + LX + 0.5*L1*np.cos(q[1]),
        LY + 0.5*L1*np.sin(q[1]),
        q[1]])

def fk2_ad(q, np=np):
    return np.array([
        q[0] + LX + L1*np.cos(q[1]) + 0.5*L2*np.cos(q[1]+q[2]),
        LY + L1*np.sin(q[1]) + 0.5*L2*np.sin(q[1]+q[2]),
        q[1] + q[2]
    ])


def calc_mass_matrix_ad(q, np=np):
    # Jacobians
    Jb = jax.jacfwd(partial(fkb_ad, np=jnp))(q)
    J1 = jax.jacfwd(partial(fk1_ad, np=jnp))(q)
    J2 = jax.jacfwd(partial(fk2_ad, np=jnp))(q)

    Gb = np.diag(np.array([Mb, Mb, 0]))
    G1 = np.diag(np.array([M1, M1, I1]))
    G2 = np.diag(np.array([M2, M2, I2]))

    return np.dot(np.dot(Jb.T, Gb), Jb) + np.dot(np.dot(J1.T, G1), J1) + np.dot(np.dot(J2.T, G2), J2)


def dynamics_ad(q, dq, ddq, np=np):
    dMdq = jax.jacfwd(partial(calc_mass_matrix_ad, np=jnp))(q)

    M = calc_mass_matrix_ad(q)
    Γ = dMdq - 0.5*dMdq.T
    g = calc_gravity_vector(q)

    return M.dot(ddq) + dq.dot(Γ).dot(dq) + g


def main():
    tau_func = auto_diff_dynamics()

    q_func = partial(configuration, np=jnp)
    dq_func = jax.jit(jax.jacfwd(partial(configuration, np=jnp)))
    ddq_func = jax.jit(jax.jacfwd(dq_func))

    t = 1.0
    q = q_func(t)
    dq = dq_func(t)
    ddq = ddq_func(t)

    dMdq_func = jax.jacfwd(partial(calc_mass_matrix, np=jnp))
    M = calc_mass_matrix(q)
    g = calc_gravity_vector(q)
    dMdq = dMdq_func(q)

    print(dynamics_ad(q, dq, ddq))
    print(manual_dynamics_mat(q, dq, ddq))

    print(tau_func(t))
    # print(np.array(symbolic_dynamics(t)).astype(np.float64).flatten())
    # print(manual_dynamics_mat(q, dq, ddq))

    # IPython.embed()


if __name__ == '__main__':
    main()
