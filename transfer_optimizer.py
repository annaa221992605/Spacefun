import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
from shooting import high_thrust_targeter
from Hohmann_Transfer import low_thrust_propagator
from scipy.optimize import minimize, NonlinearConstraint


def optimize_transfer(initial_guess):
    """
    function to opimize transfer
    """
    nlc = NonlinearConstraint(constraint_fun, 0.0, 0.0, jac=constraint_jac)
    sol = minimize(obj_func, initial_guess, args=(r0, m0, T, Isp, mu),
               constraints=[nlc],
               method='SLSQP', options={'ftol':1e-10, 'maxiter':100})


def obj_func(free_vector):
    total_mass_change = 0
    # 1. Apply DV to the initial state
    # dv1 = difference using initial state
    # solve for m1_diff using rocket equation

    DV1=np.array([DVx, DVy])
    #initial delta v
    DV1_mag = np.linalg.norm(DV1)

    v0 = np.array([vx0, vy0])         # Pre-burn velocity
    v_after_dv1 = v0 + DV1

    g0 = 9.80665

    m1_diff = m0 * np.exp(-DV1_mag / (Isp * g0))
    m1= m0+m1_diff

    # 2. Propagate the state using LT EOM
    # Gives delta m2
    LTtraj, times = low_thrust_propagator_2D(init_r, init_v, tof, 1000, isp, m0)
    #final mass on 4th row, last collumn
    m2 = LTtraj[4, -1]
    delta_m2 = m1-m2

    # 3. Get final state at target orbit
    # dv2 = difference in final states

    xf, yf, vxf, vyf, mf = LTtraj[0, -1], LTtraj[1, -1], LTtraj[2, -1], LTtraj[3,-1], LTtraj[4,-1]

    #whats the target?
    
    # solve for m3_diff using rock et equation
    Totalmasschange = m1_diff + m2 + m3_diff
    """
    Example way
    vx0, vy0, tof, DVx, DVy = p
    r0, m0, T, Isp, mu = args
    state0 = np.hstack((r0, [vx0, vy0, m0]))
    traj, _ = propagate_low_thrust(state0, tof, T, Isp, mu)
    m_f = traj[4, -1]
    """

    return total_mass_change


def residuals(p, r0, m0, target, T, Isp, mu=398600.4415):
    # P is is free vector
    vx0, vy0, tof, DVx, DVy = p
    # propagate the 5-state + STM
    state0 = np.hstack((r0, [vx0, vy0, m0]))
    traj, _ = low_thrust_propagator(state0, tof, T, Isp, mu)   # your routine
    xf, yf, vxf, vyf, _ = traj[:5, -1]

    vxf += DVx
    vyf += DVy
    # final velocity including thrust is already in traj, no Δv column now
    F = np.array([xf - target[0],
                  yf - target[1],
                  vxf - target[2],
                  vyf - target[3]])
    return F

def jacobian(p, r0, m0, target, T, Isp, mu=398600.4415):
    """
    Calculate the Jacobian matrix
    This jacobian should be the deriv of F wrt free variables
    """
    vx0, vy0, tof, DVx, DVy  = p
    state0 = np.hstack((r0, [vx0, vy0, m0]))
    traj, _ = low_thrust_propagator(state0, tof, T, Isp, mu)

    # pull STM at t_f
    Phi = traj[5:, -1].reshape(5, 5)         # rows 5-29 are the 25 STM elements
    Phi_rv = Phi[0:2, 2:4]                   # d(r_f)/d(v0)
    Phi_vv = Phi[2:4, 2:4]                   # d(v_f)/d(v0)

    # a_f for TOF column
    x_f, y_f, vx_f, vy_f = traj[:4, -1]
    r_f = np.hypot(x_f, y_f)
    m_f = traj[4, -1]
    a_f  = -mu*np.array([x_f, y_f])/r_f**3 + (T/m_f)*np.array([vx_f, vy_f])/np.hypot(vx_f, vy_f)

    J = np.zeros((4, 5))
    J[0:2, 0:2] = Phi_rv          # wrt vx0, vy0
    J[2:4, 0:2] = Phi_vv
    J[0, 2] = vx_f                # TOF column
    J[1, 2] = vy_f
    J[2, 2] = a_f[0]
    J[3, 2] = a_f[1]

    stm = Phi
    # deriv of first row
    FX[0, 0:2] = stm[0,2:4]
    FX[0,2] = vx_f
    # End of row is zeros
    # Second Row
    FX[1,0:2] = stm[1,2:4]
    FX[1,2] = vy_f
    # Third Row
    FX[2,0:2] = stm[2,2:4]
    FX[2,2] = a_f[0]
    FX[2,3] = 1.0
    # Fourth Row
    FX[3,0:2] = stm[3,2:4]
    FX[3,2] = a_f[1]
    FX[3,4] = 1.0
    return J

def constraint_fun(p, r0, m0, target, T, Isp, mu):
    """Return the 4-vector F that must be zero at the optimum."""
    return residuals(p, r0, m0, target, T, Isp, mu=mu)

def constraint_jac(p, r0, m0, target, T, Isp, mu):
    return jacobian(p, r0, m0, target, T, Isp, mu=mu)
