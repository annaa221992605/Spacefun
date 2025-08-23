import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
from shooting import high_thrust_targeter
from Hohmann_Transfer import keplerian_propagator
from LT_Propagator import low_thrust_propagator_2D
from scipy.optimize import minimize, NonlinearConstraint


def optimize_transfer(initial_guess, r0, m0, T, Isp, mu):#does the initial guess include these
    """
    function to opimize transfer
    """
    nlc = NonlinearConstraint(constraint_fun, 0.0, 0.0, jac=constraint_jac)
    # Current error is that these arguments are not getting passed correctly
    sol = minimize(obj_func, initial_guess,
               constraints=[nlc],
               method='SLSQP', options={'ftol':1e-10, 'maxiter':100})
    return sol

def obj_func(free_vector):#should there be a target array passed in - is the target values coming from shooting method
    total_mass_change = 0
    # 1. Apply DV to the initial state
    # dv1 = difference using initial state
    # solve for m1_diff using rocket equation
    initr2 = 12000
    initv2, steps = np.sqrt(mu/initr2), 1000
    vx0, vy0, tof, DVx, DVy = free_vector[:5]

    DV1=np.array([DVx, DVy])
    #initial delta v
    DV1_mag = np.linalg.norm(DV1)

    v0 = np.array([vx0, vy0])         # Pre-burn velocity
    v_after_dv1 = v0 + DV1

    g0 = 9.80665

    m1 = m0 * np.exp(-DV1_mag / (Isp * g0))
    delta_m1= m0-m1

    # 2. Propagate the state using LT EOM
    # Gives delta m2

    #hstack=horizontal
    #state0 = np.hstack((r0, [vx0, vy0, m0]))

    LTtraj, times = low_thrust_propagator_2D([r_LEO,0.0], v_after_dv1, tof, 1000, Isp, m1, thrust) #tof between the first burn and the second
    #final mass on 4th row, last collumn
    m2 = LTtraj[4, -1]
    delta_m2 = m1-m2

    # 3. Get final state at target orbit
    # dv2 = difference in final states

    #setting mf = to the mass after second burn
    xf, yf, vxf, vyf, mf = LTtraj[0, -1], LTtraj[1, -1], LTtraj[2, -1], LTtraj[3,-1], LTtraj[4,-1] #-1 is the last column (if steps=1000, last column will be 1000)


    #whats the target?
    
    # solve for m3_diff using rocket equation
    target_traj, times = keplerian_propagator([initr2,0.0], [0.0,initv2],tof, steps)

    target_vx, target_vy = target_traj[2:4,-1] #where should I get this

    DV2=np.array([target_vx - vxf, target_vy - vyf])#target velocities where
    #initial delta v
    DV2_mag = np.linalg.norm(DV2)#pythagorean thoerem

    mf_after = mf * np.exp(-DV2_mag / (Isp * g0)) #mf and m2 are the same (both 4, -1)
    delta_m3 = mf - mf_after

    total_mass_change = delta_m1 + delta_m2 + delta_m3
    """
    Example way
    vx0, vy0, tof, DVx, DVy = p
    r0, m0, T, Isp, mu = args
    state0 = np.hstack((r0, [vx0, vy0, m0]))
    traj, _ = propagate_low_thrust(state0, tof, T, Isp, mu)
    m_f = traj[4, -1]
    """

    return total_mass_change


def residuals(p, r0, m1, T, Isp, mu=398600.4415):
    # P is is free vector
    vx0, vy0, tof, DVx, DVy = p
    # propagate the 5-state + STM
    state0 = np.hstack((r0, [vx0, vy0, m0]))
    traj, _ = low_thrust_propagator_2D([r0,0.0], [vx0, vy0], tof, 1000, Isp, m1, thrust)   # your routine
    xf, yf, vxf, vyf, _ = traj[:5, -1]
    initr2 = 12000
    initv2, steps = np.sqrt(mu/initr2), 1000

    target_traj, times = keplerian_propagator([initr2,0.0], [0.0,initv2],tof, steps)

    target = target_traj[0:4,-1]
    vxf += DVx
    vyf += DVy
    # final velocity including thrust is already in traj, no Δv column now
    F = np.array([xf - target[0],
                  yf - target[1],
                  vxf - target[2],
                  vyf - target[3]])
    return F

def jacobian(p, r0, m1, T, Isp, mu=398600.4415):
    """
    Calculate the Jacobian matrix
    This jacobian should be the deriv of F wrt free variables
    """
    vx0, vy0, tof, DVx, DVy  = p
    state0 = np.hstack((r0, [vx0, vy0, m0]))
    traj, _ = low_thrust_propagator_2D([r0,0.0], [vx0, vy0], tof, 1000, Isp, m1, thrust)

    # pull STM at t_f
    Phi = traj[5:, -1].reshape(5, 5)         # rows 5-29 are the 25 STM elements
    Phi_rv = Phi[0:2, 2:4]                   # d(r_f)/d(v0)
    Phi_vv = Phi[2:4, 2:4]                   # d(v_f)/d(v0)

    # a_f for TOF column
    x_f, y_f, vx_f, vy_f = traj[:4, -1]
    r_f = np.hypot(x_f, y_f)
    m_f = traj[4, -1]
    a_f  = -mu*np.array([x_f, y_f])/r_f**3 + (T/m_f)*np.array([vx_f, vy_f])/np.hypot(vx_f, vy_f)
#missing parts of j?
    J = np.zeros((4, 5))
    J[0:2, 0:2] = Phi_rv          # wrt vx0, vy0
    J[2:4, 0:2] = Phi_vv
    J[0, 2] = vx_f                # TOF column
    J[1, 2] = vy_f
    J[2, 2] = a_f[0]
    J[3, 2] = a_f[1]
    J[2,3]=1.0
    J[3, 4]=1.0

   
    return J

def constraint_fun(p):
    """Return the 4-vector F that must be zero at the optimum."""
    r0 = 7000
    m0=1000
    T=0.5
    Isp=3000
    mu=398600.0
    vx0, vy0, tof, DVx, DVy = p[:5]
    DV1=np.array([DVx, DVy])
    #initial delta v
    DV1_mag = np.linalg.norm(DV1)
    g0 = 9.80665
    m1 = m0 * np.exp(-DV1_mag / (Isp * g0))
    return residuals(p, r0, m1, T, Isp, mu)

def constraint_jac(p):
    r0 = 7000
    m0=1000
    T=0.5
    Isp=3000
    mu=398600.0
    vx0, vy0, tof, DVx, DVy = p[:5]
    DV1=np.array([DVx, DVy])
    #initial delta v
    DV1_mag = np.linalg.norm(DV1)
    g0 = 9.80665
    m1 = m0 * np.exp(-DV1_mag / (Isp * g0))
    return jacobian(p, r0, m1, T, Isp, mu)

def plot_hybrid_trajectory (r_LEO, r_GEO, LTtraj, GTOtraj, r_vec):
    
    tof_array = np.linspace(0,2*np.pi, num=1000)
    x0 = r_LEO *np.cos(tof_array)
    y0 = r_LEO * np.sin(tof_array)
    x2 = r_GEO * np.cos(tof_array) 
    y2 = r_GEO * np.sin(tof_array)

    print("LEO", r_LEO, "GEO", r_GEO)

    # GTO ellipse for reference as dashed (peri = LEO, apo = GEO)
    a_GTO = (r_LEO + r_GEO) / 2
    b_GTO = np.sqrt(r_LEO * r_GEO)  # For a transfer ellipse (r_per, r_apo)

    # Parametric ellipse (focus at center for simplicity—slight visual offset)
    x_gto = a_GTO * np.cos(tof_array)
    y_gto = b_GTO * np.sin(tof_array)

    #calculate 

    plt.figure(figsize=(8, 7))

    plt.plot(x0, y0, 'k--', label='LEO Orbit')
    plt.plot(x2, y2, 'b--', label='GEO Orbit')
    plt.plot(x_gto, y_gto, 'c--', linewidth=1.5, label='GTO Ellipse')

    # High-thrust arc (GTOtraj): blue solid line
    plt.plot(GTOtraj[0], GTOtraj[1], color='blue', linewidth=2, label='high-thrust')
    plt.plot(LTtraj[0], LTtraj[1], 'r', linewidth=2, label='Low-Thrust Arc')

    plt.plot(GTOtraj[0,0], GTOtraj[1, 0], 'go', markersize=10, label='Start Burn (Impulsive)')
    plt.plot(LTtraj[0, -1], LTtraj[1, -1], 'mo', markersize=10, label='End Burn (Impulsive)')

    plt.scatter(0, 0, color='red', s=100, label='Centeral body')

    plt.xlabel('X (km)')
    plt.ylabel('Y (km)')
    plt.axis('equal')
    plt.grid(True)
    plt.title('Hybrid Impulsive/Low-Thrust Transfer')
    plt.legend(loc='best')
    plt.show()








r_LEO = 6378+622
r_GEO = 42164
m0=1000
Isp=3000
mu=398600.0
thrust = 0.0005

#target, times = keplerian_propagator(initr2, initv2,2*np.pi*np.sqrt(r2**3/grav), integration_steps)
#initial orbit values
v_LEO = np.sqrt(mu / r_LEO)
init_pos_LEO = np.array([r_LEO, 0.0])
init_vel_LEO = np.array([0.0, v_LEO])
#desired orbit velocity
v2 = np.sqrt(mu/r_GEO)

#transfer orbit calculations
a_GTO = (r_LEO+r_GEO)/2

v_periapse_GTO = np.sqrt(mu * (2 / r_LEO - 1 / a_GTO))  # at LEO (periapsis) after burn
GTO_tof = np.pi * np.sqrt(a_GTO ** 3 / mu)  #perigee to apogee time

# (c) Post-burn state (GTO entry)
init_vel_GTO = np.array([0.0, v_periapse_GTO])

# (d) GTO propagation (High thrust/impulsive segment: LEO→GTO apogee)
GTO_traj, GTO_times = keplerian_propagator(init_pos_LEO, init_vel_GTO, GTO_tof, 1000)
final_GTO_pos = GTO_traj[:2, -1]       # r at GTO apogee (should be [r_GEO, ...])
final_GTO_vel = GTO_traj[2:4, -1]      # v at GTO apogee

# (a) Mass/State update after impulsive burn (can use rocket equation if desired)
delta_v_GTO = v_periapse_GTO - v_LEO
g0 = 9.80665
m_after_GTO = m0 * np.exp(-delta_v_GTO / (Isp * g0))

vTransferApo = np.sqrt(mu*(2/r_LEO-1/a_GTO))


#deltav calculations

delta_v1 = v_periapse_GTO - v_LEO
delta_v2 = v2 - vTransferApo
free_vector = high_thrust_targeter(r_LEO, 0, 0, v_LEO+delta_v1, 0, -delta_v2, -r_GEO, 0, 0, -v2, GTO_tof)

#initial_guess = [vx0, vy0, tof0, DVx0, DVy0]
initial_guess = free_vector
#optimize
sol=optimize_transfer(initial_guess, r_LEO, m0, thrust, Isp, mu)
print(sol)

vx0, vy0, tof, DVx, DVy = sol.x
DV1 = np.array([DVx, DVy])
DV1_mag = np.linalg.norm(DV1)
g0 = 9.80665
m1 = m0 * np.exp(-DV1_mag / (Isp * g0))

r0_vec =  init_pos_LEO             # Starting position vector (x, y)
v0_vec =  init_vel_GTO      # Velocity after impulsive burn

LT_tof = 30*24*3600 #do not leave like this

LT_traj, LT_times = low_thrust_propagator_2D(final_GTO_pos, final_GTO_vel, LT_tof, 1000, Isp, m_after_GTO, thrust)#TOF should be optimized
HTtraj, times = keplerian_propagator(r0_vec, v0_vec, tof, 1000)

print("GTO_traj sample:", GTO_traj[0,:5], GTO_traj[1,:5])
print("LT_traj sample:", LT_traj[0,:5], LT_traj[1,:5])
print("LT_traj end:", LT_traj[0, -5:], LT_traj[1, -5:])

plot_hybrid_trajectory(r_LEO, r_GEO, LT_traj, HTtraj, r0_vec)
