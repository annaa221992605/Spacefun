import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
from shooting import high_thrust_targeter
from Hohmann_Transfer import keplerian_propagator
from LT_Propagator import low_thrust_propagator_2D
from scipy.optimize import minimize, NonlinearConstraint


def optimize_transfer(initial_guess, r0, m0, T, Isp_low, Isp_high, mu, r_GEO):
    """
    function to opimize transfer
    """
    nlc = NonlinearConstraint(
        lambda p: constraint_fun(p, r0, m0, T, Isp_low, Isp_high, mu,  r_GEO),
        0.0, 0.0,
        jac=lambda p: constraint_jac(p, r0, m0, T, Isp_low, Isp_high, mu,  r_GEO)
    )
    def my_callback(xk):
        print(f"Iteration callback: x = {xk}")

    sol = minimize(
        lambda p: obj_func(p, r0, m0, T, Isp_low, Isp_high, mu, final_GTO_pos, final_GTO_vel),
        initial_guess,
        constraints=[nlc],
        method='SLSQP', callback=my_callback, options={'ftol':1e-8, 'maxiter':5})
    return sol

def obj_func(free_vector, r0, m0, T, Isp_low, Isp_high, mu, r_GEO=42164):#should there be a target array passed in - is the target values coming from shooting method
    total_mass_change = 0
    # 1. Apply DV to the initial state
    # dv1 = difference using initial state
    # solve for m1_diff using rocket equation
    
    vx0, vy0, tof_lt = free_vector[:3]  # Only optimize these for this scenario

    g0 = 9.80665  # m/s^2

    # Initial state in LEO
    init_pos_LEO = np.array([r0, 0.0])
    init_vel_LEO = np.array([vx0, vy0])

    # Step 1: Low-thrust propagation from LEO to GTO apogee
    LT_traj, times = low_thrust_propagator_2D(
        init_pos_LEO, init_vel_LEO, tof_lt, 1000, Isp_low, m0, T, r_GEO
    )
    # State at end of low thrust (GTO apogee)
    x_ap, y_ap, vx_ap, vy_ap, m_ap = LT_traj[0,-1], LT_traj[1,-1], LT_traj[2,-1], LT_traj[3,-1], LT_traj[4,-1]

    # Step 2: High-thrust circularization burn at GTO apogee to GEO
    v_circ_geo = np.sqrt(mu / r_GEO)
    vvec_ap = np.array([vx_ap, vy_ap])
    vvec_geo = np.array([0.0, v_circ_geo])  # GEO is circular, prograde y only

    dV_vec = vvec_geo - vvec_ap
    dV_mag = np.linalg.norm(dV_vec)
    m_final = m_ap * np.exp(-dV_mag / (Isp_high * g0))
    delta_m_HT = m_ap - m_final#mass of propellant used during the high-thrust (HT) circularization burn at GTO apogee

    # Total mass change: loss during low-thrust + final high-thrust
    total_mass_change = m0 - m_final
    return total_mass_change
    


def residuals(p,r0, m0, T, Isp_low, Isp_high, mu, r_GEO=42164):
    # P is is free vector
    """
    Constraint vector for optimizer: 
    1. End of low-thrust trajectory matches GEO apogee (position, velocity).
    2. After high-thrust burn at apogee, orbit is circular GEO.
    """
    # Unpack optimization/free variables
    vx0, vy0, tof_lt = p[:3]

    # Initial position/velocity in LEO (x-axis, prograde y-velocity)
    init_pos_LEO = np.array([r0, 0.0])
    init_vel_LEO = np.array([vx0, vy0])

    # Step 1: Propagate low-thrust segment LEO→GTO apogee
    LT_traj, times = low_thrust_propagator_2D(init_pos_LEO, init_vel_LEO, tof_lt, 1000, Isp_low, m0, T, r_GEO)
    x_ap, y_ap, vx_ap, vy_ap, m_ap = LT_traj[0,-1], LT_traj[1,-1], LT_traj[2,-1], LT_traj[3,-1], LT_traj[4,-1]

    # Step 2: Compute required high-thrust burn (ΔV) to reach circular GEO velocity
    v_circ_geo = np.sqrt(mu / r_GEO)
    vvec_ap = np.array([vx_ap, vy_ap])
    vvec_geo = np.array([0.0, v_circ_geo])
    dV_vec = vvec_geo - vvec_ap
    v_after_circ = vvec_ap + dV_vec   # Should match vvec_geo exactly

    # Constraints (residuals):
    #   - Final low-thrust position x = r_GEO, y ≈ 0 (apogee at x-axis)
    #   - Velocity after burn = [0, v_circ_geo] (pure prograde, circular)
    #   - Velocity before burn x-component ≈ 0 (for a clean final burn)
    F = np.array([
        x_ap - r_GEO,         # At GEO apogee in x
        y_ap,                 # On x-axis
        v_after_circ[0],      # vx should be zero after burn
        v_after_circ[1] - v_circ_geo  # vy should match GEO speed after burn
    ])

    # For debug
    print(f"GTO Apogee Pos: ({x_ap:.2f}, {y_ap:.2f}), Vel: ({vx_ap:.4f}, {vy_ap:.4f}), Mass: {m_ap:.3f}")
    print(f"Needed GEO burn ΔV: {np.linalg.norm(dV_vec):.4f} km/s")
    print(f"Final vx after burn: {v_after_circ[0]:.4e}, vy after: {v_after_circ[1]:.4e} (target: 0, {v_circ_geo:.4e})")

    return F

def jacobian(p,r0, m0, T, Isp_low, Isp_high, mu, r_GEO=42164):
    """
    Calculate the Jacobian matrix
    This jacobian should be the deriv of F wrt free variables
    """
    vx0, vy0, tof, DVx, DVy  = p
    state0 = np.hstack((r0, [vx0, vy0, m0]))

    init_pos_LEO = np.array([r0, 0.0])
    init_vel_LEO = np.array([vx0, vy0])

    traj, _ = low_thrust_propagator_2D(init_pos_LEO, init_vel_LEO, tof_lt, 1000, Isp_low, m0, T, r_GEO)

    # pull STM at t_f
    Phi = traj[5:, -1].reshape(5, 5)         # rows 5-29 are the 25 STM elements
    Phi_rv = Phi[0:2, 2:4]                   # d(r_f)/d(v0)
    Phi_vv = Phi[2:4, 2:4]                   # d(v_f)/d(v0)

    # a_f for TOF column
    x_f, y_f, vx_f, vy_f = traj[:4, -1]
    r_f = np.hypot(x_f, y_f)
    m_f = traj[4, -1]
    a_f  = -mu*np.array([x_f, y_f])/r_f**3 + (T/m_f)*np.array([vx_f, vy_f])/np.hypot(vx_f, vy_f)

    # Circularization at apogee: ΔV
    v_circ_geo = np.sqrt(mu / r_GEO)
    dV_vec = np.array([0.0, v_circ_geo]) - np.array([vx_f, vy_f])

    J = np.zeros((4, 5))
    # 1. x_ap - r_GEO constraints
    J[0, 0:2] = Phi[0, 2:4]      # Sensitivity to vx0, vy0
    J[0, 2] = vx_f               # Sensitivity to tof

    # 2. y_ap
    J[1, 0:2] = Phi[1, 2:4]
    J[1, 2] = vy_f

    # 3. vx_after_circ = vx_f + dV_vec[0]
    J[2, 0:2] = Phi[2, 2:4]
    J[2, 2] = a_f[0]
    # Derivative of delta-v wrt vx_f is -1, so:
    J[2, 0] -= 1.0  # Partial wrt vx0
    # No sensitivity to vy0 or tof in this partial

    # 4. vy_after_circ - v_circ_geo
    J[3, 0:2] = Phi[3, 2:4]
    J[3, 2] = a_f[1]
    # Derivative of delta-v wrt vy_f is -1:
    J[3, 1] -= 1.0  # Partial wrt vy0

   
    return J

def constraint_fun(p, r_LEO, m0, T, Isp_low, Isp_high, mu, final_GTO_pos, final_GTO_vel):#should this be for both the High Thrust and the Low thrust
    """Return the 4-vector F that must be zero at the optimum."""
    r0 = 7000
    T = 0.0005
    mu = 398600.0
    v_LEO = np.sqrt(mu / r_LEO)
    vx0, vy0, tof, DVx, DVy = p[:5]

    init_pos_LEO = np.array([r0, 0.0])
    init_vel_LEO = np.array([0.0, v_LEO])


    DV1=np.array([vx0-init_vel_LEO[0], vy0-init_vel_LEO[1]])
    #initial delta v
    DV1_mag = np.linalg.norm(DV1)
    g0 = 9.80665
    m1 = m0 * np.exp(-DV1_mag / (Isp_high * g0))
    return residuals(p, r0, m1, T, Isp_low, Isp_high, mu, final_GTO_pos, final_GTO_vel)

def constraint_jac(p, r0, m0, T, Isp_low, Isp_high, mu, final_GTO_pos, final_GTO_vel):
    r0 = 7000
    T = 0.0005
    mu = 398600.0
    v_LEO = np.sqrt(mu / r_LEO)
    vx0, vy0, tof, DVx, DVy = p[:5]

    init_pos_LEO = np.array([r_LEO, 0.0])
    init_vel_LEO = np.array([0.0, v_LEO])


    DV1=np.array([vx0-init_vel_LEO[0], vy0-init_vel_LEO[1]])
    #initial delta v
    DV1_mag = np.linalg.norm(DV1)
    g0 = 9.80665
    m1 = m0 * np.exp(-DV1_mag / (Isp_high * g0))
    return jacobian(p, r0, m1, T, Isp_low, Isp_high, mu, final_GTO_pos, final_GTO_vel)

def plot_hybrid_trajectory (r_LEO, r_GEO, LTtraj,LT_tof):
    
    tof_array = np.linspace(0,2*np.pi, num=1000)
    x0 = r_LEO *np.cos(tof_array)
    y0 = r_LEO * np.sin(tof_array)
    x2 = r_GEO * np.cos(tof_array) 
    y2 = r_GEO * np.sin(tof_array)

    print("LEO", r_LEO, "GEO", r_GEO)

    # GTO ellipse for reference as dashed (peri = LEO, apo = GEO)
    # Calculate eccentricity and focal distance
    a_GTO = (r_LEO + r_GEO)/2
    b_GTO = np.sqrt(r_LEO * r_GEO)
    c = a_GTO - r_LEO  # distance from Earth's center (focus) to ellipse center


    x_gto = a_GTO * np.cos(tof_array) - c
    y_gto = b_GTO * np.sin(tof_array)

# Parametric equations, centered at (−c, 0) so focus (Earth) is at (0, 0)
    

    print("LT Start:", LTtraj[:4,0])
    print("LT End:", LTtraj[:4,-1])

    #calculate 

    plt.figure(figsize=(8, 7))

    plt.plot(x0, y0, 'k--', label='LEO Orbit')
    plt.plot(x2, y2, 'b--', label='GEO Orbit')
    plt.plot(x_gto, y_gto, 'c--', linewidth=1.5, label='GTO Ellipse')

    # High-thrust arc (GTOtraj): blue solid line
    plt.plot(GTOtraj[0], GTOtraj[1], color='blue', linewidth=2, label='high-thrust')
    plt.plot(LTtraj[0], LTtraj[1], 'r', linewidth=2, label='Low-Thrust Arc')

    #plt.plot(GTOtraj[0, 0], GTOtraj[1, 0], 'go', markersize=10, label='LEO Start')
    #plt.plot(final_GTO_pos[0], final_GTO_pos[1], 'bo', markersize=10, label='GTO Apogee / LT Start')
    plt.plot(LTtraj[0, -1], LTtraj[1, -1], 'mo', markersize=10, label='End Burn (Low Thrust)')

    plt.scatter(0, 0, color='red', s=100, label='Centeral body')

    plt.xlabel('X (km)')
    plt.ylabel('Y (km)')
    plt.axis('equal')
    plt.grid(True)
    plt.title('Hybrid Impulsive/Low-Thrust Transfer')
    plt.legend(loc='best')
    plt.show()

def plot_mass_overtime(m0, m_after_impulse, LTtraj, LT_times, delta_v1_time=0):
    """""
    Plot spacecraft mass over the entire mission:
       Instant drop for first impulsive burn,
       Continuous decrease for low-thrust segment.
       Time axis in days.

    Parameters:
    m0: 
        Initial mass before first burn (kg)
    m_after_impulse: 
        Mass immediately after impulsive burn (kg)
    LTtraj: ndarray
        Trajectory output from your low-thrust propagator (row 5 is mass)
    LT_times: ndarray
        Time vector (seconds) matching columns of LTtraj
    delta_v1_time: 
        Time (seconds) at which impulsive burn occurs; typically 0, else can provide for future completeness.
    """""

    # Segment 1: from t=0 to first burn
    t_burn = delta_v1_time / 86400  # convert to days (usually 0)
    t_LT_days = LT_times / 86400.0

    # Assemble full timeline and mass
    times_days = np.concatenate([ t_LT_days + t_burn])
    mass_profile = np.concatenate([LTtraj[4, :]])

    fig = plt.figure()
    ax = plt.axes()
    ax.plot(times_days, mass_profile, label='Total mass (kg)', color='b')
    ax.set_xlabel('Time (days)')
    ax.set_ylabel('Spacecraft Mass (kg)')
    plt.title('Mission Mass: Impulsive Burn + Low-Thrust Transfer')

    # Annotate burns
    plt.axvline(x=t_burn, color='red', linestyle='--', alpha=0.7, label='Impulsive Burn')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()

    # print total mass used for both phases
    print(f"Impulsive burn mass loss: {m0-m1:.3f} kg")
    print(f"Low-thrust burn mass loss: {m1-LTtraj[4,-1]:.3f} kg")
    print(f"Total mass used: {m0-LTtraj[4,-1]:.3f} kg")

    radius = np.sqrt(LTtraj[0,:]**2 + LTtraj[1,:]**2)
    plt.plot(LT_times, radius)
    plt.xlabel("Time (s)")
    plt.ylabel("Radius (km)")
    plt.title("Radius over Low-Thrust Phase")
    plt.show()


r_LEO = 6378+622
r_GEO = 42164
m0=500
Isp_low = 1500
Isp_high = 350
mu=398600.0
thrust = 0.0005

#target, times = keplerian_propagator(initr2, initv2,2*np.pi*np.sqrt(r2**3/grav), integration_steps)
#initial orbit values
v_LEO = np.sqrt(mu / r_LEO)
init_pos_LEO = np.array([r_LEO, 0.0])
init_vel_LEO = np.array([0.0, v_LEO])


# --- Free variables for optimizer ---
x0, y0 = r_LEO, 0
vx0, vy0 = 0, v_LEO
xf, yf = r_GEO, 0
vx_f, vy_f = 0, np.sqrt(mu / r_GEO)
tof_guess = (np.pi * np.sqrt(((r_LEO + r_GEO)/2)**3 / mu))  # Hohmann transfer time as a rough guess
DVx, DVy = 0, 0  # for pure transfer

# Use the high-thrust targeter for initial guess
initial_guess = high_thrust_targeter(x0, y0, vx0, vy0, DVx, DVy, xf, yf, vx_f, vy_f, tof_guess)#should shooting method only be use for high thrust part?
# --- Optimize ---
sol = optimize_transfer(
    initial_guess, r_LEO, m0, thrust, Isp_low, Isp_high, mu, r_GEO
)
print(sol)

vx0, vy0, tof_lt = sol.x
g0 = 9.80665

# Low-thrust propagation: LEO to GTO apogee
LT_traj, LT_times = low_thrust_propagator_2D(
    init_pos_LEO, np.array([vx0, vy0]), tof_lt, 1000, Isp_low, m0, thrust, r_GEO
)
x_ap, y_ap, vx_ap, vy_ap, m_ap = LT_traj[0,-1], LT_traj[1,-1], LT_traj[2,-1], LT_traj[3,-1], LT_traj[4,-1]



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
#delta_v_GTO = v_periapse_GTO - v_LEO
#g0 = 9.80665
#m_after_GTO = m0 * np.exp(-delta_v_GTO / (Isp_high * g0))

#vTransferApo = np.sqrt(mu*(2/r_LEO-1/a_GTO))


#deltav calculations

#delta_v1 = v_periapse_GTO - v_LEO
#delta_v2 = v2 - vTransferApo


v_LEO = np.sqrt(mu / r_LEO)       # km/s
v_LEO_mps = v_LEO * 1000          # convert to m/s

m_after_LEO = m0 * np.exp(-v_LEO_mps / (Isp_high * g0))

v_GEO = np.sqrt(mu / r_GEO)
a_thrust = (T * 1000) / m0  # convert kN to N

tof_HT = (v_GEO - v_LEO) / a_thrust  # seconds


free_vector = high_thrust_targeter (-r_Earth, 0, 0, 0, r_LEO, 0, 0, v_LEO,0, v_LEO, tof_HT)

#initial_guess = [vx0, vy0, tof0, DVx0, DVy0]
initial_guess = free_vector

# Changing guess
initial_guess[1] -= 0.5
initial_guess[4] += 0.5


#optimize
sol=optimize_transfer(initial_guess, r_LEO, m0, thrust, Isp_low, Isp_high, mu,init_pos_LEO, init_vel_LEO)
print(sol)

vx0, vy0, tof, DVx, DVy = sol.x

DV1=np.array([vx0-init_vel_LEO[0], vy0-init_vel_LEO[1]])
DV1_mag = np.linalg.norm(DV1)
g0 = 9.80665
m1 = m0 * np.exp(-DV1_mag / (Isp_high * g0))

r0_vec =  [0, 0]             # Starting position vector (x, y)
v0_vec =  [vx0, vy0]      # Velocity after impulsive burn

LT_tof = 3000000 # get optimized TOF from solver

print("Low-Thrust Time of Flight (s):", LT_tof)

LT_traj, LT_times = low_thrust_propagator_2D(init_pos_LEO, init_vel_LEO, LT_tof, 1000, Isp_low, m0, thrust, r_GEO)
HTtraj, times = keplerian_propagator(r0_vec, v0_vec, tof, 1000)
#TOF should be optimized
#GTO_traj, times = keplerian_propagator(r0_vec, [0.0,v_periapse_GTO], tof, 1000)

print("Final LEO Position (km):", init_pos_LEO)
print("Final LEO Velocity (km/s):", init_vel_LEO)

LT_traj, LT_times = low_thrust_propagator_2D(init_pos_LEO,init_vel_LEO, LT_tof, 1000, Isp_low, m_after_LEO, thrust, r_GEO)#TOF should be optimized


"""""
print("GTO_traj sample:", GTO_traj[0,:5], GTO_traj[1,:5])
print("LT_traj sample:", LT_traj[0,:5], LT_traj[1,:5])
print("LT_traj end:", LT_traj[0, -5:], LT_traj[1, -5:])
"""""
plot_hybrid_trajectory(r_LEO, r_GEO, LT_traj, HTtraj)
plot_mass_overtime(m0, m_after_LEO, LT_traj, LT_times)
plt.show()


