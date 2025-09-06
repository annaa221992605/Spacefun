import numpy as np
from scipy.integrate import solve_ivp
from thrust_angle2 import steering_vector, compute_orbital_elements

def low_thrust_propagator_2D(init_r, init_v, tof, steps, isp, m0, T,r_GEO ):
    """
    Function to propagate a given orbit
    """
    # Time vector
    tspan = [0, tof]

    # Array of time values
    tof_array = np.linspace(0,tof, num=steps)
    init_stm = np.eye(5)
    #init_state = np.append(init_r,np.append(init_v,m0))
    init_state = np.concatenate((init_r,np.append(init_v,m0),init_stm.flatten()))
    # Do the integration
    sol = solve_ivp(fun = lambda t,x:low_thrust_eoms_STM(t,x,isp, T), t_span=tspan, y0=init_state, method="DOP853", t_eval=tof_array, rtol = 1e-8, atol = 1e-8)

    # Return everything
    return sol.y, sol.t

def low_thrust_eoms_STM(t, state, isp, T):

    a0 = 24396       # Initial GTO a [km]
    e0 = 0.7283      # Initial GTO e
    a_target = 42164
    e_target = 0

    mu = 398600.441500000
    g0 = 9.80665 #m/s

    # Use actual osculating a0, e0 for each call for optimal steering
    a0, e0, _ = compute_orbital_elements(state[:2], state[2:4], mu)
    a_target = 42164.0
    e_target = 0.0
    
    
    # Extract state from init
    x, y, vx, vy, mass = state[:5]
    r_vec = np.array([x, y])
    v_vec = np.array([vx, vy])
    r = np.linalg.norm(r_vec)
    v = np.linalg.norm(v_vec)

    # steering law
    thrust_unit = steering_vector(state, mu, a_target, e_target, a0, e0)

    """
    Equation of motion for 2body orbits
    """
    
    

    

    #thrust_dir = v_dot

    # Compute current eccentricity
    #e = compute_eccentricity(r_vec, v_vec, mu)

    # Compute thrust direction using Pollard law (tangent/radial mix)
    #thrust_dir = pollard_law(r_vec, v_vec, e)
    #thrust_dir = thrust_dir / np.linalg.norm(thrust_dir)   # make sure it's a unit vector


    # w: smooth transition from 0 (all thrust in v_hat) to 1 (all thrust in -v_hat) as r approaches r_GEO
    #if r < r_GEO:
    # --- Compute thrust acceleration ---
    accel_mag = T / mass
    ax_pert = accel_mag * thrust_unit[0]
    ay_pert = accel_mag * thrust_unit[1]

    # --- Gravity acceleration ---
    ax_grav = -mu * x / r**3
    ay_grav = -mu * y / r**3

    # --- Maneuver efficiency window ---
    # True anomaly
    nu = np.arctan2(y, x)
    # Eq (Table 4 in paper): a, e efficiency
    eff_a = 1.0 / (1.0 + e0 * np.cos(nu)) if np.abs(e0) > 1e-6 else 1.0
    eff_e = np.abs(np.sin(nu)) if np.abs(e0) > 1e-6 else 1.0
    maneuver_eff = min(eff_a, eff_e)
    threshold = 0.5  # Try 0.4–0.7 for study; 0.5 is a good default

    if maneuver_eff < threshold:
        ax_pert = 0.0
        ay_pert = 0.0

    # --- Final derivatives ---
    ax = ax_grav + ax_pert
    ay = ay_grav + ay_pert
    mdot = -T / (isp * g0) if (ax_pert != 0 or ay_pert != 0) else 0.0

       
    gamma = T / mass

    
    # ---- Jacobian ---------------------------------------------------------
    A = np.zeros((5, 5))
    A[0:2, 2:4] = np.eye(2)                      # kinematics

    # gravity derivatives
    A[2,0] = -mu/r**3 * (1 - 3*x**2/r**2)
    A[2,1] =  3*mu*x*y / r**5
    A[3,0] =  3*mu*y*x / r**5
    A[3,1] = -mu/r**3 * (1 - 3*y**2/r**2)

    # thrust wrt velocity
    if v > 1e-8:
        A[2,2] =  gamma * vy**2 / v**3
        A[2,3] = -gamma * vx*vy / v**3
        A[3,2] = -gamma * vx*vy / v**3
        A[3,3] =  gamma * vx**2 / v**3
        A[2,4] = -gamma/mass * vx / v
        A[3,4] = -gamma/mass * vy / v
    # A[4,*] is already zero

    # ---- STM dot ----------------------------------------------------------
    Phi = state[5:].reshape(5, 5)
    Phi_dot = A @ Phi

    xdot = np.array([vx, vy, ax, ay, mdot])
    return np.hstack((xdot, Phi_dot.flatten()))