import numpy as np
from scipy.integrate import solve_ivp

def low_thrust_propagator_2D(init_r, init_v, tof, steps, isp, m0):
    """
    Function to propagate a given orbit
    """
    # Time vector
    tspan = [0, tof]

    # Array of time values
    tof_array = np.linspace(0,tof, num=steps)
    init_state = np.append(init_r,np.append(init_v,m0))
    # Do the integration
    sol = solve_ivp(fun = lambda t,x:low_thrust_eoms_STM(t,x,isp), t_span=tspan, y0=init_state, method="DOP853", t_eval=tof_array, rtol = 1e-12, atol = 1e-12)

    # Return everything
    return sol.y, sol.t

def low_thrust_eoms_STM(t, state, isp):
    """
    Equation of motion for 2body orbits
    """
    mu = 398600.441500000
    g0 = 9.81
    T = 1
    
    # Extract values from init
    x, y, vx, vy, m = state[:5]
    r = np.hypot(x, y)
    v = np.hypot(vx, vy)
    gamma = T / m

    # ---- acceleration -----------------------------------------------------
    ax = -mu*x/r**3 + gamma * vx / v
    ay = -mu*y/r**3 + gamma * vy / v
    mdot = -T / (isp * g0)

    # ---- Jacobian ---------------------------------------------------------
    A = np.zeros((5, 5))
    A[0:2, 2:4] = np.eye(2)                      # kinematics

    # gravity derivatives
    A[2,0] = -mu/r**3 * (1 - 3*x**2/r**2)
    A[2,1] =  3*mu*x*y / r**5
    A[3,0] =  3*mu*y*x / r**5
    A[3,1] = -mu/r**3 * (1 - 3*y**2/r**2)

    # thrust wrt velocity
    A[2,2] =  gamma * vy**2 / v**3
    A[2,3] = -gamma * vx*vy / v**3
    A[3,2] = -gamma * vx*vy / v**3
    A[3,3] =  gamma * vx**2 / v**3

    # thrust wrt mass
    A[2,4] = -gamma/m * vx / v
    A[3,4] = -gamma/m * vy / v
    # A[4,*] already zero

    # ---- STM dot ----------------------------------------------------------
    Phi = state[5:].reshape(5, 5)
    Phi_dot = A @ Phi

    xdot = np.array([vx, vy, ax, ay, mdot])
    return np.hstack((xdot, Phi_dot.flatten()))