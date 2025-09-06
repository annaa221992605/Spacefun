import numpy as np
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt

def high_thrust_targeter(x0, y0, xdot0, ydot0, DVx, DVy, xf, yf, xdotf, ydotf,tof):
    """
    Function to target final state based on initial guess
    """
    mu = 398600.441500000
    converged = 0
    init_state = [x0, y0, xdot0, ydot0]
    free_vector = [xdot0, ydot0, tof, DVx, DVy]
    integration_steps = 500
    while converged == 0:
        # propagate initial state
        traj1, times =  keplerian_propagator_with_STM(init_state[0:2], free_vector[0:2], free_vector[2], integration_steps)

        traj1 = np.array(traj1)


        if traj1.ndim == 1:
        # Assume data should be shaped (Nrows, Ncols=1)
            traj1 = traj1.reshape((-1, 1))

        final_x = traj1[0,-1]
        final_y = traj1[1,-1]
        final_xdot = traj1[2,-1]
        final_ydot = traj1[3,-1]
        
        # Plot it
        """
        fig = plt.figure()
        # Define axes in that figure
        ax = plt.axes(projection='3d',computed_zorder=False)
        #initial
        ax.plot(traj1[0],traj1[1],traj1[2],zorder=5)
        plt.title("Hohmann Transfer")
        ax.set_xlabel("X-axis (km)")
        ax.set_ylabel("Y-axis (km)")
        ax.set_zlabel("Z-axis (km)")
        ax.xaxis.set_tick_params(labelsize=7)
        ax.yaxis.set_tick_params(labelsize=7)
        ax.zaxis.set_tick_params(labelsize=7)
        ax.set_aspect('equal', adjustable='box')
        plt.show()
        """
        
        dim = 4
        Phi_flat_final = traj1[dim:, -1]        # rows 4 … end, last time step
        stm = Phi_flat_final.reshape(4,4)

        actual_final_xdot = final_xdot + free_vector[3]
        actual_final_ydot = final_ydot + free_vector[4]

        # Need final acceleration
        r_f = np.hypot(final_x, final_y)
        ax_f, ay_f = -mu * np.array([final_x, final_y]) / r_f**3

        F = [final_x-xf, final_y-yf, actual_final_xdot - xdotf, actual_final_ydot - ydotf]
        print(np.linalg.norm(F))
        if np.linalg.norm(F) > 1e-8:
            # FX = deriv of F wrt Free
            FX = np.zeros((4, 5))
            # deriv of first row
            FX[0, 0:2] = stm[0,2:4]
            FX[0,2] = final_xdot
            # End of row is zeros
            # Second Row
            FX[1,0:2] = stm[1,2:4]
            FX[1,2] = final_ydot
            # Third Row
            FX[2,0:2] = stm[2,2:4]
            FX[2,2] = ax_f
            FX[2,3] = 1.0
            # Fourth Row
            FX[3,0:2] = stm[3,2:4]
            FX[3,2] = ay_f
            FX[3,4] = 1.0

            update = -np.linalg.lstsq(FX, F, rcond=None)[0]
            free_vector = free_vector + update

        else:
            converged = 1

    return free_vector


def keplerian_propagator_with_STM(init_r, init_v, tof, steps):
    """
    Function to propagate a given orbit
    """
    # Time vector
    tspan = [0, tof]

    # Array of time values
    tof_array = np.linspace(0,tof, num=steps)
    init_stm = np.eye(4)
    init_state = np.concatenate((init_r,init_v,init_stm.flatten()))
    # Do the integration
    sol = solve_ivp(fun = lambda t,x:keplerian_eoms_2D(t,x), t_span=tspan, y0=init_state, method="DOP853", t_eval=tof_array, rtol = 1e-12, atol = 1e-12)

    # Return everything
    return sol.y, sol.t

def keplerian_eoms_2D(t, state):
    """
    Equation of motion for 2body orbits
    """
    mu = 398600.441500000
    # Extract values from init
    x, y, vx, vy = state[0:4]
    r_dot = np.array([vx, vy])
    
    # Define r
    r = np.linalg.norm([x, y])
    # Solve for the acceleration
    ax = - (mu/r**3) * x
    ay = - (mu/r**3) * y

    v_dot = np.array([ax, ay])
    # x Derivs
    Jxx = -(mu/r**3)*(1-(3*x**2)/(r**2))
    Jxy = (3*mu*x*y)/r**5
    # Y Derivs
    Jyx = (3*mu*x*y)/r**5
    Jyy = -(mu/r**3)*(1-(3*y**2)/(r**2))

    # Make A
    A = np.zeros((4, 4))
    
    # upper-right: ∂ṙ/∂v = I₃
    A[0:2, 2:4] = np.eye(2)

    A[2, 0] = Jxx       # ∂a_x/∂x
    A[2, 1] = Jxy       # ∂a_x/∂y

    A[3, 0] = Jyx       # ∂a_y/∂x
    A[3, 1] = Jyy       # ∂a_y/∂y
    
    Phi = state[4:].reshape(4, 4)
    Phi_dot = A @ Phi
    xdot = np.concatenate((r_dot, v_dot))
    dx = np.concatenate((xdot, Phi_dot.flatten()))

    return dx

