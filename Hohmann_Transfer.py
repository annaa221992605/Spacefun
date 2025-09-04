import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
from shooting import high_thrust_targeter

def hohmannTransfer(r1, r2, grav = 398600.4418):
    # Main function
    #need:deltaV's and ToF's
    print("Starting Main Function")

    """""
    initial_position = np.array([7000,0,0])
    initial_velocity = np.array([0, 7.72, 5])
    integration_time = 24*60*60
    integration_steps = 1000                                                                   

    trajectory, times = keplerian_propagator(initial_position, initial_velocity, integration_time, integration_steps)
    """
    integration_steps = 1000 
    
    #initial orbit values
    v1 = np.sqrt(grav/r1)

    #desired orbit velocity
    v2 = np.sqrt(grav/r2)

    #transfer orbit calculations
    semiaxis = (r1+r2)/2
    vTransferPeri = np.sqrt(grav*(2/r1 - 1/semiaxis))
    vTransferApo = np.sqrt(grav*(2/r2-1/semiaxis))
    Tof = np.pi*np.sqrt(semiaxis**3/grav)

    #deltav calculations

    delta_v1 = vTransferPeri - v1
    delta_v2 = v2 - vTransferApo
    totalDeltaV = delta_v1+delta_v2
    print("---------------------------")
    print("DV 1 = ", delta_v1)
    print("DV 2 = ", delta_v2)
    print("---------------------------")
    #initial orbit propagrator

    initr1=np.array([r1, 0])
    initv1=np.array([0, v1])

    isp = 19000
    m0 = 4.800
    #transfer Orbits initial values
    init_r_transfer = np.array([r1, 0])
    init_v_transfer = np.array([0, vTransferPeri])

    # Low thrust propagator
    low_thrust_traj, times = low_thrust_propagator(init_r_transfer, init_v_transfer-[0.0,0.5], 10*Tof, integration_steps, isp, m0)

    # Pass initial guess to targetter
    # high_thrust_targeter(x0, y0, xdot0, ydot0, DVx, DVy, xf, yf, xdotf, ydotf,tof)
    """
    free_vector = high_thrust_targeter(r1, 0, 0, v1+delta_v1, 0, -delta_v2, -r2, 0, 0, -v2, Tof)
    print("Diff in initial x dot ", 0-free_vector[0])
    print("Diff in initial y dot ", v1+delta_v1-free_vector[1])
    print("Diff in initial time ", Tof-free_vector[2])
    print("Diff in final DV X ", 0-free_vector[3])
    print("Diff in final DV y ", -delta_v2-free_vector[4])
    """
    # Init Orbit
    traj1, times = keplerian_propagator(initr1, initv1, 2*np.pi*np.sqrt(r1**3/grav), integration_steps)

    # Transfer
    traj_transfer, times = keplerian_propagator(init_r_transfer, init_v_transfer,Tof, integration_steps)

    #target circular orbit propagation

    initr2=np.array([-r2, 0])
    initv2 = np.array([0, v2])
    # Target Orbit
    traj2, times = keplerian_propagator(initr2, initv2,2*np.pi*np.sqrt(r2**3/grav), integration_steps)

    fig, ax = plt.subplots()
    ax.plot(traj1, traj1[1])
    ax.plot(traj_transfer, traj_transfer[1])
    ax.plot(low_thrust_traj, low_thrust_traj[1])
    ax.plot(traj2, traj2[1])
    #ax.scatter(, , color='red', s=100, label='Central body')
    ax.set_xlabel("X-axis (km)")
    ax.set_ylabel("Y-axis (km)")
    plt.title("Hohmann Transfer (2D)")
    plt.show()
    """""
    # Plot it
    fig = plt.figure()
    # Define axes in that figure
    ax = plt.axes(projection='3d',computed_zorder=False)

    #old:
    # Plot x, y, z
    #ax.plot(trajectory[0],trajectory[1],trajectory[2],zorder=5)


    #initial
    ax.plot(traj1[0],traj1[1],traj1[2],zorder=5)
    #transfer
    ax.plot(traj_transfer[0],traj_transfer[1],traj_transfer[2],zorder=5)
    # Low thrust traj
    ax.plot(low_thrust_traj[0],low_thrust_traj[1],low_thrust_traj[2],zorder=5)
    #desired
    ax.plot(traj2[0],traj2[1],traj2[2],zorder=5)

    #marking  center
    ax.scatter([0], [0], [0], color='red', s=100, label='Centeral body')


    plt.title("Hohmann Transfer")
    ax.set_xlabel("X-axis (km)")
    ax.set_ylabel("Y-axis (km)")
    ax.set_zlabel("Z-axis (km)")
    ax.xaxis.set_tick_params(labelsize=7)
    ax.yaxis.set_tick_params(labelsize=7)
    #ax.zaxis.set_tick_params(labelsize=7)
    ax.set_aspect('equal', adjustable='box')
    plt.show()
    """

def low_thrust_propagator(init_r, init_v, tof, steps, isp, m0):
    """
    Function to propagate a given orbit
    """
    # Time vector
    tspan = [0, tof]

    # Array of time values
    tof_array = np.linspace(0,tof, num=steps)
    init_state = np.append(init_r,np.append(init_v,m0))
    # Do the integration
    sol = solve_ivp(fun = lambda t,x:low_thrust_eoms(t,x,isp), t_span=tspan, y0=init_state, method="DOP853", t_eval=tof_array, rtol = 1e-12, atol = 1e-12)

    # Return everything
    return sol.y, sol.t

def low_thrust_eoms(t, state, isp):
    """
    Equation of motion for 2body orbits
    """
    earth_nu = 398600.441500000
    g0 = 9.81
    # Extract values from init
    x, y, vx, vy, mass = state[:5]
    r_dot = np.array([vx, vy])
    
    # Define r
    r = np.linalg.norm([x, y])

    force = 1
    accel_mag = (force/mass) * 1e-3

    # Solve for the acceleration
    ax = - (earth_nu/r**3) * x + (vx/np.linalg.norm(r_dot))*accel_mag
    ay = - (earth_nu/r**3) * y + (vy/np.linalg.norm(r_dot))*accel_mag

    m_dot = -force/(isp*g0)

    v_dot = np.array([ax, ay])

    dx = np.append(r_dot, np.append(v_dot, m_dot))

    return dx


def keplerian_propagator(init_r, init_v, tof, steps):
    """
    Function to propagate a given orbit
    """
    # Time vector
    tspan = [0, tof]

    # Array of time values
    tof_array = np.linspace(0,tof, num=steps) #number of times all the variable is calculated=1000
    init_state = np.concatenate((init_r,init_v))
    # Do the integration
    sol = solve_ivp(fun = lambda t,x:keplerian_eoms(t,x), t_span=tspan, y0=init_state, method="DOP853", t_eval=tof_array, rtol = 1e-12, atol = 1e-12)

    # Return everything
    return sol.y, sol.t


def keplerian_eoms(t, state):
    """
    Equation of motion for 2body orbits
    """
    earth_nu = 398600.441500000
    # Extract values from init
    x, y, vx, vy = state
    r_dot = np.array([vx, vy])
    
    # Define r
    r = np.linalg.norm([x, y])
    # Solve for the acceleration
    ax = - (earth_nu/r**3) * x
    ay = - (earth_nu/r**3) * y

    v_dot = np.array([ax, ay])

    dx = np.append(r_dot, v_dot)

    return dx


if __name__ == '__main__':
    hohmannTransfer(8000, 12000)