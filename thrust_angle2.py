import numpy as np

def pollard_law(r, v, e, beta_max=np.radians(60), e_decay=0.2):
    """Pollard-style law for thrust direction.
    r: position vector
    v: velocity vector
    e: current eccentricity
    Returns: thrust direction unit vector
    """
    v_hat = v / np.linalg.norm(v)
    r_hat = r / np.linalg.norm(r)
    beta = beta_max * np.exp(-e / e_decay)

    if np.isclose(np.linalg.norm(r), 42164, atol=2000):  # within 2,000 km
        return v_hat
    beta = beta_max * np.exp(-e / e_decay)

    return np.cos(beta)*v_hat + np.sin(beta)*r_hat

def compute_eccentricity(r_vec, v_vec, mu):
    # r_vec: position [x, y], v_vec: velocity [vx, vy]
    r = np.linalg.norm(r_vec)
    v = np.linalg.norm(v_vec)
    h_vec = np.cross(np.append(r_vec, 0), np.append(v_vec, 0))   # 2D cross
    h = np.linalg.norm(h_vec)
    e_vec = ( (v**2 - mu/r)*r_vec - np.dot(r_vec, v_vec)*v_vec ) / mu
    return np.linalg.norm(e_vec)

def compute_orbital_elements(r_vec, v_vec, mu):
    r_vec = np.asarray(r_vec)
    v_vec = np.asarray(v_vec)
    if r_vec.shape[-1] == 2:
        r_vec3 = np.append(r_vec, 0.0)
    else:
        r_vec3 = r_vec
    if v_vec.shape[-1] == 2:
        v_vec3 = np.append(v_vec, 0.0)
    else:
        v_vec3 = v_vec
    r = np.linalg.norm(r_vec3)
    v = np.linalg.norm(v_vec3)
    h_vec = np.cross(r_vec3, v_vec3)
    e_vec3 = (np.cross(v_vec3, h_vec) / mu) - (r_vec3 / r)
    e_vec2 = e_vec3[:2]  # TAKE ONLY XY COMPONENT
    e = np.linalg.norm(e_vec2)
    a = 1 / (2 / r - v**2 / mu)
    return a, e, e_vec2  # RETURN ONLY XY, ALWAYS

def thrust_dir_a(v_vec):
    # Tangential direction (prograde)
    return v_vec / np.linalg.norm(v_vec)

def thrust_dir_e(r_vec, v_vec, mu):
    a, e, e_vec2 = compute_orbital_elements(r_vec, v_vec, mu)
    return -e_vec2 / np.linalg.norm(e_vec2)   # negative, but now always (2,)

def steering_vector(state, mu, a_target, e_target, a0, e0):
    r_vec = state[:2]
    v_vec = state[2:4]

    a, e, e_vec = compute_orbital_elements(r_vec, v_vec, mu)

    # weights according to distance-to-target
    denom_a = abs(a_target - a0) if abs(a_target - a0) > 1e-8 else 1e-8
    wa = abs(a_target - a) / denom_a

    denom_e = abs(e_target - e0) if abs(e_target - e0) > 1e-8 else 1e-8
    we = abs(e_target - e) / denom_e

    # individual thrust optimal directions
    Ta = thrust_dir_a(v_vec)
    Te = thrust_dir_e(r_vec, v_vec, mu)

    direction = wa*Ta + we*Te
    return direction / np.linalg.norm(direction)