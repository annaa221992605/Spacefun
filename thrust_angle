import numpy as np

def pollard_law(r, v, e, beta_max=np.radians(60), e_decay=0.2):
    """Pollard-style β law for thrust direction.
    r: position vector
    v: velocity vector
    e: current eccentricity
    Returns: thrust direction unit vector
    """
    v_hat = v / np.linalg.norm(v)
    r_hat = r / np.linalg.norm(r)
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