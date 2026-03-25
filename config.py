"""
================================================================================
 Unified Controller Configuration
================================================================================

Single source of truth for ALL controller parameters.
PID, LQR, and MPC all read from this file.

ACTUATOR LIMITS (shared across all controllers):
  Derived from LQR diagnostic on helix + wind:
    delta_T : [-1.265, 2.326] N   → [-3.0, 3.0] with margin
    tau_x   : [-1.020, 1.020] N·m → [-1.5, 1.5] with margin
    tau_y   : [-0.082, 0.047] N·m → [-0.5, 0.5] with margin
    tau_z   : [-0.007, 0.006] N·m → [-0.1, 0.1] with margin

HOW TO TUNE:
  - Adjust Q, R to change tracking aggressiveness vs control effort
  - Adjust PID gains to tune cascade loops
  - Adjust U_MIN/U_MAX to change actuator limits (applies to ALL controllers)
  - Run run_benchmark.py to compare
================================================================================
"""

import numpy as np
from scipy.linalg import solve_discrete_are
from scipy.signal import cont2discrete
from sim.quadrotor import QuadrotorParams


# ── Shared actuator limits [delta_T, tau_x, tau_y, tau_z] ────────────────────
U_MIN = np.array([-2.0, -1, -0.5, -0.1])
U_MAX = np.array([ 2.0,  1,  0.5,  0.1])


def _build_linearisation(params: QuadrotorParams) -> tuple:
    """
    Build continuous-time A, B matrices linearised at hover.

    state:  [x, y, z, vx, vy, vz, phi, theta, psi, p, q, r]
    input:  [delta_T, tau_x, tau_y, tau_z]
    """
    g   = params.g
    m   = params.mass
    Ixx = params.Ixx
    Iyy = params.Iyy
    Izz = params.Izz

    A_c = np.zeros((12, 12))
    A_c[0, 3]  = 1.0   # x_dot     = vx
    A_c[1, 4]  = 1.0   # y_dot     = vy
    A_c[2, 5]  = 1.0   # z_dot     = vz
    A_c[3, 7]  =  g    # vx_dot   ≈  g * theta
    A_c[4, 6]  = -g    # vy_dot   ≈ -g * phi
    A_c[6, 9]  = 1.0   # phi_dot   = p
    A_c[7, 10] = 1.0   # theta_dot = q
    A_c[8, 11] = 1.0   # psi_dot   = r

    B_c = np.zeros((12, 4))
    B_c[5,  0] = 1.0 / m    # delta_T → vz_dot
    B_c[9,  1] = 1.0 / Ixx  # tau_x   → p_dot
    B_c[10, 2] = 1.0 / Iyy  # tau_y   → q_dot
    B_c[11, 3] = 1.0 / Izz  # tau_z   → r_dot

    return A_c, B_c


def get_pid_config(params: QuadrotorParams = None) -> dict:
    """Return PID controller configuration."""
    return {
        'Kp_pos': np.array([1.5, 1.5, 2.0]),
        'Ki_pos': np.array([0.0, 0.0, 0.1]),
        'Kp_vel': np.array([3.0, 3.0, 4.0]),
        'Ki_vel': np.array([0.1, 0.1, 0.2]),
        'Kp_att': np.array([10.0, 10.0, 2.0]),
        'Kd_att': np.array([ 1.0,  1.0, 0.5]),
        'max_tilt':     0.5,
        'max_integral': 5.0,
        'u_min': U_MIN,
        'u_max': U_MAX,
    }


def get_lqr_config(params: QuadrotorParams = None, dt: float = 0.01) -> dict:
    """
    Build and return Discrete LQR configuration.
    Solves the discrete algebraic Riccati equation to get optimal K.
    """
    params = params or QuadrotorParams()
    A_c, B_c = _build_linearisation(params)

    # 1. Discretise the system (Zero-Order Hold)
    sys_d = cont2discrete((A_c, B_c, np.eye(12), np.zeros((12,4))), dt, method='zoh')
    A_d, B_d = sys_d[0], sys_d[1]

    # Q: penalise state error [x,y,z, vx,vy,vz, phi,theta,psi, p,q,r]
    q_diag = np.array([
        20, 20, 30,   # position
         5,  5,  5,   # velocity
         5,  5,  5,   # orientation
         1,  1,  1    # body rates
    ])
    Q = np.diag(q_diag)
    R = np.diag([0.1, 0.8, 0.8, 0.8])

    # 2. Solve DARE to get the optimal cost-to-go matrix P
    P = solve_discrete_are(A_d, B_d, Q, R)
    
    # 3. Compute the discrete optimal gain matrix K
    K = np.linalg.inv(R + B_d.T @ P @ B_d) @ (B_d.T @ P @ A_d)

    return {
        'A':     A_d,
        'B':     B_d,
        'K':     K,
        'Q':     Q,
        'R':     R,
        'u_min': U_MIN,
        'u_max': U_MAX,
    }


def get_mpc_config(params: QuadrotorParams = None, dt: float = 0.01) -> dict:
    """
    Build and return MPC configuration, including discretised dynamics.
    """
    params = params or QuadrotorParams()
    A_c, B_c = _build_linearisation(params)
    
    # Discretise the system (Zero-Order Hold)
    sys_d = cont2discrete((A_c, B_c, np.eye(12), np.zeros((12,4))), dt, method='zoh')
    A_d, B_d = sys_d[0], sys_d[1]
    
    # Q and R weights - tuned slightly more aggressively than LQR
    q_diag = np.array([
        150, 150, 250,   # position (x, y, z)
         10,  10,  10,   # velocity (vx, vy, vz)
        20, 20, 20,   # orientation (phi, theta, psi)
         1,  1,  1    # body rates (p, q, r)
    ])
    Q = np.diag(q_diag)
    R = np.diag([0.1, 0.1, 0.1, 0.1])
    
    return {
        'Ad':    A_d,
        'Bd':    B_d,
        'Q':     Q,
        'R':     R,
        'N':     30,    # Prediction horizon (20 steps @ 0.01s = 0.2s lookahead)
        'u_min': U_MIN,
        'u_max': U_MAX,
    }