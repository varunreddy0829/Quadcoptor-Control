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
from scipy.linalg import solve_continuous_are
from scipy.signal import cont2discrete
from sim.quadrotor import QuadrotorParams


# ── Shared actuator limits [delta_T, tau_x, tau_y, tau_z] ────────────────────
U_MIN = np.array([-3.0, -1.5, -0.5, -0.1])
U_MAX = np.array([ 3.0,  1.5,  0.5,  0.1])


def _build_linearisation(params: QuadrotorParams) -> tuple:
    """
    Build continuous-time A, B matrices linearised at hover.

    state:  [x, y, z, vx, vy, vz, phi, theta, psi, p, q, r]
    input:  [delta_T, tau_x, tau_y, tau_z]

    Small angle approximation at hover:
      vx_dot ≈  g * theta
      vy_dot ≈ -g * phi
      vz_dot =  delta_T / m
      p_dot  =  tau_x / Ixx
      q_dot  =  tau_y / Iyy
      r_dot  =  tau_z / Izz

    Returns:
        A_c : continuous-time state matrix  (12, 12)
        B_c : continuous-time input matrix  (12, 4)
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
    """
    Return PID controller configuration.

    Returns:
        dict with keys: Kp_pos, Ki_pos, Kp_vel, Ki_vel,
                        Kp_att, Kd_att, max_tilt, max_integral,
                        u_min, u_max
    """
    return {
        # Position loop gains [x, y, z]
        'Kp_pos': np.array([1.5, 1.5, 2.0]),
        'Ki_pos': np.array([0.0, 0.0, 0.1]),

        # Velocity loop gains [x, y, z]
        'Kp_vel': np.array([3.0, 3.0, 4.0]),
        'Ki_vel': np.array([0.1, 0.1, 0.2]),

        # Attitude loop gains [phi, theta, psi]
        'Kp_att': np.array([10.0, 10.0, 2.0]),
        'Kd_att': np.array([ 1.0,  1.0, 0.5]),

        # Safety limits
        'max_tilt':     0.5,   # max tilt angle [rad] (~30 deg)
        'max_integral': 5.0,   # anti-windup clamp

        # Shared actuator limits
        'u_min': U_MIN,
        'u_max': U_MAX,
    }


def get_lqr_config(params: QuadrotorParams = None) -> dict:
    """
    Build and return LQR configuration.

    Solves the continuous algebraic Riccati equation to get optimal K.

    Returns:
        dict with keys: A, B, K, Q, R, u_min, u_max
    """
    params = params or QuadrotorParams()
    A_c, B_c = _build_linearisation(params)

    # Q: penalise state error [x,y,z, vx,vy,vz, phi,theta,psi, p,q,r]
    q_diag = np.array([
        20, 20, 30,   # position
         5,  5,  5,   # velocity
         5,  5,  5,   # orientation
         1,  1,  1    # body rates
    ])
    Q = np.diag(q_diag)

    # R: penalise virtual input effort [delta_T, tau_x, tau_y, tau_z]
    R = np.diag([0.1, 0.8, 0.8, 0.8])

    # Solve CARE → optimal gain matrix K
    P = solve_continuous_are(A_c, B_c, Q, R)
    K = np.linalg.inv(R) @ B_c.T @ P

    return {
        'A':     A_c,
        'B':     B_c,
        'K':     K,
        'Q':     Q,
        'R':     R,
        'u_min': U_MIN,
        'u_max': U_MAX,
    }
