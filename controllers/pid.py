"""
================================================================================
 Cascade PID Controller
================================================================================

3-layer cascade PID controller for quadrotor trajectory tracking.
All tunable parameters loaded from config.py via get_pid_config().

SIGNAL FLOW:
  pos error → vel target → accel target → attitude target → torques → forces
================================================================================
"""

import numpy as np
from .base import BaseController
from sim.quadrotor import QuadrotorParams


class PIDController(BaseController):
    """3-layer cascade PID. Parameters from get_pid_config()."""

    def __init__(self, cfg: dict, params: QuadrotorParams = None, dt: float = 0.01):
        """
        Args:
            cfg    : dict from get_pid_config()
            params : QuadrotorParams
            dt     : control timestep [s]
        """
        super().__init__(dt=dt)
        self.p = params or QuadrotorParams()

        self.Kp_pos = cfg['Kp_pos']
        self.Ki_pos = cfg['Ki_pos']
        self.Kp_vel = cfg['Kp_vel']
        self.Ki_vel = cfg['Ki_vel']
        self.Kp_att = cfg['Kp_att']
        self.Kd_att = cfg['Kd_att']

        self.max_tilt     = cfg['max_tilt']
        self.max_integral = cfg['max_integral']

        # Shared actuator limits [delta_T, tau_x, tau_y, tau_z]
        self.u_min = cfg['u_min']
        self.u_max = cfg['u_max']

        # Mixer inverse
        L     = self.p.arm_length
        kd_kt = self.p.k_drag / self.p.k_thrust
        B = np.array([
            [ 1,       1,       1,       1      ],
            [ L,       L,      -L,      -L      ],
            [-L,       L,       L,      -L      ],
            [-kd_kt,   kd_kt,  -kd_kt,   kd_kt  ],
        ])
        self.B_inv = np.linalg.inv(B)

        self.reset()

    def reset(self):
        super().clear_history()
        self.int_pos = np.zeros(3)
        self.int_vel = np.zeros(3)

    def compute_forces(self, state: np.ndarray, ref: np.ndarray, t: float) -> np.ndarray:
        """
        3-layer cascade control law.

        Layer 1: position error  → desired velocity
        Layer 2: velocity error  → desired acceleration
        Layer 3: acceleration    → desired attitude → torques [u2,u3,u4]
        Mixer  : [u1,u2,u3,u4]  → [f1,f2,f3,f4]
        """
        pos = state[0:3]; vel = state[3:6]
        phi, theta, psi = state[6], state[7], state[8]
        p, q, r         = state[9], state[10], state[11]

        pos_ref = ref[0:3]; vel_ref = ref[3:6]
        psi_ref = ref[8]

        # ── Layer 1: position → desired velocity ───────────────────────────
        err_pos      = pos_ref - pos
        self.int_pos = np.clip(
            self.int_pos + err_pos * self.dt,
            -self.max_integral, self.max_integral)
        target_vel   = vel_ref + self.Kp_pos * err_pos + self.Ki_pos * self.int_pos

        # ── Layer 2: velocity → desired acceleration ───────────────────────
        err_vel      = target_vel - vel
        self.int_vel = np.clip(
            self.int_vel + err_vel * self.dt,
            -self.max_integral, self.max_integral)
        target_accel = self.Kp_vel * err_vel + self.Ki_vel * self.int_vel

        # ── Layer 3: acceleration → thrust + desired angles ────────────────
        gravity = np.array([0.0, 0.0, self.p.g])
        F_des   = self.p.mass * (target_accel + gravity)

        u1        = np.linalg.norm(F_des)
        u1        = np.clip(u1, 0.0, 4.0 * self.p.f_max)
        z_b_des   = F_des / (u1 + 1e-6)
        phi_des   = np.clip(np.arcsin(np.clip(-z_b_des[1], -0.9, 0.9)),
                            -self.max_tilt, self.max_tilt)
        theta_des = np.clip(np.arctan2(z_b_des[0], z_b_des[2]),
                            -self.max_tilt, self.max_tilt)

        # ── Attitude → torques ─────────────────────────────────────────────
        err_phi   = phi_des - phi
        err_theta = theta_des - theta
        err_psi   = (psi_ref - psi + np.pi) % (2 * np.pi) - np.pi

        u2 = self.Kp_att[0] * err_phi   - self.Kd_att[0] * p
        u3 = self.Kp_att[1] * err_theta - self.Kd_att[1] * q
        u4 = self.Kp_att[2] * err_psi   - self.Kd_att[2] * r

        # ── Clip virtual inputs to shared actuator limits ──────────────────
        hover = self.p.mass * self.p.g
        u1    = hover + np.clip(u1 - hover, self.u_min[0], self.u_max[0])
        u2    = np.clip(u2, self.u_min[1], self.u_max[1])
        u3    = np.clip(u3, self.u_min[2], self.u_max[2])
        u4    = np.clip(u4, self.u_min[3], self.u_max[3])

        # ── Mixer: virtual inputs → rotor forces ───────────────────────────
        u      = np.array([u1, u2, u3, u4])
        forces = np.clip(self.B_inv @ u, self.p.f_min, self.p.f_max)

        self.log(t, state, ref, forces)
        return forces

    def get_name(self):
        return "PID"