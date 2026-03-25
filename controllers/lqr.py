"""
================================================================================
 Linear Quadratic Regulator (LQR) Controller
================================================================================

Optimal LQR controller for quadrotor trajectory tracking.
All tunable parameters loaded from config.py via get_lqr_config().

CONTROL LAW:
  u = u_hover + u_ff - K @ (state - reference)

  u_hover = [m*g, 0, 0, 0]       — gravity feedforward (always active)
  u_ff    = [m*az_ref, 0, 0, 0]  — trajectory acceleration feedforward
  K       — optimal gain matrix computed offline via CARE in config.py
================================================================================
"""

import numpy as np
from .base import BaseController
from sim.quadrotor import QuadrotorParams


class LQRController(BaseController):
    """Optimal LQR controller. Parameters from get_lqr_config()."""

    def __init__(self, cfg: dict, params: QuadrotorParams = None, dt: float = 0.01):
        """
        Args:
            cfg    : dict from get_lqr_config()
            params : QuadrotorParams
            dt     : control timestep [s]
        """
        super().__init__(dt=dt)
        self.p = params or QuadrotorParams()

        # Optimal gain matrix (4, 12) — computed in config.py via CARE
        self.K = cfg['K']

        # Shared actuator limits [delta_T, tau_x, tau_y, tau_z]
        self.u_min = cfg['u_min']
        self.u_max = cfg['u_max']

        # Mixer inverse
        L     = self.p.arm_length
        kd_kt = self.p.k_drag / self.p.k_thrust
        mixer_B = np.array([
            [ 1,       1,       1,       1      ],
            [ L,       L,      -L,      -L      ],
            [-L,       L,       L,      -L      ],
            [-kd_kt,   kd_kt,  -kd_kt,   kd_kt  ],
        ])
        self.mixer_B_inv = np.linalg.inv(mixer_B)

        self.reset()

    def reset(self):
        """Clear history buffers."""
        super().clear_history()

    def compute_forces(self,
                       state:     np.ndarray,
                       reference: np.ndarray,
                       t:         float,
                       accel_ref: np.ndarray = None) -> np.ndarray:
        """
        LQR control law with hover + trajectory feedforward.

        Args:
            state     : current state (12,)
            reference : reference state (12,)
            t         : current time [s]
            accel_ref : reference acceleration (3,) [m/s²] for feedforward

        Returns:
            forces : rotor thrust forces (4,) [N]
        """
        m, g = self.p.mass, self.p.g

        # Hover feedforward — always cancels gravity
        u_hover = np.array([m * g, 0.0, 0.0, 0.0])

        # Trajectory acceleration feedforward — anticipates reference motion
        u_ff = np.zeros(4)
        if accel_ref is not None:
            u_ff[0] = m * accel_ref[2]

        # State error with yaw wrapping to [-pi, pi]
        error    = state - reference
        error[8] = (error[8] + np.pi) % (2 * np.pi) - np.pi

        # LQR feedback correction
        u_fb = -self.K @ error

        # Total virtual input [T, tau_x, tau_y, tau_z]
        u = u_hover + u_ff + u_fb

        # ── Clip virtual inputs to shared actuator limits ──────────────────
        # Clip delta_T (deviation from hover), not total thrust directly
        hover = m * g
        u[0]  = hover + np.clip(u[0] - hover, self.u_min[0], self.u_max[0])
        u[1]  = np.clip(u[1], self.u_min[1], self.u_max[1])
        u[2]  = np.clip(u[2], self.u_min[2], self.u_max[2])
        u[3]  = np.clip(u[3], self.u_min[3], self.u_max[3])

        # ── Mixer: virtual inputs → rotor forces ───────────────────────────
        forces = np.clip(self.mixer_B_inv @ u, self.p.f_min, self.p.f_max)

        self.log(t, state, reference, forces)
        return forces

    def get_name(self):
        return "LQR"