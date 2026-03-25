"""
================================================================================
 Quadrotor Full 6-DOF Rigid Body Dynamics with Wind Disturbances
================================================================================

STATE VECTOR  (12 elements)
─────────────────────────────────────────────────────────────────────────────
  Index  Symbol   Description                    Frame   Units
  ─────  ──────   ───────────────────────────── ──────  ──────
  [0]    x        position along X               World   m
  [1]    y        position along Y               World   m
  [2]    z        position along Z (up)          World   m
  [3]    vx       velocity along X               World   m/s
  [4]    vy       velocity along Y               World   m/s
  [5]    vz       velocity along Z               World   m/s
  [6]    phi      roll  angle  (rotation about X) World   rad
  [7]    theta    pitch angle  (rotation about Y) World   rad
  [8]    psi      yaw   angle  (rotation about Z) World   rad
  [9]    p        roll  rate                      Body    rad/s
  [10]   q        pitch rate                      Body    rad/s
  [11]   r        yaw   rate                      Body    rad/s

  Angle convention: ZYX Euler (yaw → pitch → roll), aerospace standard.

================================================================================
"""

import numpy as np
from dataclasses import dataclass, field


from .wind.noise import WindModel


@dataclass
class QuadrotorParams:
    """Stores all physical and aerodynamic parameters of the quadrotor."""
    mass: float = 0.5          # total mass                          [kg]
    g: float = 9.81            # gravitational acceleration          [m/s²]
    arm_length: float = 0.17   # motor-to-center distance            [m]

    # Inertia
    Ixx: float = 4.856e-3      # moment of inertia about X (roll)   [kg·m²]
    Iyy: float = 4.856e-3      # moment of inertia about Y (pitch)  [kg·m²]
    Izz: float = 8.801e-3      # moment of inertia about Z (yaw)    [kg·m²]

    # Rotor coefficients
    k_thrust: float = 2.9265e-6   # thrust coefficient               [N·s²/rad²]
    k_drag:   float = 1.1691e-7   # drag   coefficient               [N·m·s²/rad²]

    # Motor limits
    f_min: float = 0.0            # minimum rotor thrust              [N]
    f_max: float = 3.0            # maximum rotor thrust              [N]

    # Derived
    I: np.ndarray = field(init=False)
    I_inv: np.ndarray = field(init=False)
    hover_thrust: float = field(init=False)

    def __post_init__(self):
        """Precompute derived quantities."""
        self.I     = np.diag([self.Ixx, self.Iyy, self.Izz])
        self.I_inv = np.diag([1/self.Ixx, 1/self.Iyy, 1/self.Izz])
        self.hover_thrust = self.mass * self.g / 4.0


class QuadrotorDynamics:
    """Simulates 6-DOF rigid body dynamics with stochastic wind."""

    STATE_DIM  = 12
    ACTION_DIM = 4

    X, Y, Z         = 0, 1, 2
    VX, VY, VZ      = 3, 4, 5
    PHI, THETA, PSI = 6, 7, 8
    P, Q, R         = 9, 10, 11

    def __init__(self, params: QuadrotorParams = None, dt: float = 0.01):
        self.p  = params or QuadrotorParams()
        self.dt = dt
        self._build_allocation_matrix()
        
        # Modular stochastic wind model
        self.wind = WindModel(dt=dt)

    def _build_allocation_matrix(self):
        L     = self.p.arm_length
        kd_kt = self.p.k_drag / self.p.k_thrust
        self.B = np.array([
            [ 1,       1,       1,       1      ],
            [ L,       L,      -L,      -L      ],
            [-L,       L,       L,      -L      ],
            [-kd_kt,   kd_kt,  -kd_kt,   kd_kt  ],
        ])

    def mix(self, forces: np.ndarray) -> np.ndarray:
        f = np.clip(forces, self.p.f_min, self.p.f_max)
        return self.B @ f

    @staticmethod
    def rotation_matrix(phi, theta, psi):
        cp, sp = np.cos(phi),   np.sin(phi)
        ct, st = np.cos(theta), np.sin(theta)
        cy, sy = np.cos(psi),   np.sin(psi)
        return np.array([
            [cy*ct,  cy*st*sp - sy*cp,  cy*st*cp + sy*sp],
            [sy*ct,  sy*st*sp + cy*cp,  sy*st*cp - cy*sp],
            [-st,    ct*sp,             ct*cp            ],
        ])

    @staticmethod
    def euler_rate_matrix(phi, theta):
        cp, sp = np.cos(phi), np.sin(phi)
        ct, tt = np.cos(theta), np.tan(theta)
        return np.array([
            [1,  sp*tt,  cp*tt],
            [0,  cp,    -sp   ],
            [0,  sp/ct,  cp/ct]
        ])

    def derivatives(self, state: np.ndarray, forces: np.ndarray) -> np.ndarray:
        phi, theta, psi = state[self.PHI], state[self.THETA], state[self.PSI]
        vx, vy, vz      = state[self.VX],  state[self.VY],   state[self.VZ]
        p,  q,  r       = state[self.P],   state[self.Q],    state[self.R]

        T, tau_x, tau_y, tau_z = self.mix(forces)
        R_mat = self.rotation_matrix(phi, theta, psi)
        W_mat = self.euler_rate_matrix(phi, theta)

        # Translational acceleration
        thrust_world = R_mat @ np.array([0, 0, T])
        # Include current wind force from the model in the dynamics
        acc = np.array([0, 0, -self.p.g]) + (thrust_world + self.wind.force) / self.p.mass

        # Euler rates
        omega_body  = np.array([p, q, r])
        euler_rates = W_mat @ omega_body

        # Angular acceleration
        tau     = np.array([tau_x, tau_y, tau_z])
        ang_acc = self.p.I_inv @ (tau - np.cross(omega_body, self.p.I @ omega_body))

        dstate = np.zeros(self.STATE_DIM)
        dstate[self.X:self.Z+1]     = [vx, vy, vz]
        dstate[self.VX:self.VZ+1]   = acc
        dstate[self.PHI:self.PSI+1] = euler_rates
        dstate[self.P:self.R+1]     = ang_acc
        return dstate

    def step(self, state: np.ndarray, forces: np.ndarray) -> np.ndarray:
        """Advance simulation and Evolve the wind state."""
        self.wind.step()
        
        dt = self.dt
        k1 = self.derivatives(state,             forces)
        k2 = self.derivatives(state + 0.5*dt*k1, forces)
        k3 = self.derivatives(state + 0.5*dt*k2, forces)
        k4 = self.derivatives(state +     dt*k3, forces)
        new_state = state + (dt / 6.0) * (k1 + 2*k2 + 2*k3 + k4)
        new_state[self.PSI] = (new_state[self.PSI] + np.pi) % (2 * np.pi) - np.pi
        return new_state

    def hover_forces(self) -> np.ndarray:
        return np.full(4, self.p.hover_thrust)

    @staticmethod
    def state_from_pos(x=0., y=0., z=0.) -> np.ndarray:
        s = np.zeros(12)
        s[0], s[1], s[2] = x, y, z
        return s

    def hover_forces(self) -> np.ndarray:
        return np.full(4, self.p.hover_thrust)

    @staticmethod
    def state_from_pos(x=0., y=0., z=0.) -> np.ndarray:
        s = np.zeros(12)
        s[0], s[1], s[2] = x, y, z
        return s