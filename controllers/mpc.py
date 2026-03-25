"""
================================================================================
 Model Predictive Control (MPC) Controller
================================================================================

Optimal receding-horizon controller for quadrotor trajectory tracking.
Uses cvxpy and OSQP to solve a constrained quadratic program at each timestep.

FEATURES:
  - Uses discrete-time linearised dynamics (ZOH).
  - Enforces strict minimum/maximum bounds on virtual actuators.
  - Implements Differential Flatness to map trajectory accelerations 
    into physically accurate pitch and roll references.
  - Pre-compiles the optimization graph using cvxpy Parameters for speed.
================================================================================
"""

import numpy as np
import cvxpy as cp
from .base import BaseController
from sim.quadrotor import QuadrotorParams

class MPCController(BaseController):
    """Receding horizon MPC controller. Parameters from get_mpc_config()."""

    def __init__(self, cfg: dict, params: QuadrotorParams, dt: float, traj=None):
        """
        Args:
            cfg    : dict from get_mpc_config()
            params : QuadrotorParams
            dt     : control timestep [s]
            traj   : Trajectory object (e.g., HelixTrajectory) for lookahead
        """
        super().__init__(dt=dt)
        self.p = params
        self.traj = traj
        
        # ── Setup parameters ──
        self.Ad = cfg['Ad']
        self.Bd = cfg['Bd']
        self.Q  = cfg['Q']
        self.R  = cfg['R']
        self.N  = cfg['N']
        
        self.nx = self.Ad.shape[0] # 12 states
        self.nu = self.Bd.shape[1] # 4 virtual inputs (delta_T, tau_x, tau_y, tau_z)
        
        # ── Mixer Inverse Setup ──
        L = self.p.arm_length
        kd_kt = self.p.k_drag / self.p.k_thrust
        mixer_B = np.array([
            [ 1,       1,       1,       1      ],
            [ L,       L,      -L,      -L      ],
            [-L,       L,       L,      -L      ],
            [-kd_kt,   kd_kt,  -kd_kt,   kd_kt  ],
        ])
        self.mixer_B_inv = np.linalg.inv(mixer_B)
        
        # ── CVXPY Problem Setup (Compiled once for execution speed) ──
        self.x_init = cp.Parameter(self.nx)
        self.x_ref  = cp.Parameter((self.nx, self.N))
        
        self.x = cp.Variable((self.nx, self.N + 1))
        self.u = cp.Variable((self.nu, self.N))
        
        cost = 0
        constraints = [self.x[:, 0] == self.x_init]
        
        for k in range(self.N):
            # Tracking cost
            cost += cp.quad_form(self.x[:, k] - self.x_ref[:, k], self.Q)
            # Control effort cost
            cost += cp.quad_form(self.u[:, k], self.R)
            
            # System Dynamics Constraint (Discrete)
            constraints += [self.x[:, k+1] == self.Ad @ self.x[:, k] + self.Bd @ self.u[:, k]]
            
            # Virtual Input Constraints (Deviation limits)
            constraints += [self.u[:, k] >= cfg['u_min'], 
                            self.u[:, k] <= cfg['u_max']]
                            
        # Terminal cost (Using Q directly, multiplied for heavier terminal weight)
        cost += cp.quad_form(self.x[:, self.N] - self.x_ref[:, self.N-1], self.Q * 10)
        
        self.prob = cp.Problem(cp.Minimize(cost), constraints)
        
        # Internal buffer to hold reference trajectories
        self._ref_buffer = np.zeros((self.nx, self.N))
        
        self.reset()

    def reset(self):
        """Clear history buffers and initialize state."""
        super().reset()
        
    def compute_forces(self, 
                       state: np.ndarray, 
                       reference: np.ndarray, 
                       t: float) -> np.ndarray:
        """
        Solve the MPC optimization and return the immediate rotor forces.
        """
        # 1. Standardise the current state (Wrap yaw to [-pi, pi] to avoid solver discontinuities)
        current_state = state.copy()
        current_state[8] = (current_state[8] + np.pi) % (2 * np.pi) - np.pi
        self.x_init.value = current_state
        
        # 2. Look ahead to populate the reference horizon
        for k in range(self.N):
            if self.traj is not None:
                t_future = t + k * self.dt
                ref_state = self.traj.get_reference(t_future).copy()
                ref_accel = self.traj.get_acceleration(t_future)
                
                # DIFFERENTIAL FLATNESS: Map acceleration to required roll/pitch
                # x_ddot = g * theta  --> theta_ref = x_ddot / g
                # y_ddot = -g * phi   --> phi_ref   = -y_ddot / g
                ref_state[6] = -ref_accel[1] / self.p.g  # phi (roll)
                ref_state[7] =  ref_accel[0] / self.p.g  # theta (pitch)
                
                self._ref_buffer[:, k] = ref_state
            else:
                self._ref_buffer[:, k] = reference
                
        self.x_ref.value = self._ref_buffer
        
        # 3. Solve the optimization problem
        try:
            # OSQP is fast and warm_start=True reuses the previous solution 
            self.prob.solve(solver=cp.OSQP, warm_start=True)
            if self.prob.status not in ["optimal", "optimal_inaccurate"]:
                raise cp.error.SolverError(f"Solver status: {self.prob.status}")
            u_opt = self.u[:, 0].value
        except cp.error.SolverError as e:
            # Fallback to hover inputs if the solver fails or becomes infeasible
            print(f"Warning: MPC failed at t={t:.2f} ({e}). Falling back to hover.")
            u_opt = np.zeros(self.nu)
            
        # 4. Map virtual inputs back to actual rotor forces
        # u_opt[0] is the change in thrust from hover (delta_T)
        virtual_forces = u_opt.copy()
        virtual_forces[0] += self.p.mass * self.p.g 
        
        rotor_forces = self.mixer_B_inv @ virtual_forces
        rotor_forces = np.clip(rotor_forces, self.p.f_min, self.p.f_max)
        
        # 5. Log internal data for the benchmark plots
        self.log(t, state, self._ref_buffer[:, 0], rotor_forces)
        
        return rotor_forces

    def get_name(self) -> str:
        return "MPC"