"""
================================================================================
 Base Controller — Abstract Interface
================================================================================

Defines the contract that every controller in this project must follow.
PID, LQR, and MPC all inherit from this class and implement its methods.

WHY THIS EXISTS:
  The benchmark script (run_all.py) needs to run all controllers in the same
  loop without knowing or caring about their internal logic. By enforcing a
  common interface here, the benchmark can treat every controller identically:

      for controller in [pid, lqr, mpc]:
          forces = controller.compute_forces(state, reference, t)
          state  = quad.step(state, forces)

  If a controller does NOT implement all abstract methods, Python will raise
  a TypeError at instantiation time — catching mistakes early.

USAGE (for each new controller you write):
  1. Import and inherit from BaseController
  2. Implement compute_forces() with your control logic
  3. Implement reset() to clear any internal state
  4. Optionally override get_name() for clean plot labels

================================================================================
"""

from abc import ABC, abstractmethod
import numpy as np


class BaseController(ABC):
    """
    Abstract base class for all quadrotor controllers.

    Every controller (PID, LQR, MPC) must inherit from this class and
    implement the two abstract methods: compute_forces() and reset().

    Provides:
      - A common interface so the benchmark loop works for all controllers
      - Shared logging of state/force history for plotting
      - A get_name() method for plot labels (override in subclass)
    """

    def __init__(self, dt: float = 0.01):
        """
        Initialise the base controller.

        Args:
            dt : control loop timestep [s]. Must match the simulation dt
                 in QuadrotorDynamics to keep everything in sync.
        """
        self.dt = dt

        # ── History buffers (filled during simulation, used for plotting) ──
        # Each call to compute_forces() should append to these via log()
        self.state_history  = []   # list of 12D state vectors
        self.force_history  = []   # list of 4D force vectors
        self.ref_history    = []   # list of 12D reference state vectors
        self.time_history   = []   # list of time stamps [s]

    # ── Abstract methods — MUST be implemented by every subclass ───────────

    @abstractmethod
    def compute_forces(self,
                       state:     np.ndarray,
                       reference: np.ndarray,
                       t:         float) -> np.ndarray:
        """
        Compute the rotor thrust forces to apply at this timestep.

        This is the main control law. Takes the current state and the
        reference state at time t, returns 4 rotor forces.

        Args:
            state     : current quadrotor state (12,)
                        [x, y, z, vx, vy, vz, phi, theta, psi, p, q, r]
            reference : reference state at time t (12,)
                        same layout as state — from HelixTrajectory.get_reference(t)
            t         : current simulation time [s]

        Returns:
            forces : rotor thrust forces (4,) [N]
                     ordered [f1, f2, f3, f4] matching QuadrotorDynamics
        """
        pass

    @abstractmethod
    def reset(self):
        """
        Reset all internal controller state to initial conditions.

        Called before each new simulation run so controllers start fresh.
        Each subclass must clear its own internal variables here
        (e.g. PID integral terms, MPC warm-start solutions).

        The base class history buffers are also cleared here automatically
        via super().reset() — subclasses should call this.
        """
        pass

    # ── Concrete methods — shared by all controllers, no need to override ──

    def log(self,
            t:         float,
            state:     np.ndarray,
            reference: np.ndarray,
            forces:    np.ndarray):
        """
        Record one timestep of data into the history buffers.

        Call this inside compute_forces() of every subclass to automatically
        build up the data needed for benchmark plots and metrics.

        Args:
            t         : current time [s]
            state     : current state (12,)
            reference : reference state (12,)
            forces    : rotor forces applied (4,)
        """
        self.time_history.append(t)
        self.state_history.append(state.copy())
        self.ref_history.append(reference.copy())
        self.force_history.append(forces.copy())

    def clear_history(self):
        """
        Clear all history buffers.

        Called automatically by reset() so each simulation run starts
        with empty buffers. Also useful if you want to re-use a controller
        object across multiple trajectories.
        """
        self.state_history  = []
        self.force_history  = []
        self.ref_history    = []
        self.time_history   = []

    def get_name(self) -> str:
        """
        Return a human-readable name for this controller.

        Used as plot titles and legend labels in the benchmark.
        Override this in each subclass to return e.g. 'PID', 'LQR', 'MPC'.

        Default fallback returns the class name if not overridden.
        """
        return self.__class__.__name__

    def get_position_error(self) -> np.ndarray:
        """
        Compute position tracking error at each logged timestep.

        Returns the Euclidean distance between actual and reference position
        at every timestep — the primary metric for comparing controllers.

        Returns:
            errors : array (N,) — position error [m] at each timestep.
                     Lower is better. Used for RMSE computation.
        """
        if not self.state_history:
            return np.array([])

        states = np.array(self.state_history)   # (N, 12)
        refs   = np.array(self.ref_history)     # (N, 12)
        pos_error = states[:, 0:3] - refs[:, 0:3]          # (N, 3)
        return np.linalg.norm(pos_error, axis=1)            # (N,)

    def get_rmse(self) -> float:
        """
        Compute Root Mean Square Error (RMSE) of position tracking.

        RMSE is the primary benchmark metric for comparing controllers.
        A lower RMSE means the controller tracked the helix more accurately.

        Returns:
            rmse : scalar [m] — lower is better.
        """
        errors = self.get_position_error()
        if len(errors) == 0:
            return float('inf')
        return float(np.sqrt(np.mean(errors**2)))

    def get_control_effort(self) -> float:
        """
        Compute total control effort over the simulation.

        Measured as the sum of squared rotor forces across all timesteps.
        A lower value means the controller achieves tracking more efficiently
        (uses less energy). Used as secondary benchmark metric.

        Returns:
            effort : scalar — lower means more energy efficient.
        """
        if not self.force_history:
            return float('inf')
        forces = np.array(self.force_history)   # (N, 4)
        return float(np.sum(forces**2))

    def __repr__(self):
        return (f"{self.get_name()}(dt={self.dt}, "
                f"steps_logged={len(self.time_history)})")