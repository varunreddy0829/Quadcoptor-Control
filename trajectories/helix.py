"""
================================================================================
 Helix Trajectory Generator
================================================================================

Generates a smooth helix (corkscrew) reference trajectory for the quadrotor
to follow. This is used by all controllers as the target path.

HELIX GEOMETRY:
  - The drone starts at the origin (0, 0, z_start)
  - It spirals upward in a circle of fixed radius
  - While climbing at a constant vertical rate
  - Completing a set number of full loops before stopping

  At time t, the reference position is:
      x(t) = radius · cos(omega · t + phase)
      y(t) = radius · sin(omega · t + phase)
      z(t) = z_start + (climb_rate · t)

  Where omega = 2π / T_loop  (angular velocity of the spiral)

WHAT IS RETURNED PER TIMESTEP:
  Each call to get_reference(t) returns a 12D reference state vector in the
  same format as the quadrotor state — so controllers can directly compute
  the error as:  error = state - reference

  Velocities (vx, vy, vz) are the analytical derivatives of the position,
  so the reference is smooth and physically consistent (no jumps).

  Accelerations are also returned separately for feedforward control terms
  in LQR and MPC (they need to know the expected acceleration at each step).

================================================================================
"""

import numpy as np
from dataclasses import dataclass


@dataclass
class HelixParams:
    """
    Configuration for the helix trajectory.

    Tune these to make the trajectory harder or easier for the controllers.
    A larger radius or faster loop_time will stress the controllers more.
    """
    radius:     float = 1.0    # radius of the horizontal circle         [m]
    loop_time:  float = 10.0   # time to complete one full 360° loop     [s]
    climb_rate: float = 0.3    # vertical speed (upward)                 [m/s]
    n_loops:    float = 3.0    # total number of loops before trajectory ends
    z_start:    float = 0.5    # starting height above ground            [m]
    phase:      float = 0.0    # starting angle offset                   [rad]


class HelixTrajectory:
    """
    Generates position, velocity, and acceleration references for a helix path.

    Usage in a controller simulation loop:
        traj  = HelixTrajectory()
        t     = 0.0
        while t < traj.total_time:
            ref   = traj.get_reference(t)      # 12D reference state
            accel = traj.get_acceleration(t)   # 3D feedforward acceleration
            error = state - ref
            ...
            t += dt

    The trajectory is parameterised by time only — no need to track a waypoint
    index or interpolate between points.
    """

    def __init__(self, params: HelixParams = None):
        """
        Initialise the trajectory generator.

        Args:
            params : HelixParams instance. Uses default values if None.
                     Default gives a 1m radius helix, 3 loops over 30 seconds,
                     climbing at 0.3 m/s to a final height of ~9.5m.
        """
        self.p = params or HelixParams()

        # Angular velocity of the horizontal circle [rad/s]
        self.omega = 2.0 * np.pi / self.p.loop_time

        # Total duration of the trajectory [s]
        self.total_time = self.p.n_loops * self.p.loop_time

        # Final height reached at end of trajectory [m]
        self.z_final = self.p.z_start + self.p.climb_rate * self.total_time

    def get_position(self, t: float) -> np.ndarray:
        """
        Return the reference [x, y, z] position at time t.

        After the trajectory ends (t > total_time), the drone holds the
        final position — useful so controllers don't get a disappearing target.

        Args:
            t : current simulation time [s]

        Returns:
            pos : array (3,) — [x, y, z] in world frame [m]
        """
        t = np.clip(t, 0.0, self.total_time)
        x = self.p.radius * np.cos(self.omega * t + self.p.phase)
        y = self.p.radius * np.sin(self.omega * t + self.p.phase)
        z = self.p.z_start + self.p.climb_rate * t
        return np.array([x, y, z])

    def get_velocity(self, t: float) -> np.ndarray:
        """
        Return the reference [vx, vy, vz] velocity at time t.

        These are the analytical time derivatives of get_position(),
        so the reference is perfectly smooth with no finite-difference noise.

        After trajectory ends, velocity reference is zero (hold position).

        Args:
            t : current simulation time [s]

        Returns:
            vel : array (3,) — [vx, vy, vz] in world frame [m/s]
        """
        if t >= self.total_time:
            return np.zeros(3)

        vx = -self.p.radius * self.omega * np.sin(self.omega * t + self.p.phase)
        vy =  self.p.radius * self.omega * np.cos(self.omega * t + self.p.phase)
        vz =  self.p.climb_rate
        return np.array([vx, vy, vz])

    def get_acceleration(self, t: float) -> np.ndarray:
        """
        Return the reference [ax, ay, az] acceleration at time t.

        Used as a feedforward term in LQR and MPC controllers to improve
        tracking performance. Without feedforward, controllers must rely
        purely on error feedback which causes lag on curved paths.

        After trajectory ends, acceleration reference is zero.

        Args:
            t : current simulation time [s]

        Returns:
            acc : array (3,) — [ax, ay, az] in world frame [m/s²]
        """
        if t >= self.total_time:
            return np.zeros(3)

        ax = -self.p.radius * self.omega**2 * np.cos(self.omega * t + self.p.phase)
        ay = -self.p.radius * self.omega**2 * np.sin(self.omega * t + self.p.phase)
        az = 0.0  # constant climb rate → zero vertical acceleration
        return np.array([ax, ay, az])

    def get_reference(self, t: float) -> np.ndarray:
        """
        Return the full 12D reference state vector at time t.

        The reference state has the same layout as the quadrotor state:
            [x, y, z, vx, vy, vz, phi, theta, psi, p, q, r]

        Attitude references (phi, theta, psi, p, q, r) are all set to zero.
        This means the controllers will try to keep the drone level while
        following the helix — a reasonable assumption for smooth trajectories.

        Args:
            t : current simulation time [s]

        Returns:
            ref : array (12,) — full reference state vector
        """
        pos = self.get_position(t)
        vel = self.get_velocity(t)

        ref = np.zeros(12)
        ref[0:3] = pos   # x, y, z
        ref[3:6] = vel   # vx, vy, vz
        # ref[6:12] stays zero — level attitude, zero body rates
        return ref

    def get_full_trajectory(self, dt: float = 0.01):
        """
        Pre-compute the entire trajectory as arrays — useful for plotting
        the reference path before running any controller.

        Args:
            dt : time resolution for sampling the trajectory [s]

        Returns:
            times     : array (N,)    — time stamps
            positions : array (N, 3)  — [x, y, z] at each timestep
            velocities: array (N, 3)  — [vx, vy, vz] at each timestep
        """
        times      = np.arange(0, self.total_time + dt, dt)
        positions  = np.array([self.get_position(t)  for t in times])
        velocities = np.array([self.get_velocity(t)  for t in times])
        return times, positions, velocities

    def __repr__(self):
        return (f"HelixTrajectory(radius={self.p.radius}m, "
                f"loops={self.p.n_loops}, "
                f"duration={self.total_time:.1f}s, "
                f"z_final={self.z_final:.2f}m)")


if __name__ == "__main__":
    """
    Quick sanity check — run with:  python -m trajectories.helix
    Prints key values and saves a 3D plot of the helix to results/helix_preview.png
    """
    import os
    import matplotlib.pyplot as plt

    traj = HelixTrajectory()
    print(traj)
    print(f"  omega      : {traj.omega:.4f} rad/s")
    print(f"  total_time : {traj.total_time:.1f}s")
    print(f"  z_final    : {traj.z_final:.2f}m")

    # Check position at t=0 (should be on +x axis at z_start)
    p0 = traj.get_position(0)
    print(f"\n  pos at t=0 : {p0}  (expected [{traj.p.radius}, 0, {traj.p.z_start}])")

    # Check velocity at t=0 (should be purely in +y direction)
    v0 = traj.get_velocity(0)
    expected_vy = traj.p.radius * traj.omega
    print(f"  vel at t=0 : {v0}  (expected [0, {expected_vy:.4f}, {traj.p.climb_rate}])")

    # Plot 3D helix
    times, positions, _ = traj.get_full_trajectory(dt=0.05)
    os.makedirs("results", exist_ok=True)

    from mpl_toolkits.mplot3d import Axes3D  # explicit import forces registration
    fig = plt.figure(figsize=(7, 7))
    ax  = fig.add_subplot(111, projection='3d')
    ax.plot(positions[:, 0], positions[:, 1], positions[:, 2],
            color='royalblue', linewidth=1.5, label='helix path')
    ax.scatter(*positions[0],  color='green', s=60, zorder=5, label='start')
    ax.scatter(*positions[-1], color='red',   s=60, zorder=5, label='end')
    ax.set_xlabel('X [m]')
    ax.set_ylabel('Y [m]')
    ax.set_zlabel('Z [m]')
    ax.set_title('Helix reference trajectory')
    ax.legend()
    plt.tight_layout()
    plt.savefig("results/helix_preview.png", dpi=150)
    print("\nPlot saved to results/helix_preview.png")