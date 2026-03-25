"""
================================================================================
 Controller Benchmark — PID vs LQR
================================================================================
Runs both controllers on the same helix trajectory under identical
wind conditions (fixed random seed). All parameters from config.py.

Usage: python run_benchmark.py
================================================================================
"""

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import os

from sim.quadrotor import QuadrotorDynamics, QuadrotorParams
from trajectories.helix import HelixTrajectory
from controllers.pid import PIDController
from controllers.lqr import LQRController
from config import get_pid_config, get_lqr_config

# ── Global settings ───────────────────────────────────────────────────────────
RANDOM_SEED  = 42
DT           = 0.01
WIND_ENABLED = True
WIND_MEAN    = [0.25, 0.25, 0.0]
WIND_SIGMA   = 0.5

os.makedirs("results", exist_ok=True)


def run_simulation(controller, traj, params, dt, label):
    """
    Run one controller simulation with fixed wind seed.

    Returns controller with populated history buffers.
    """
    np.random.seed(RANDOM_SEED)

    quad = QuadrotorDynamics(params=params, dt=dt)
    quad.wind.enabled = WIND_ENABLED
    quad.wind.set_params(mean=WIND_MEAN, sigma=WIND_SIGMA)

    state = quad.state_from_pos(*traj.get_position(0))
    controller.reset()

    print(f"  Running {label}...")
    t = 0.0
    while t <= traj.total_time:
        ref   = traj.get_reference(t)
        accel = traj.get_acceleration(t)

        if isinstance(controller, LQRController):
            forces = controller.compute_forces(state, ref, t, accel_ref=accel)
        else:
            forces = controller.compute_forces(state, ref, t)

        state = quad.step(state, forces)
        t    += dt

    print(f"    RMSE           : {controller.get_rmse():.4f} m")
    print(f"    Control effort : {controller.get_control_effort():.2f}")
    return controller


# ── Build controllers from config ─────────────────────────────────────────────
params  = QuadrotorParams()
traj    = HelixTrajectory()

pid_cfg = get_pid_config(params=params)
lqr_cfg = get_lqr_config(params=params)

pid = PIDController(cfg=pid_cfg, params=params, dt=DT)
lqr = LQRController(cfg=lqr_cfg, params=params, dt=DT)

controllers = [
    (pid, "PID", "tab:blue"),
    (lqr, "LQR", "tab:orange"),
]

# ── Run all ───────────────────────────────────────────────────────────────────
print(f"\nBenchmark settings:")
print(f"  Trajectory : helix  ({traj.total_time:.0f}s, {traj.p.n_loops:.0f} loops)")
print(f"  Wind       : {'ON' if WIND_ENABLED else 'OFF'}"
      f"  (seed={RANDOM_SEED}, mean={WIND_MEAN}, sigma={WIND_SIGMA})")
print(f"  Timestep   : {DT}s\n")

for ctrl, label, color in controllers:
    run_simulation(ctrl, traj, params, DT, label)

# ── Summary table ─────────────────────────────────────────────────────────────
print(f"\n{'─'*52}")
print(f"{'Controller':<12} {'RMSE (m)':<14} {'Control Effort':<16}")
print(f"{'─'*52}")
for ctrl, label, _ in controllers:
    print(f"{label:<12} {ctrl.get_rmse():<14.4f} {ctrl.get_control_effort():<16.2f}")
print(f"{'─'*52}\n")

# ── Figure 1: 3D trajectories ─────────────────────────────────────────────────
fig1 = plt.figure(figsize=(12, 5))
fig1.suptitle(
    f"3D Trajectory Comparison  |  Wind={'ON' if WIND_ENABLED else 'OFF'}"
    f"  |  Seed={RANDOM_SEED}", fontsize=13)

for idx, (ctrl, label, color) in enumerate(controllers):
    states = np.array(ctrl.state_history)
    refs   = np.array(ctrl.ref_history)
    ax = fig1.add_subplot(1, 2, idx + 1, projection='3d')
    ax.plot(refs[:,0], refs[:,1], refs[:,2], 'g--', linewidth=1,
            label='reference', alpha=0.7)
    ax.plot(states[:,0], states[:,1], states[:,2], color=color,
            linewidth=1, label=label)
    ax.set_xlabel('X [m]'); ax.set_ylabel('Y [m]'); ax.set_zlabel('Z [m]')
    ax.set_title(f'{label}  (RMSE={ctrl.get_rmse():.3f}m)')
    ax.legend(fontsize=8)

plt.tight_layout()
plt.savefig('results/benchmark_3d.png', dpi=150)

# ── Figure 2: Position error comparison ──────────────────────────────────────
fig2, ax = plt.subplots(figsize=(12, 5))
for ctrl, label, color in controllers:
    states  = np.array(ctrl.state_history)
    refs    = np.array(ctrl.ref_history)
    times   = np.array(ctrl.time_history)
    pos_err = np.linalg.norm(states[:,0:3] - refs[:,0:3], axis=1)
    ax.plot(times, pos_err, color=color, linewidth=1.2,
            label=f"{label}  (RMSE={ctrl.get_rmse():.3f}m)")

ax.set_xlabel('Time [s]')
ax.set_ylabel('Position error [m]')
ax.set_title('Position tracking error comparison')
ax.legend(); ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('results/benchmark_error.png', dpi=150)

# ── Figure 3: Rotor forces grid ───────────────────────────────────────────────
fig3, axs = plt.subplots(2, 4, figsize=(18, 7))
fig3.suptitle('Rotor Forces Comparison', fontsize=13)
motor_labels = ['Front-Left (f1)', 'Back-Left (f2)',
                'Back-Right (f3)', 'Front-Right (f4)']

for row, (ctrl, label, color) in enumerate(controllers):
    forces_hist = np.array(ctrl.force_history)
    times       = np.array(ctrl.time_history)
    for col in range(4):
        axs[row, col].plot(times, forces_hist[:, col],
                           color=color, linewidth=0.8)
        axs[row, col].set_ylim([params.f_min - 0.1, params.f_max + 0.1])
        axs[row, col].grid(True, alpha=0.3)
        if row == 0: axs[row, col].set_title(motor_labels[col])
        if col == 0: axs[row, col].set_ylabel(f'{label}\nForce [N]')
        if row == 1: axs[row, col].set_xlabel('Time [s]')

plt.tight_layout()
plt.savefig('results/benchmark_forces.png', dpi=150)

# ── Figure 4: Summary bar charts ──────────────────────────────────────────────
fig4, axes = plt.subplots(1, 2, figsize=(10, 4))
labels  = [l for _, l, _ in controllers]
colors  = [c for _, _, c in controllers]
rmses   = [ctrl.get_rmse()           for ctrl, _, _ in controllers]
efforts = [ctrl.get_control_effort() for ctrl, _, _ in controllers]

axes[0].bar(labels, rmses, color=colors, alpha=0.8,
            edgecolor='black', linewidth=0.5)
axes[0].set_ylabel('RMSE [m]')
axes[0].set_title('Position tracking error')
axes[0].grid(True, alpha=0.3, axis='y')
for i, v in enumerate(rmses):
    axes[0].text(i, v + 0.005, f'{v:.3f}', ha='center', fontsize=10)

axes[1].bar(labels, efforts, color=colors, alpha=0.8,
            edgecolor='black', linewidth=0.5)
axes[1].set_ylabel('Control effort')
axes[1].set_title('Total control effort')
axes[1].grid(True, alpha=0.3, axis='y')
for i, v in enumerate(efforts):
    axes[1].text(i, v + 100, f'{v:.0f}', ha='center', fontsize=10)

fig4.suptitle(
    f"Controller benchmark  |  Wind={'ON' if WIND_ENABLED else 'OFF'}"
    f"  |  Seed={RANDOM_SEED}", fontsize=12)
plt.tight_layout()
plt.savefig('results/benchmark_summary.png', dpi=150)

plt.show()
print("Benchmark complete.")
print("Saved: benchmark_3d.png, benchmark_error.png, "
      "benchmark_forces.png, benchmark_summary.png")