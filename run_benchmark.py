import numpy as np
import matplotlib.pyplot as plt
import os
import yaml
from datetime import datetime

from sim.quadrotor import QuadrotorDynamics, QuadrotorParams
from trajectories.helix import HelixTrajectory
from trajectories.lemniscate import LemniscateTrajectory
from controllers.pid import PIDController
from controllers.lqr import LQRController
from controllers.mpc import MPCController
from config import get_pid_config, get_lqr_config, get_mpc_config

# ── Configuration ───────────────────────────────────────────────────────────
RANDOM_SEED = 42
DT = 0.01

TEST_CASES = [
    {"name": "Helix_NoWind",   "traj": "helix",      "wind": False},
    {"name": "Helix_Windy",    "traj": "helix",      "wind": True},
    {"name": "Lemn_NoWind",    "traj": "lemniscate", "wind": False},
    {"name": "Lemn_Windy",     "traj": "lemniscate", "wind": True},
]

WIND_PARAMS = {"mean": [0.25, 0.25, 0.0], "sigma": 0.5}

os.makedirs("results/images", exist_ok=True)
os.makedirs("results/data", exist_ok=True)

def run_simulation(controller, traj, params, dt, wind_enabled):
    np.random.seed(RANDOM_SEED)
    quad = QuadrotorDynamics(params=params, dt=dt)
    quad.wind.enabled = wind_enabled
    quad.wind.set_params(mean=WIND_PARAMS["mean"], sigma=WIND_PARAMS["sigma"])

    state = quad.state_from_pos(*traj.get_position(0))
    controller.reset()

    t = 0.0
    while t <= traj.total_time:
        ref = traj.get_reference(t)
        accel = traj.get_acceleration(t)

        if isinstance(controller, LQRController):
            forces = controller.compute_forces(state, ref, t, accel_ref=accel)
        else:
            forces = controller.compute_forces(state, ref, t)

        state = quad.step(state, forces)
        t += dt
    return controller

def save_visuals(controllers, test_name, params):
    """Generates the 4-figure suite for a test case."""
    num_ctrls = len(controllers)
    
    # 1. 3D Path
    fig1 = plt.figure(figsize=(18, 5))
    fig1.suptitle(f"3D Trajectory: {test_name}", fontsize=14)
    for idx, (ctrl, label, color) in enumerate(controllers):
        states, refs = np.array(ctrl.state_history), np.array(ctrl.ref_history)
        ax = fig1.add_subplot(1, num_ctrls, idx + 1, projection='3d')
        ax.plot(refs[:,0], refs[:,1], refs[:,2], 'g--', alpha=0.7, label='Reference')
        ax.plot(states[:,0], states[:,1], states[:,2], color=color, label=label)
        ax.set_title(f"{label} (RMSE: {ctrl.get_rmse():.3f}m)")
        ax.legend(fontsize=8)
    plt.savefig(f"results/images/{test_name}_3d.png", dpi=150)
    plt.close()

    # 2. Error Plot
    fig2, ax = plt.subplots(figsize=(12, 5))
    for ctrl, label, color in controllers:
        states, refs = np.array(ctrl.state_history), np.array(ctrl.ref_history)
        err = np.linalg.norm(states[:,0:3] - refs[:,0:3], axis=1)
        ax.plot(np.array(ctrl.time_history), err, color=color, label=f"{label} RMSE:{ctrl.get_rmse():.3f}")
    ax.set_title(f"Position Error: {test_name}")
    ax.set_ylabel("Error [m]"); ax.set_xlabel("Time [s]")
    ax.legend(); ax.grid(True, alpha=0.3)
    plt.savefig(f"results/images/{test_name}_error.png", dpi=150)
    plt.close()

    # 3. Rotor Forces
    fig3, axs = plt.subplots(num_ctrls, 4, figsize=(18, 3.5 * num_ctrls))
    fig3.suptitle(f"Rotor Forces: {test_name}", fontsize=14)
    motor_labels = ['FL (f1)', 'BL (f2)', 'BR (f3)', 'FR (f4)']
    for row, (ctrl, label, color) in enumerate(controllers):
        forces = np.array(ctrl.force_history)
        times = np.array(ctrl.time_history)
        for col in range(4):
            axs[row, col].plot(times, forces[:, col], color=color, linewidth=0.8)
            axs[row, col].set_ylim([params.f_min - 0.1, params.f_max + 0.1])
            if row == 0: axs[row, col].set_title(motor_labels[col])
            if col == 0: axs[row, col].set_ylabel(f"{label}\nForce [N]")
            if row == num_ctrls - 1: axs[row, col].set_xlabel("Time [s]")
    plt.tight_layout()
    plt.savefig(f"results/images/{test_name}_forces.png", dpi=150)
    plt.close()

    # 4. Summary Benchmark
    fig4, axes = plt.subplots(1, 2, figsize=(12, 5))
    labels = [l for _, l, _ in controllers]
    colors = [c for _, _, c in controllers]
    rmses = [c.get_rmse() for c, _, _ in controllers]
    efforts = [c.get_control_effort() for c, _, _ in controllers]

    axes[0].bar(labels, rmses, color=colors, alpha=0.8, edgecolor='black')
    axes[0].set_title("RMSE (Lower is Better)")
    for i, v in enumerate(rmses): axes[0].text(i, v, f"{v:.3f}", ha='center')

    axes[1].bar(labels, efforts, color=colors, alpha=0.8, edgecolor='black')
    axes[1].set_title("Control Effort")
    for i, v in enumerate(efforts): axes[1].text(i, v, f"{v:.0f}", ha='center')
    
    plt.savefig(f"results/images/{test_name}_summary.png", dpi=150)
    plt.close()

# ── Main Runner ─────────────────────────────────────────────────────────────
params = QuadrotorParams()
all_results_data = {}

for case in TEST_CASES:
    print(f"Running Test Case: {case['name']}")
    
    traj = HelixTrajectory() if case["traj"] == "helix" else LemniscateTrajectory()

    controllers = [
        (PIDController(get_pid_config(params), params, DT), "PID", "tab:blue"),
        (LQRController(get_lqr_config(params, DT), params, DT), "LQR", "tab:orange"),
        (MPCController(get_mpc_config(params, DT), params, DT, traj), "MPC", "tab:green")
    ]

    case_metrics = {}
    for ctrl, label, color in controllers:
        run_simulation(ctrl, traj, params, DT, case["wind"])
        case_metrics[label] = {
            "rmse": float(ctrl.get_rmse()),
            "effort": float(ctrl.get_control_effort())
        }
    
    save_visuals(controllers, case["name"], params)
    all_results_data[case["name"]] = case_metrics

with open("results/data/benchmark_results.yaml", "w") as f:
    yaml.dump(all_results_data, f, default_flow_style=False)

print("\nBenchmark complete. Check results/images and results/data/benchmark_results.yaml")