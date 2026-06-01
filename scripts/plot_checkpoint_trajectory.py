#!/usr/bin/env python3
"""Plot v8 checkpoint trajectory: standard vs bypass vs ACIS-prune vs random-prune."""

import json
import glob
import os
from pathlib import Path

os.chdir(Path(__file__).resolve().parent.parent)

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

output_dir = Path("eval_results/trajectory")
artifact_dir = Path("artifacts/v8_checkpoint_trajectory")
artifact_dir.mkdir(parents=True, exist_ok=True)

# Collect results
standard_files = sorted(glob.glob(str(output_dir / "v8_step_*_standard.json")))

steps = []
standard_losses = []       # eps0.0_pruneFalse, ACIS active but no pruning
acis_prune_losses = []     # eps0.0_pruneTrue, ACIS importance-based pruning
bypass_losses = []         # eps0.0_pruneFalse, no ACIS
random_prune_losses = []   # eps0.0_pruneTrue from bypass, random pruning

for sf in standard_files:
    step_name = sf.split("v8_")[1].split("_standard")[0]
    step_num = int(step_name.split("_")[1])

    with open(sf) as f:
        sd = json.load(f)

    bf = sf.replace("_standard.json", "_bypass.json")
    if not os.path.exists(bf):
        print(f"  Bypass missing for {step_name}, skipping")
        continue

    with open(bf) as f:
        bd = json.load(f)

    steps.append(step_num)
    standard_losses.append(float(sd["eps0.0_pruneFalse"]["action_loss"]))
    acis_prune_losses.append(float(sd["eps0.0_pruneTrue"]["action_loss"]))
    bypass_losses.append(float(bd["eps0.0_pruneFalse"]["action_loss"]))
    random_prune_losses.append(float(bd["eps0.0_pruneTrue"]["action_loss"]))

if not steps:
    print("No complete results found!")
    exit(1)

print(f"Collected {len(steps)} checkpoints")

# Baseline reference
baseline_final_loss = 0.00408

# --- Plot 1: 4-curve trajectory ---
fig, ax = plt.subplots(1, 1, figsize=(10, 6))

ax.plot(steps, standard_losses, 'o-', color='#2196F3', linewidth=2, markersize=6,
        label='Standard (ACIS, no prune)')
ax.plot(steps, acis_prune_losses, 'D-', color='#4CAF50', linewidth=2, markersize=6,
        label='ACIS 50% prune')
ax.plot(steps, bypass_losses, 's--', color='#FF9800', linewidth=2, markersize=6,
        label='Bypass (no ACIS)')
ax.plot(steps, random_prune_losses, '^:', color='#F44336', linewidth=2, markersize=6,
        label='Random 50% prune')
ax.axhline(y=baseline_final_loss, color='gray', linestyle='-', alpha=0.5,
           label=f'Baseline final ({baseline_final_loss})')

ax.set_xlabel('Training Step', fontsize=14)
ax.set_ylabel('Action Loss', fontsize=14)
ax.set_title('v8 (Gradient Isolation) — Checkpoint Trajectory', fontsize=15)
ax.legend(fontsize=11)
ax.tick_params(labelsize=12)
ax.grid(True, alpha=0.3)
plt.tight_layout()

plot_path = artifact_dir / "checkpoint_trajectory.png"
plt.savefig(str(plot_path), dpi=300, bbox_inches='tight')
print(f"Plot saved: {plot_path}")
plt.close()

# --- Plot 2: ACIS advantage (gap between ACIS prune and random prune) ---
fig, ax = plt.subplots(1, 1, figsize=(10, 5))
acis_advantage = [r - a for r, a in zip(random_prune_losses, acis_prune_losses)]
colors = ['#4CAF50' if v > 0 else '#F44336' for v in acis_advantage]
ax.bar(steps, acis_advantage, width=1500, color=colors, alpha=0.8)
ax.axhline(y=0, color='black', linewidth=0.8)
ax.set_xlabel('Training Step', fontsize=14)
ax.set_ylabel('Random Prune Loss - ACIS Prune Loss', fontsize=14)
ax.set_title('ACIS Advantage over Random Pruning', fontsize=15)
ax.tick_params(labelsize=12)
ax.grid(True, alpha=0.3, axis='y')
plt.tight_layout()

adv_path = artifact_dir / "acis_advantage.png"
plt.savefig(str(adv_path), dpi=300, bbox_inches='tight')
print(f"ACIS advantage plot saved: {adv_path}")
plt.close()

# --- Save data JSON ---
data = {
    "steps": steps,
    "standard_loss": standard_losses,
    "acis_prune_loss": acis_prune_losses,
    "bypass_loss": bypass_losses,
    "random_prune_loss": random_prune_losses,
    "acis_advantage": acis_advantage,
    "baseline_final": baseline_final_loss,
}
data_path = artifact_dir / "trajectory_data.json"
with open(data_path, "w") as f:
    json.dump(data, f, indent=2)
print(f"Data saved: {data_path}")

# --- Summary table ---
print(f"\n{'Step':>8} | {'Standard':>10} | {'ACIS Prune':>11} | {'Bypass':>10} | {'Rand Prune':>11} | {'ACIS Adv':>9}")
print("-" * 75)
for i, s in enumerate(steps):
    print(f"{s:>8} | {standard_losses[i]:>10.5f} | {acis_prune_losses[i]:>11.5f} | {bypass_losses[i]:>10.5f} | {random_prune_losses[i]:>11.5f} | {acis_advantage[i]:>+9.5f}")

