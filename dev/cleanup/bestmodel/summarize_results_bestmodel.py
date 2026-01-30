"""Summarize best model results across all seeds and create robustness visualization.

Looks at results/best_checkpoints/ for the best validation Q-tables.
"""
import json
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt

# Look at best_checkpoints (from run_best_checkpoints.sh)
checkpoints_dir = Path(__file__).parent.parent.parent / "results" / "best_checkpoints"
output_file = Path(__file__).parent / "bestmodel_summary.txt"
plot_file = Path(__file__).parent / "bestmodel_robustness.png"

# Collect all best checkpoint results
runs = []

if checkpoints_dir.exists():
    for config_file in sorted(checkpoints_dir.glob("*_config.json")):
        try:
            config = json.load(open(config_file))
            seed = config.get("seed", "?")
            best_val_pnl = config.get("best_val_pnl", 0)
            best_episode = config.get("best_episode", "?")

            runs.append({
                "seed": seed,
                "best_val_pnl": best_val_pnl,
                "best_episode": best_episode,
                "config": config,
                "config_file": str(config_file),
            })
        except Exception as e:
            print(f"Warning: Could not read {config_file}: {e}")

if not runs:
    print("No best checkpoint results found.")
    print("Run run_best_checkpoints.sh first.")
    print(f"(Looking in: {checkpoints_dir})")
    exit(1)

# Sort by seed
runs.sort(key=lambda x: (x["seed"] if isinstance(x["seed"], int) else 9999))

# Extract stats
best_pnls = [r["best_val_pnl"] for r in runs]

# Write text summary
with open(output_file, "w") as f:
    f.write("=" * 70 + "\n")
    f.write("Best Model Robustness Analysis\n")
    f.write(f"Total runs: {len(runs)} seeds\n")
    f.write("=" * 70 + "\n\n")

    # Get config from first run
    if runs:
        cfg = runs[0]["config"]
        f.write("Configuration:\n")
        f.write(f"  gamma={cfg.get('gamma')}, alpha={cfg.get('alpha')}\n")
        f.write(f"  episodes={cfg.get('episodes')}, epsilon_decay={cfg.get('epsilon_decay')}\n")
        f.write(f"  reward_shaping={cfg.get('reward_shaping_enabled')}\n")
        f.write(f"  potential_scale={cfg.get('potential_scale_factor')}\n")
        f.write("\n" + "-" * 70 + "\n\n")

    f.write("Per-Seed Results (Best Validation Checkpoint):\n\n")
    f.write(f"{'Seed':<6} {'Best Val PnL':>14} {'Best @ Episode':>16}\n")
    f.write("-" * 40 + "\n")

    for r in runs:
        f.write(f"{r['seed']:<6} {r['best_val_pnl']:>14,.0f} {str(r['best_episode']):>16}\n")

    f.write("\n" + "-" * 70 + "\n")
    f.write("\nAggregate Statistics (Best Validation PnL):\n")
    f.write(f"  Mean:   {np.mean(best_pnls):>12,.0f}\n")
    f.write(f"  Std:    {np.std(best_pnls):>12,.0f}\n")
    f.write(f"  Min:    {np.min(best_pnls):>12,.0f}\n")
    f.write(f"  Max:    {np.max(best_pnls):>12,.0f}\n")
    f.write(f"  Median: {np.median(best_pnls):>12,.0f}\n")

    f.write("\n" + "-" * 70 + "\n")
    f.write("\nRobustness Assessment:\n")
    cv = 100 * np.std(best_pnls) / np.mean(best_pnls) if np.mean(best_pnls) > 0 else float('inf')
    f.write(f"  Coefficient of Variation: {cv:.1f}%\n")
    if cv < 20:
        f.write("  --> Low variance: Model is robust across seeds\n")
    elif cv < 40:
        f.write("  --> Moderate variance: Some seed sensitivity\n")
    else:
        f.write("  --> High variance: Model performance varies significantly by seed\n")

    f.write("\n" + "-" * 70 + "\n")
    f.write("\nQ-tables for submission are in:\n")
    f.write(f"  {checkpoints_dir}/\n")
    f.write("\n" + "=" * 70 + "\n")

print(f"Summary saved to: {output_file}")

# Create robustness visualization
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Plot 1: Bar chart of best validation PnL per seed
ax1 = axes[0]
seeds = [r["seed"] for r in runs]
ax1.bar(range(len(runs)), best_pnls, alpha=0.7, color="steelblue", edgecolor="black")
ax1.axhline(np.mean(best_pnls), color="red", linestyle="--", linewidth=2,
            label=f"Mean: {np.mean(best_pnls):,.0f}")
ax1.set_xticks(range(len(runs)))
ax1.set_xticklabels([f"Seed {s}" for s in seeds], rotation=45, ha="right")
ax1.set_ylabel("Best Validation PnL")
ax1.set_title("Per-Seed Best Validation PnL")
ax1.legend(loc="upper right", fontsize=10)
ax1.grid(axis="y", alpha=0.3)

# Plot 2: Distribution histogram
ax2 = axes[1]
ax2.hist(best_pnls, bins=min(10, len(runs)), alpha=0.7, color="steelblue",
         edgecolor="black")
ax2.axvline(np.mean(best_pnls), color="red", linestyle="--", linewidth=2,
            label=f"Mean: {np.mean(best_pnls):,.0f}")
ax2.axvline(np.mean(best_pnls) - np.std(best_pnls), color="orange",
            linestyle=":", label=f"-1 Std: {np.mean(best_pnls) - np.std(best_pnls):,.0f}")
ax2.axvline(np.mean(best_pnls) + np.std(best_pnls), color="orange",
            linestyle=":", label=f"+1 Std: {np.mean(best_pnls) + np.std(best_pnls):,.0f}")
ax2.set_xlabel("Best Validation PnL")
ax2.set_ylabel("Count")
ax2.set_title("PnL Distribution Across Seeds")
ax2.legend(fontsize=8)

fig.suptitle("Best Model Robustness Analysis", fontsize=14, fontweight="bold")
fig.tight_layout(rect=[0, 0, 1, 0.95])
fig.savefig(plot_file, dpi=200)
plt.close(fig)

print(f"Robustness plot saved to: {plot_file}")
