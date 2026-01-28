"""Summarize Q-learning results by scale."""
import json
from pathlib import Path
from collections import defaultdict
import numpy as np

results_dir = Path(__file__).parent.parent / "results" / "qlearning"
output_file = Path(__file__).parent / "results_summary.txt"

# Group results by scale
by_scale = defaultdict(list)

for exp_dir in sorted(results_dir.iterdir()):
    if not exp_dir.is_dir():
        continue
    config_path = exp_dir / "config.json"
    eval_path = exp_dir / "eval_metrics.json"
    if not (config_path.exists() and eval_path.exists()):
        continue

    config = json.load(open(config_path))
    eval_m = json.load(open(eval_path))

    if not config.get("reward_shaping_enabled"):
        continue

    # Only include sweep results (label pattern: s{scale}_seed{N})
    label = config.get("label", "")
    if not (label and "_seed" in label):
        continue

    scale = config.get("potential_scale_factor", 10.0)
    seed = config.get("seed", "?")
    pnl = eval_m.get("validation_pnl", 0)
    buys = eval_m.get("action_counts", {}).get("buy", 0)

    by_scale[scale].append({"seed": seed if seed is not None else "?", "pnl": pnl, "buys": buys})

# Write summary
with open(output_file, "w") as f:
    f.write("=" * 60 + "\n")
    f.write("Q-Learning Reward Shaping Results Summary\n")
    f.write("gamma=0.9, episodes=160, reward_shaping=True\n")
    f.write("=" * 60 + "\n\n")

    for scale in sorted(by_scale.keys()):
        runs = by_scale[scale]
        runs.sort(key=lambda x: (x["seed"] if isinstance(x["seed"], int) else 9999))

        pnls = [r["pnl"] for r in runs]
        mean_pnl = np.mean(pnls)
        std_pnl = np.std(pnls)

        f.write(f"Results: Scale={scale}, Episodes=160 ({len(runs)} Seeds)\n\n")

        for r in runs:
            f.write(f"Seed {r['seed']}: {r['pnl']:,.0f} PnL | {r['buys']:,} buys\n")

        f.write(f"\nStats:\n")
        f.write(f"Mean: {mean_pnl:,.0f} ± {std_pnl:,.0f}\n")
        f.write(f"Range: {min(pnls):,.0f} – {max(pnls):,.0f}\n")
        f.write(f"Average: ~{mean_pnl/1000:.0f}k ({100*mean_pnl/40000:.0f}% of 40k target)\n")
        f.write(f"Best: ~{max(pnls)/1000:.0f}k ({100*max(pnls)/40000:.0f}% of target)\n")
        f.write(f"Worst: ~{min(pnls)/1000:.0f}k ({100*min(pnls)/40000:.0f}% of target)\n")
        if mean_pnl > 0:
            f.write(f"Variance: ±{100*std_pnl/mean_pnl:.0f}% around mean\n")

        f.write(f"\nConfig:\n")
        f.write(f"gamma=0.9, potential_scale={scale}, episodes=160, reward_shaping=True\n")
        f.write("\n" + "-" * 60 + "\n\n")

print(f"Summary saved to: {output_file}")
