"""
Multi-seed hyperparameter tuning with alpha decay tracking and visualization.

Key additions:
- Alpha decay properly tracked in config labels
- Displayed in all plots and tables
- Saved in JSON summary
- Shows in recommendations
"""

from __future__ import annotations

from dataclasses import replace
from datetime import datetime
from pathlib import Path
import json
from typing import Optional

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.patches import Rectangle
import seaborn as sns

# IMPORTANT: change this import to your actual module name
from qtable_feats_potential import *

PROJECT_ROOT = Path(__file__).resolve().parent.parent
RESULTS_DIR = PROJECT_ROOT / "results" / "qlearning_tuning"

# Set style for better-looking plots
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")

# FIXED SEEDS FOR REPRODUCIBILITY
SEEDS = [42, 123, 456]


def ensure_dir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)


def pick_eval_every(episodes: int, frac: float = 0.05, min_every: int = 25) -> int:
    return max(min_every, int(round(episodes * frac)))


def format_config_label(cfg: QConfig) -> str:
    """
    Create a comprehensive label for the configuration that includes all key parameters.
    This ensures alpha decay is visible everywhere.
    """
    parts = []
    
    # Add alpha (with decay if applicable)
    if hasattr(cfg, 'alpha_decay') and cfg.alpha_decay is not None and cfg.alpha_decay != 1.0:
        parts.append(f"α={cfg.alpha:.3f}(decay={cfg.alpha_decay:.4f})")
    else:
        parts.append(f"α={cfg.alpha:.3f}")
    
    # Add gamma
    parts.append(f"γ={cfg.gamma:.3f}")
    
    # Add episodes if non-standard
    if hasattr(cfg, 'episodes') and cfg.episodes != 250:
        parts.append(f"ep={cfg.episodes}")
    
    # Add other relevant params if they vary
    if hasattr(cfg, 'epsilon_decay') and cfg.epsilon_decay is not None and cfg.epsilon_decay != 0.995:
        parts.append(f"ε_decay={cfg.epsilon_decay:.4f}")
    
    return ", ".join(parts)


def val_curve_from_history(val_pnl_history: list[tuple[int, float]]):
    if not val_pnl_history:
        return [], []
    xs = [int(ep) for ep, _ in val_pnl_history]
    ys = [float(pnl) for _, pnl in val_pnl_history]
    return xs, ys


def summarize_run(training_result, eval_result) -> dict:
    xs, ys = val_curve_from_history(training_result.val_pnl_history)
    best_pnl = max(ys) if ys else float(eval_result.total_pnl)
    best_ep = xs[int(np.argmax(ys))] if ys else None

    rewards = training_result.episode_rewards
    td = training_result.td_error_history

    # Calculate additional stability metrics
    if ys and len(ys) > 10:
        last_10_std = np.std(ys[-10:])
        last_10_mean = np.mean(ys[-10:])
    else:
        last_10_std = None
        last_10_mean = None

    return {
        "final_val_pnl": float(eval_result.total_pnl),
        "best_val_pnl_during_training": float(best_pnl),
        "best_val_pnl_episode": best_ep,
        "final_episode_reward": float(rewards[-1]) if rewards else None,
        "mean_reward_last_50": float(np.mean(rewards[-50:])) if len(rewards) >= 50 else None,
        "mean_td_error_last_50": float(np.mean(td[-50:])) if len(td) >= 50 else None,
        "last_10_val_std": float(last_10_std) if last_10_std is not None else None,
        "last_10_val_mean": float(last_10_mean) if last_10_mean is not None else None,
        "action_counts": eval_result.action_counts,
        "all_val_pnls": ys,
    }


def run_one(train_features: dict, val_features: dict, cfg: QConfig) -> dict:
    """Run single training iteration with one seed."""
    tr = train_q_learning(train_features, cfg, val_features=val_features)

    ev = evaluate(
        tr.Q,
        val_features,
        tr.storage_bins,
        tr.price_bins,
        price_window=cfg.price_window,
        record=False,
    )

    summary = summarize_run(tr, ev)

    return {
        "config": cfg,
        "summary": summary,
        "val_pnl_history": tr.val_pnl_history,
        "episode_rewards": tr.episode_rewards,
        "td_error_history": tr.td_error_history,
    }


def run_with_multiple_seeds(train_features: dict, val_features: dict, cfg: QConfig, seeds: list[int], label: str = None) -> dict:
    """
    Run same configuration with multiple seeds and aggregate results.
    Returns mean, std, and all individual runs.
    
    Args:
        label: Custom label for this config (will include alpha decay if present)
    """
    all_runs = []
    
    for seed in seeds:
        cfg_with_seed = replace(cfg, seed=seed)
        run = run_one(train_features, val_features, cfg_with_seed)
        all_runs.append(run)
    
    # Aggregate metrics across seeds
    final_pnls = [r["summary"]["final_val_pnl"] for r in all_runs]
    best_pnls = [r["summary"]["best_val_pnl_during_training"] for r in all_runs]
    stabilities = [r["summary"]["last_10_val_std"] for r in all_runs if r["summary"]["last_10_val_std"] is not None]
    
    # Aggregate validation curves
    all_val_curves = []
    for r in all_runs:
        xs, ys = val_curve_from_history(r["val_pnl_history"])
        if xs:
            all_val_curves.append((xs, ys))
    
    # Compute mean and std of validation curves
    if all_val_curves:
        ref_xs = all_val_curves[0][0]
        aligned_ys = []
        
        for xs, ys in all_val_curves:
            if xs == ref_xs:
                aligned_ys.append(ys)
            else:
                aligned_ys.append(np.interp(ref_xs, xs, ys))
        
        mean_val_curve = np.mean(aligned_ys, axis=0)
        std_val_curve = np.std(aligned_ys, axis=0)
        val_curve_history = [(int(x), float(y)) for x, y in zip(ref_xs, mean_val_curve)]
    else:
        mean_val_curve = []
        std_val_curve = []
        val_curve_history = []
    
    # Use provided label or generate from config
    if label is None:
        label = format_config_label(cfg)
    
    return {
        "config": cfg,
        "label": label,
        "seeds": seeds,
        "all_runs": all_runs,
        "aggregated_summary": {
            "mean_final_pnl": float(np.mean(final_pnls)),
            "std_final_pnl": float(np.std(final_pnls)),
            "min_final_pnl": float(np.min(final_pnls)),
            "max_final_pnl": float(np.max(final_pnls)),
            "mean_best_pnl": float(np.mean(best_pnls)),
            "std_best_pnl": float(np.std(best_pnls)),
            "mean_stability": float(np.mean(stabilities)) if stabilities else None,
            "std_stability": float(np.std(stabilities)) if stabilities else None,
        },
        "mean_val_curve": mean_val_curve,
        "std_val_curve": std_val_curve,
        "val_pnl_history": val_curve_history,
    }


# ============================================================================
# VISUALIZATION FUNCTIONS (SAME AS BEFORE BUT USE LABELS)
# ============================================================================

def plot_learning_curves_with_confidence(runs: list[dict], title: str, outpath: Path, 
                                        show_config_details: bool = True) -> None:
    """Plot learning curves with confidence bands showing variance across seeds."""
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 10))
    
    # Top plot: Learning curves with confidence bands
    best_run = None
    best_mean_pnl = -float('inf')
    
    for r in runs:
        if len(r["mean_val_curve"]) == 0:
            continue
            
        label = r["label"]
        mean_pnl = r["aggregated_summary"]["mean_final_pnl"]
        std_pnl = r["aggregated_summary"]["std_final_pnl"]
        
        # Track best run
        if mean_pnl > best_mean_pnl:
            best_mean_pnl = mean_pnl
            best_run = label
        
        # Add performance info to label
        if show_config_details:
            label_with_stats = f"{label} | μ={mean_pnl:.1f}±{std_pnl:.1f}"
        else:
            label_with_stats = f"{label} (μ={mean_pnl:.1f}±{std_pnl:.1f})"
        
        linewidth = 3 if mean_pnl == best_mean_pnl else 1.5
        
        # Plot mean curve
        xs = np.arange(len(r["mean_val_curve"]))
        mean_curve = r["mean_val_curve"]
        std_curve = r["std_val_curve"]
        
        line = ax1.plot(xs, mean_curve, marker="o", markersize=4, 
                       label=label_with_stats, linewidth=linewidth, alpha=0.8)[0]
        
        # Add confidence band (±1 std)
        ax1.fill_between(xs, 
                        np.array(mean_curve) - np.array(std_curve),
                        np.array(mean_curve) + np.array(std_curve),
                        alpha=0.2, color=line.get_color())
    
    ax1.set_xlabel("Evaluation Index", fontsize=12)
    ax1.set_ylabel("Validation PnL (mean ± std)", fontsize=12)
    ax1.set_title(f"{title}\n(Best: {best_run}) | Seeds: {SEEDS}", 
                  fontsize=14, fontweight='bold')
    ax1.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=8)
    ax1.grid(True, alpha=0.3)
    
    # Bottom plot: Mean final PnL with error bars
    labels = [r["label"] for r in runs]
    mean_pnls = [r["aggregated_summary"]["mean_final_pnl"] for r in runs]
    std_pnls = [r["aggregated_summary"]["std_final_pnl"] for r in runs]
    mean_best_pnls = [r["aggregated_summary"]["mean_best_pnl"] for r in runs]
    
    x = np.arange(len(labels))
    width = 0.35
    
    bars1 = ax2.bar(x - width/2, mean_pnls, width, yerr=std_pnls, 
                   label='Mean Final PnL', alpha=0.8, capsize=5)
    bars2 = ax2.bar(x + width/2, mean_best_pnls, width,
                   label='Mean Best PnL', alpha=0.8)
    
    # Highlight best performer
    best_idx = np.argmax(mean_pnls)
    bars1[best_idx].set_edgecolor('red')
    bars1[best_idx].set_linewidth(3)
    
    ax2.set_xlabel("Configuration", fontsize=12)
    ax2.set_ylabel("PnL", fontsize=12)
    ax2.set_title("Mean Performance Across Seeds (error bars = ±1 std)", fontsize=12, fontweight='bold')
    ax2.set_xticks(x)
    ax2.set_xticklabels(labels, rotation=45, ha='right', fontsize=8)
    ax2.legend()
    ax2.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    plt.savefig(outpath, dpi=300, bbox_inches='tight')
    plt.close()


def plot_comparison_table_multiseed(runs: list[dict], title: str, outpath: Path) -> None:
    """Create a detailed comparison table for multi-seed runs with full config details."""
    fig, ax = plt.subplots(figsize=(20, max(8, len(runs) * 0.5)))
    ax.axis('tight')
    ax.axis('off')
    
    # Prepare data - now with full config label
    headers = ['Configuration', 'Mean PnL', 'Std PnL', 'Mean Best', 'CV', 'Stability', 'Hold', 'Buy', 'Sell']
    rows = []
    
    for r in runs:
        agg = r["aggregated_summary"]
        # Average action counts across seeds
        all_action_counts = [run["summary"]["action_counts"] for run in r["all_runs"]]
        avg_holds = np.mean([ac.get('HOLD', 0) for ac in all_action_counts])
        avg_buys = np.mean([ac.get('BUY', 0) for ac in all_action_counts])
        avg_sells = np.mean([ac.get('SELL', 0) for ac in all_action_counts])
        
        cv = agg["std_final_pnl"] / abs(agg["mean_final_pnl"]) if agg["mean_final_pnl"] != 0 else 0
        
        rows.append([
            r["label"],  # Full label including alpha decay
            f"{agg['mean_final_pnl']:.2f}",
            f"{agg['std_final_pnl']:.2f}",
            f"{agg['mean_best_pnl']:.2f}",
            f"{cv:.3f}",
            f"{agg['mean_stability']:.2f}" if agg['mean_stability'] is not None else "N/A",
            f"{avg_holds:.0f}",
            f"{avg_buys:.0f}",
            f"{avg_sells:.0f}",
        ])
    
    # Create table
    table = ax.table(cellText=rows, colLabels=headers, cellLoc='center', loc='center',
                    colWidths=[0.20, 0.10, 0.10, 0.10, 0.08, 0.10, 0.08, 0.08, 0.08])
    
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1, 2)
    
    # Style header
    for i in range(len(headers)):
        cell = table[(0, i)]
        cell.set_facecolor('#4CAF50')
        cell.set_text_props(weight='bold', color='white')
    
    # Highlight best performer in each metric column
    metrics_cols = [1, 3, 4, 5]  # Mean PnL, Mean Best, CV (lower better), Stability (lower better)
    for col_idx in metrics_cols:
        values = []
        for row_idx in range(1, len(rows) + 1):
            cell_text = table[(row_idx, col_idx)].get_text().get_text()
            try:
                values.append(float(cell_text))
            except:
                values.append(None)
        
        if any(v is not None for v in values):
            if col_idx in [4, 5]:  # CV and Stability (lower is better)
                best_idx = np.nanargmin([v if v is not None else np.inf for v in values])
            else:
                best_idx = np.nanargmax([v if v is not None else -np.inf for v in values])
            
            table[(best_idx + 1, col_idx)].set_facecolor('#90EE90')
    
    # Alternate row colors
    for i in range(1, len(rows) + 1):
        for j in range(len(headers)):
            if table[(i, j)].get_facecolor()[0:3] == (1.0, 1.0, 1.0):
                if i % 2 == 0:
                    table[(i, j)].set_facecolor('#f0f0f0')
    
    plt.title(title + f"\nSeeds: {SEEDS}", 
             fontsize=16, fontweight='bold', pad=20)
    plt.savefig(outpath, dpi=300, bbox_inches='tight')
    plt.close()


def generate_recommendation_report_multiseed(all_runs: dict, outpath: Path) -> dict:
    """Analyze all runs and generate best parameter recommendations with full config details."""
    recommendations = {}
    
    for sweep_name, runs in all_runs.items():
        if not runs:
            continue
        
        # Find best by different criteria
        best_mean = max(runs, key=lambda r: r["aggregated_summary"]["mean_final_pnl"])
        best_peak = max(runs, key=lambda r: r["aggregated_summary"]["mean_best_pnl"])
        
        # Most reproducible (lowest CV among top performers)
        top_performers = sorted(runs, key=lambda r: r["aggregated_summary"]["mean_final_pnl"], 
                               reverse=True)[:max(3, len(runs)//3)]
        most_reproducible = min(top_performers, 
                               key=lambda r: r["aggregated_summary"]["std_final_pnl"] / 
                                           abs(r["aggregated_summary"]["mean_final_pnl"]) 
                                           if r["aggregated_summary"]["mean_final_pnl"] != 0 else np.inf)
        
        recommendations[sweep_name] = {
            "best_mean_pnl": {
                "config_label": best_mean["label"],  # Full label with decay
                "config_dict": best_mean["config"].__dict__,
                "mean": best_mean["aggregated_summary"]["mean_final_pnl"],
                "std": best_mean["aggregated_summary"]["std_final_pnl"],
            },
            "best_peak_pnl": {
                "config_label": best_peak["label"],
                "config_dict": best_peak["config"].__dict__,
                "mean": best_peak["aggregated_summary"]["mean_best_pnl"],
                "std": best_peak["aggregated_summary"]["std_best_pnl"],
            },
            "most_reproducible_top_performer": {
                "config_label": most_reproducible["label"],
                "config_dict": most_reproducible["config"].__dict__,
                "mean": most_reproducible["aggregated_summary"]["mean_final_pnl"],
                "std": most_reproducible["aggregated_summary"]["std_final_pnl"],
                "cv": most_reproducible["aggregated_summary"]["std_final_pnl"] / 
                     abs(most_reproducible["aggregated_summary"]["mean_final_pnl"]),
            }
        }
    
    # Generate text report with full config details
    report_lines = ["=" * 100]
    report_lines.append("HYPERPARAMETER TUNING RECOMMENDATIONS (MULTI-SEED)")
    report_lines.append(f"Seeds used: {SEEDS}")
    report_lines.append("=" * 100)
    report_lines.append("")
    
    for sweep_name, rec in recommendations.items():
        report_lines.append(f"\n{sweep_name.upper().replace('_', ' ')}:")
        report_lines.append("-" * 80)
        
        report_lines.append(f"\n  ⭐ Best Mean PnL: {rec['best_mean_pnl']['config_label']}")
        report_lines.append(f"     Mean PnL: {rec['best_mean_pnl']['mean']:.2f} ± {rec['best_mean_pnl']['std']:.2f}")
        report_lines.append(f"     Full config: {rec['best_mean_pnl']['config_dict']}")
        
        report_lines.append(f"\n  📈 Best Peak PnL: {rec['best_peak_pnl']['config_label']}")
        report_lines.append(f"     Mean Peak: {rec['best_peak_pnl']['mean']:.2f} ± {rec['best_peak_pnl']['std']:.2f}")
        report_lines.append(f"     Full config: {rec['best_peak_pnl']['config_dict']}")
        
        report_lines.append(f"\n  🎯 Most Reproducible (Top 33%): {rec['most_reproducible_top_performer']['config_label']}")
        report_lines.append(f"     Mean PnL: {rec['most_reproducible_top_performer']['mean']:.2f} ± {rec['most_reproducible_top_performer']['std']:.2f}")
        report_lines.append(f"     CV: {rec['most_reproducible_top_performer']['cv']:.3f}")
        report_lines.append(f"     Full config: {rec['most_reproducible_top_performer']['config_dict']}")
    
    report_lines.append("\n" + "=" * 100)
    report_lines.append("\nKEY INSIGHTS:")
    report_lines.append("- Look for configs with high mean PnL AND low std dev")
    report_lines.append("- Low CV (coefficient of variation) = more reproducible results")
    report_lines.append("- Alpha decay shown in labels as: α=X.XXX(decay=Y.YYYY)")
    report_lines.append("- Full config dict shows all parameters used")
    report_lines.append("=" * 100)
    
    report_text = "\n".join(report_lines)
    
    with open(outpath, 'w') as f:
        f.write(report_text)
    
    print("\n" + report_text)
    
    return recommendations


# ============================================================================
# MAIN EXECUTION
# ============================================================================

def main():
    print("=" * 80)
    print("MULTI-SEED Q-LEARNING HYPERPARAMETER TUNING")
    print(f"Running each config with seeds: {SEEDS}")
    print("=" * 80)
    
    print("\n[1/4] Loading data...")
    train_features = load_data("train.xlsx")
    val_features = load_data("validate.xlsx")

    # Base configuration
    # IMPORTANT: Set your alpha_decay here if your QConfig supports it
    base = QConfig(
        episodes=250,
        eval_every=20,
        seed=42,  # Will be overridden
        track_visits=False,
        track_td_error=True,
        reward_shaping_enabled=False,
        # alpha_decay=0.9995,  # UNCOMMENT if you have this parameter
    )

    # Optimized parameter grids
    gammas = [0.90, 0.95, 0.99, 0.995, 0.999]
    alphas = [0.5, 0.1, 0.05, 0.01, 0.005]
    
    # If you want to test different alpha decays, add them here:
    # alpha_decays = [1.0, 0.9999, 0.9995, 0.999]  # 1.0 = no decay

    # Create output directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    outdir = RESULTS_DIR / f"tune_multiseed_{timestamp}"
    ensure_dir(outdir)
    print(f"\nSaving outputs to: {outdir}")
    
    # Print config info
    print("\nBase configuration:")
    print(f"  Alpha: {base.alpha}")
    if hasattr(base, 'alpha_decay'):
        print(f"  Alpha decay: {base.alpha_decay}")
    print(f"  Gamma: {base.gamma}")
    print(f"  Episodes: {base.episodes}")
    print(f"  Seeds: {SEEDS}")

    all_runs = {}

    # ========================================================================
    # GAMMA SWEEP
    # ========================================================================
    print("\n[2/4] Running gamma sweep with multiple seeds...")
    gamma_runs = []
    for i, g in enumerate(gammas, 1):
        cfg = replace(base, gamma=g)
        print(f"  [{i}/{len(gammas)}] Testing {format_config_label(cfg)}")
        print(f"    Running with seeds: {SEEDS}")
        
        multiseed_run = run_with_multiple_seeds(
            train_features, val_features, cfg, SEEDS,
            label=f"γ={g:.3f}"  # Simple label for gamma sweep
        )
        gamma_runs.append(multiseed_run)
        
        agg = multiseed_run["aggregated_summary"]
        print(f"    → Mean PnL: {agg['mean_final_pnl']:.2f} ± {agg['std_final_pnl']:.2f}")

    all_runs['gamma_sweep'] = gamma_runs
    
    print("  Generating gamma sweep visualizations...")
    plot_learning_curves_with_confidence(
        gamma_runs,
        f"Gamma Sweep (α={base.alpha}" + 
        (f", α_decay={base.alpha_decay}" if hasattr(base, 'alpha_decay') and base.alpha_decay != 1.0 else "") + ")",
        outdir / "gamma_sweep_learning_curves.png"
    )
    plot_comparison_table_multiseed(
        gamma_runs,
        f"Gamma Sweep: Detailed Comparison",
        outdir / "gamma_sweep_table.png"
    )

    # ========================================================================
    # ALPHA SWEEP (with episode adjustment for small alphas)
    # ========================================================================
    print("\n[3/4] Running alpha sweep with multiple seeds...")
    alpha_runs = []
    for i, a in enumerate(alphas, 1):
        # Use 300 episodes for two smallest learning rates
        episodes = 300 if a in alphas[-2:] else 250
        eval_every = pick_eval_every(episodes, frac=0.05, min_every=25)
        
        cfg = replace(base, alpha=a, episodes=episodes, eval_every=eval_every)
        
        # Create label showing decay if present
        if hasattr(cfg, 'alpha_decay') and cfg.alpha_decay != 1.0:
            label = f"α={a:.3f}(decay={cfg.alpha_decay:.4f})"
        else:
            label = f"α={a:.3f}"
        
        if episodes != 250:
            label += f", ep={episodes}"
        
        print(f"  [{i}/{len(alphas)}] Testing {format_config_label(cfg)}")
        print(f"    Running with seeds: {SEEDS}")
        
        multiseed_run = run_with_multiple_seeds(
            train_features, val_features, cfg, SEEDS,
            label=label
        )
        alpha_runs.append(multiseed_run)
        
        agg = multiseed_run["aggregated_summary"]
        print(f"    → Mean PnL: {agg['mean_final_pnl']:.2f} ± {agg['std_final_pnl']:.2f}")

    all_runs['alpha_sweep'] = alpha_runs
    
    print("  Generating alpha sweep visualizations...")
    plot_learning_curves_with_confidence(
        alpha_runs,
        f"Alpha Sweep (γ={base.gamma})",
        outdir / "alpha_sweep_learning_curves.png",
        show_config_details=True
    )
    plot_comparison_table_multiseed(
        alpha_runs,
        f"Alpha Sweep: Detailed Comparison",
        outdir / "alpha_sweep_table.png"
    )

    # ========================================================================
    # 2D GRID SEARCH: ALPHA vs GAMMA
    # ========================================================================
    print("\n[4/4] Running 2D grid search (alpha vs gamma) with multiple seeds...")
    
    # Smaller 2D grid for efficiency
    alpha_grid_2d = [0.05, 0.10, 0.20, 0.50]
    gamma_grid_2d = [0.93, 0.95, 0.97, 0.99, 0.995]
    
    grid_runs = []
    total_runs = len(alpha_grid_2d) * len(gamma_grid_2d)
    run_count = 0
    
    for g in gamma_grid_2d:
        for a in alpha_grid_2d:
            run_count += 1
            cfg = replace(base, alpha=a, gamma=g)
            
            label = format_config_label(cfg)
            print(f"  [{run_count}/{total_runs}] Testing {label}")
            print(f"    Running with seeds: {SEEDS}")
            
            multiseed_run = run_with_multiple_seeds(
                train_features, val_features, cfg, SEEDS,
                label=label
            )
            grid_runs.append(multiseed_run)
            
            agg = multiseed_run["aggregated_summary"]
            print(f"    → Mean PnL: {agg['mean_final_pnl']:.2f} ± {agg['std_final_pnl']:.2f}")
    
    all_runs['grid_2d_alpha_gamma'] = grid_runs

    # ========================================================================
    # GENERATE RECOMMENDATIONS
    # ========================================================================
    print("\nGenerating recommendations and summary...")
    recommendations = generate_recommendation_report_multiseed(
        all_runs,
        outdir / "RECOMMENDATIONS.txt"
    )

    # ========================================================================
    # SAVE COMPREHENSIVE JSON SUMMARY
    # ========================================================================
    def make_json_serializable(obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, dict):
            return {k: make_json_serializable(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [make_json_serializable(item) for item in obj]
        return obj

    summary = {
        "timestamp": timestamp,
        "seeds_used": SEEDS,
        "base_config": base.__dict__,
        "parameter_grids": {
            "gammas": gammas,
            "alphas": alphas,
            "alpha_grid_2d": alpha_grid_2d,
            "gamma_grid_2d": gamma_grid_2d,
        },
        "sweeps": {},
        "recommendations": make_json_serializable(recommendations),
    }
    
    for sweep_name, runs in all_runs.items():
        summary["sweeps"][sweep_name] = [
            {
                "label": r["label"],  # Now includes alpha decay
                "config": r["config"].__dict__,
                "aggregated_summary": make_json_serializable(r["aggregated_summary"]),
                "seeds": r["seeds"],
            }
            for r in runs
        ]
    
    with open(outdir / "tuning_summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    # ========================================================================
    # FINAL SUMMARY
    # ========================================================================
    print("\n" + "=" * 80)
    print("MULTI-SEED TUNING COMPLETE!")
    print("=" * 80)
    print(f"\nAll results saved to: {outdir}")
    print(f"\nSeeds used: {SEEDS}")
    print("\nGenerated visualizations:")
    for p in sorted(outdir.glob("*.png")):
        print(f"  ✓ {p.name}")
    print(f"\n  ✓ {(outdir / 'tuning_summary.json').name}")
    print(f"  ✓ {(outdir / 'RECOMMENDATIONS.txt').name}")
    print("\n" + "=" * 80)
    print("\nALPHA DECAY INFO:")
    if hasattr(base, 'alpha_decay'):
        print(f"  Alpha decay used: {base.alpha_decay}")
        print(f"  Shown in labels as: α=X.XXX(decay={base.alpha_decay:.4f})")
    else:
        print("  No alpha decay in QConfig (or alpha_decay=1.0)")
    print("=" * 80)


if __name__ == "__main__":
    main()