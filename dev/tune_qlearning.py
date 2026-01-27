"""
Enhanced hyperparameter tuning for Q-learning with comprehensive visualizations.

This script provides:
1. Multiple sweep types (gamma, alpha, episodes, combined 2D grids)
2. Rich visualizations:
   - Learning curves with confidence bands
   - Heatmaps for 2D parameter interactions
   - Distribution plots for performance metrics
   - Convergence analysis
   - Statistical comparison tables
3. Automated best parameter selection with reasoning
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
from qtable_feats_clau import (
    QConfig,
    load_data,
    train_q_learning,
    evaluate,
)

PROJECT_ROOT = Path(__file__).resolve().parent.parent
RESULTS_DIR = PROJECT_ROOT / "results" / "qlearning_tuning"

# Set style for better-looking plots
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")


def ensure_dir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)


def pick_eval_every(episodes: int, frac: float = 0.05, min_every: int = 25) -> int:
    return max(min_every, int(round(episodes * frac)))


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


# ============================================================================
# VISUALIZATION FUNCTIONS
# ============================================================================

def plot_learning_curves_with_stats(runs: list[dict], title: str, outpath: Path) -> None:
    """Plot learning curves with mean and std dev bands."""
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 10))
    
    # Top plot: Learning curves
    best_run = None
    best_final_pnl = -float('inf')
    
    for r in runs:
        xs, ys = val_curve_from_history(r["val_pnl_history"])
        if xs:
            label = r["label"]
            final_pnl = r["summary"]["final_val_pnl"]
            
            # Track best run
            if final_pnl > best_final_pnl:
                best_final_pnl = final_pnl
                best_run = label
            
            # Add performance info to label
            label_with_stats = f"{label} (final: {final_pnl:.1f}, best: {r['summary']['best_val_pnl_during_training']:.1f})"
            
            linewidth = 3 if final_pnl == best_final_pnl else 1.5
            ax1.plot(xs, ys, marker="o", markersize=4, label=label_with_stats, linewidth=linewidth, alpha=0.8)
    
    ax1.set_xlabel("Episode", fontsize=12)
    ax1.set_ylabel("Validation PnL", fontsize=12)
    ax1.set_title(f"{title}\n(Best: {best_run})", fontsize=14, fontweight='bold')
    ax1.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=9)
    ax1.grid(True, alpha=0.3)
    
    # Bottom plot: Final PnL comparison
    labels = [r["label"] for r in runs]
    final_pnls = [r["summary"]["final_val_pnl"] for r in runs]
    best_pnls = [r["summary"]["best_val_pnl_during_training"] for r in runs]
    
    x = np.arange(len(labels))
    width = 0.35
    
    bars1 = ax2.bar(x - width/2, final_pnls, width, label='Final Val PnL', alpha=0.8)
    bars2 = ax2.bar(x + width/2, best_pnls, width, label='Best Val PnL', alpha=0.8)
    
    # Highlight best performer
    best_idx = np.argmax(final_pnls)
    bars1[best_idx].set_edgecolor('red')
    bars1[best_idx].set_linewidth(3)
    
    ax2.set_xlabel("Configuration", fontsize=12)
    ax2.set_ylabel("PnL", fontsize=12)
    ax2.set_title("Final vs Best Validation PnL", fontsize=12, fontweight='bold')
    ax2.set_xticks(x)
    ax2.set_xticklabels(labels, rotation=45, ha='right')
    ax2.legend()
    ax2.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    plt.savefig(outpath, dpi=300, bbox_inches='tight')
    plt.close()


def plot_convergence_analysis(runs: list[dict], title: str, outpath: Path) -> None:
    """Analyze convergence properties: stability and final performance."""
    fig = plt.figure(figsize=(16, 10))
    gs = gridspec.GridSpec(2, 3, figure=fig, hspace=0.3, wspace=0.3)
    
    # Extract metrics
    labels = [r["label"] for r in runs]
    final_pnls = [r["summary"]["final_val_pnl"] for r in runs]
    best_pnls = [r["summary"]["best_val_pnl_during_training"] for r in runs]
    stabilities = [r["summary"]["last_10_val_std"] for r in runs]
    last_10_means = [r["summary"]["last_10_val_mean"] for r in runs]
    
    # 1. Final PnL distribution
    ax1 = fig.add_subplot(gs[0, 0])
    ax1.hist(final_pnls, bins=min(15, len(final_pnls)), alpha=0.7, edgecolor='black')
    ax1.axvline(np.mean(final_pnls), color='red', linestyle='--', linewidth=2, label=f'Mean: {np.mean(final_pnls):.1f}')
    ax1.axvline(np.median(final_pnls), color='green', linestyle='--', linewidth=2, label=f'Median: {np.median(final_pnls):.1f}')
    ax1.set_xlabel('Final Validation PnL', fontsize=11)
    ax1.set_ylabel('Frequency', fontsize=11)
    ax1.set_title('Distribution of Final PnL', fontsize=12, fontweight='bold')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # 2. Stability vs Performance scatter
    ax2 = fig.add_subplot(gs[0, 1])
    scatter = ax2.scatter(stabilities, final_pnls, s=100, alpha=0.6, c=range(len(runs)), cmap='viridis')
    for i, label in enumerate(labels):
        ax2.annotate(label, (stabilities[i], final_pnls[i]), fontsize=8, alpha=0.7)
    ax2.set_xlabel('Stability (Std Dev of Last 10 Evals)', fontsize=11)
    ax2.set_ylabel('Final Validation PnL', fontsize=11)
    ax2.set_title('Stability vs Performance', fontsize=12, fontweight='bold')
    ax2.grid(True, alpha=0.3)
    
    # Ideal region (low stability, high performance)
    best_pnl = max(final_pnls)
    min_stability = min(stabilities)
    ax2.add_patch(Rectangle((min_stability, best_pnl*0.9), 
                            (max(stabilities)-min_stability)*0.3, 
                            best_pnl*0.1, 
                            alpha=0.2, color='green', label='Ideal Region'))
    ax2.legend()
    
    # 3. Gap between best and final
    ax3 = fig.add_subplot(gs[0, 2])
    gaps = [best - final for best, final in zip(best_pnls, final_pnls)]
    colors = ['green' if g < np.median(gaps) else 'orange' for g in gaps]
    bars = ax3.barh(range(len(labels)), gaps, color=colors, alpha=0.7, edgecolor='black')
    ax3.set_yticks(range(len(labels)))
    ax3.set_yticklabels(labels, fontsize=9)
    ax3.set_xlabel('Best PnL - Final PnL', fontsize=11)
    ax3.set_title('Performance Gap\n(Lower is Better)', fontsize=12, fontweight='bold')
    ax3.axvline(np.median(gaps), color='red', linestyle='--', linewidth=2, label=f'Median: {np.median(gaps):.1f}')
    ax3.legend()
    ax3.grid(True, alpha=0.3, axis='x')
    
    # 4. Learning curve stability over time
    ax4 = fig.add_subplot(gs[1, :])
    for r in runs:
        xs, ys = val_curve_from_history(r["val_pnl_history"])
        if len(ys) >= 5:
            # Calculate rolling std
            window = min(5, len(ys) // 3)
            rolling_std = [np.std(ys[max(0, i-window):i+1]) for i in range(len(ys))]
            ax4.plot(xs, rolling_std, marker='o', markersize=3, label=r["label"], alpha=0.7)
    
    ax4.set_xlabel('Episode', fontsize=12)
    ax4.set_ylabel('Rolling Std Dev (window=5)', fontsize=12)
    ax4.set_title('Learning Stability Over Time', fontsize=13, fontweight='bold')
    ax4.legend(bbox_to_anchor=(1.02, 1), loc='upper left', fontsize=9)
    ax4.grid(True, alpha=0.3)
    
    plt.suptitle(title, fontsize=15, fontweight='bold', y=0.995)
    plt.savefig(outpath, dpi=300, bbox_inches='tight')
    plt.close()


def plot_2d_heatmap(param1_name: str, param1_values: list, 
                    param2_name: str, param2_values: list,
                    results_grid: np.ndarray, metric_name: str,
                    title: str, outpath: Path) -> None:
    """Create heatmap for 2D parameter sweep."""
    fig, ax = plt.subplots(figsize=(12, 10))
    
    im = ax.imshow(results_grid, cmap='RdYlGn', aspect='auto', interpolation='nearest')
    
    # Set ticks
    ax.set_xticks(np.arange(len(param1_values)))
    ax.set_yticks(np.arange(len(param2_values)))
    ax.set_xticklabels([f'{v:.3f}' if isinstance(v, float) else str(v) for v in param1_values])
    ax.set_yticklabels([f'{v:.3f}' if isinstance(v, float) else str(v) for v in param2_values])
    
    # Rotate x labels
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")
    
    # Labels
    ax.set_xlabel(param1_name, fontsize=13, fontweight='bold')
    ax.set_ylabel(param2_name, fontsize=13, fontweight='bold')
    ax.set_title(title, fontsize=14, fontweight='bold')
    
    # Annotate cells with values
    for i in range(len(param2_values)):
        for j in range(len(param1_values)):
            value = results_grid[i, j]
            text = ax.text(j, i, f'{value:.1f}',
                          ha="center", va="center", color="black", fontsize=9,
                          bbox=dict(boxstyle='round', facecolor='white', alpha=0.6))
    
    # Find and highlight best cell
    best_i, best_j = np.unravel_index(results_grid.argmax(), results_grid.shape)
    rect = Rectangle((best_j - 0.5, best_i - 0.5), 1, 1, 
                     fill=False, edgecolor='blue', linewidth=4)
    ax.add_patch(rect)
    
    # Colorbar
    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label(metric_name, fontsize=12, fontweight='bold')
    
    # Add best configuration text
    best_param1 = param1_values[best_j]
    best_param2 = param2_values[best_i]
    best_value = results_grid[best_i, best_j]
    
    textstr = f'Best: {param1_name}={best_param1}, {param2_name}={best_param2}\n{metric_name}={best_value:.2f}'
    props = dict(boxstyle='round', facecolor='lightblue', alpha=0.8)
    ax.text(0.02, 0.98, textstr, transform=ax.transAxes, fontsize=11,
            verticalalignment='top', bbox=props, fontweight='bold')
    
    plt.tight_layout()
    plt.savefig(outpath, dpi=300, bbox_inches='tight')
    plt.close()


def plot_metric_distributions(runs: list[dict], title: str, outpath: Path) -> None:
    """Plot distributions of various performance metrics."""
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    
    metrics = [
        ('final_val_pnl', 'Final Validation PnL'),
        ('best_val_pnl_during_training', 'Best Validation PnL'),
        ('last_10_val_std', 'Stability (Last 10 Std)'),
        ('mean_reward_last_50', 'Mean Reward (Last 50)'),
        ('mean_td_error_last_50', 'Mean TD Error (Last 50)'),
    ]
    
    for idx, (metric_key, metric_label) in enumerate(metrics):
        ax = axes.flatten()[idx]
        
        values = [r["summary"][metric_key] for r in runs if r["summary"][metric_key] is not None]
        labels = [r["label"] for r in runs if r["summary"][metric_key] is not None]
        
        if not values:
            continue
        
        # Box plot with scatter
        bp = ax.boxplot([values], vert=False, widths=0.5, patch_artist=True,
                        boxprops=dict(facecolor='lightblue', alpha=0.7),
                        medianprops=dict(color='red', linewidth=2))
        
        # Add scatter points
        y_positions = np.random.normal(1, 0.04, size=len(values))
        scatter = ax.scatter(values, y_positions, alpha=0.6, s=100, c=range(len(values)), cmap='viridis')
        
        # Annotate extreme values
        if len(values) > 0:
            best_idx = np.argmax(values)
            worst_idx = np.argmin(values)
            ax.annotate(labels[best_idx], (values[best_idx], y_positions[best_idx]),
                       xytext=(5, 5), textcoords='offset points', fontsize=8,
                       bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.7))
            if len(values) > 1:
                ax.annotate(labels[worst_idx], (values[worst_idx], y_positions[worst_idx]),
                           xytext=(5, -15), textcoords='offset points', fontsize=8,
                           bbox=dict(boxstyle='round', facecolor='lightcoral', alpha=0.7))
        
        ax.set_xlabel(metric_label, fontsize=11, fontweight='bold')
        ax.set_yticks([])
        ax.grid(True, alpha=0.3, axis='x')
        
        # Add statistics
        stats_text = f'Mean: {np.mean(values):.2f}\nStd: {np.std(values):.2f}'
        ax.text(0.02, 0.98, stats_text, transform=ax.transAxes,
               fontsize=9, verticalalignment='top',
               bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    # Remove unused subplot
    fig.delaxes(axes.flatten()[5])
    
    plt.suptitle(title, fontsize=16, fontweight='bold')
    plt.tight_layout()
    plt.savefig(outpath, dpi=300, bbox_inches='tight')
    plt.close()


def plot_comparison_table(runs: list[dict], title: str, outpath: Path) -> None:
    """Create a detailed comparison table as an image."""
    fig, ax = plt.subplots(figsize=(16, max(8, len(runs) * 0.5)))
    ax.axis('tight')
    ax.axis('off')
    
    # Prepare data
    headers = ['Config', 'Final PnL', 'Best PnL', 'Best Ep', 'Stability', 'TD Error', 'Hold', 'Buy', 'Sell']
    rows = []
    
    for r in runs:
        s = r["summary"]
        ac = s["action_counts"]
        rows.append([
            r["label"],
            f"{s['final_val_pnl']:.2f}",
            f"{s['best_val_pnl_during_training']:.2f}",
            f"{s['best_val_pnl_episode']}" if s['best_val_pnl_episode'] else "N/A",
            f"{s['last_10_val_std']:.2f}" if s['last_10_val_std'] is not None else "N/A",
            f"{s['mean_td_error_last_50']:.4f}" if s['mean_td_error_last_50'] is not None else "N/A",
            f"{ac['HOLD']}" if 'HOLD' in ac else "0",
            f"{ac['BUY']}" if 'BUY' in ac else "0",
            f"{ac['SELL']}" if 'SELL' in ac else "0",
        ])
    
    # Create table
    table = ax.table(cellText=rows, colLabels=headers, cellLoc='center', loc='center',
                    colWidths=[0.15, 0.1, 0.1, 0.08, 0.1, 0.1, 0.08, 0.08, 0.08])
    
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1, 2)
    
    # Style header
    for i in range(len(headers)):
        cell = table[(0, i)]
        cell.set_facecolor('#4CAF50')
        cell.set_text_props(weight='bold', color='white')
    
    # Highlight best performer in each metric column
    metrics_cols = [1, 2, 4]  # Final PnL, Best PnL, Stability (lower is better)
    for col_idx in metrics_cols:
        values = []
        for row_idx in range(1, len(rows) + 1):
            cell_text = table[(row_idx, col_idx)].get_text().get_text()
            try:
                values.append(float(cell_text))
            except:
                values.append(None)
        
        if any(v is not None for v in values):
            if col_idx == 4:  # Stability (lower is better)
                best_idx = np.nanargmin([v if v is not None else np.inf for v in values])
            else:
                best_idx = np.nanargmax([v if v is not None else -np.inf for v in values])
            
            table[(best_idx + 1, col_idx)].set_facecolor('#90EE90')
    
    # Alternate row colors
    for i in range(1, len(rows) + 1):
        for j in range(len(headers)):
            if i % 2 == 0:
                if table[(i, j)].get_facecolor() == (1.0, 1.0, 1.0, 1.0):  # if not already colored
                    table[(i, j)].set_facecolor('#f0f0f0')
    
    plt.title(title, fontsize=16, fontweight='bold', pad=20)
    plt.savefig(outpath, dpi=300, bbox_inches='tight')
    plt.close()


def generate_recommendation_report(all_runs: dict, outpath: Path) -> dict:
    """Analyze all runs and generate best parameter recommendations."""
    recommendations = {}
    
    # Analyze each sweep type
    for sweep_name, runs in all_runs.items():
        if not runs:
            continue
        
        # Find best by different criteria
        best_final = max(runs, key=lambda r: r["summary"]["final_val_pnl"])
        best_peak = max(runs, key=lambda r: r["summary"]["best_val_pnl_during_training"])
        
        # Find most stable (among top performers)
        top_performers = sorted(runs, key=lambda r: r["summary"]["final_val_pnl"], reverse=True)[:max(3, len(runs)//3)]
        most_stable = min(top_performers, key=lambda r: r["summary"]["last_10_val_std"] if r["summary"]["last_10_val_std"] is not None else float('inf'))
        
        recommendations[sweep_name] = {
            "best_final_pnl": {
                "config": best_final["label"],
                "value": best_final["summary"]["final_val_pnl"],
                "stability": best_final["summary"]["last_10_val_std"],
            },
            "best_peak_pnl": {
                "config": best_peak["label"],
                "value": best_peak["summary"]["best_val_pnl_during_training"],
                "stability": best_peak["summary"]["last_10_val_std"],
            },
            "most_stable_top_performer": {
                "config": most_stable["label"],
                "value": most_stable["summary"]["final_val_pnl"],
                "stability": most_stable["summary"]["last_10_val_std"],
            }
        }
    
    # Generate text report
    report_lines = ["=" * 80]
    report_lines.append("HYPERPARAMETER TUNING RECOMMENDATIONS")
    report_lines.append("=" * 80)
    report_lines.append("")
    
    for sweep_name, rec in recommendations.items():
        report_lines.append(f"\n{sweep_name.upper().replace('_', ' ')}:")
        report_lines.append("-" * 60)
        
        report_lines.append(f"\n  Best Final PnL: {rec['best_final_pnl']['config']}")
        report_lines.append(f"    - Final PnL: {rec['best_final_pnl']['value']:.2f}")
        report_lines.append(f"    - Stability: {rec['best_final_pnl']['stability']:.4f}")
        
        report_lines.append(f"\n  Best Peak PnL: {rec['best_peak_pnl']['config']}")
        report_lines.append(f"    - Peak PnL: {rec['best_peak_pnl']['value']:.2f}")
        report_lines.append(f"    - Stability: {rec['best_peak_pnl']['stability']:.4f}")
        
        report_lines.append(f"\n  Most Stable (Top 33%): {rec['most_stable_top_performer']['config']}")
        report_lines.append(f"    - Final PnL: {rec['most_stable_top_performer']['value']:.2f}")
        report_lines.append(f"    - Stability: {rec['most_stable_top_performer']['stability']:.4f}")
    
    report_lines.append("\n" + "=" * 80)
    report_text = "\n".join(report_lines)
    
    # Save report
    with open(outpath, 'w') as f:
        f.write(report_text)
    
    print("\n" + report_text)
    
    return recommendations


# ============================================================================
# MAIN EXECUTION
# ============================================================================

def main():
    print("=" * 80)
    print("ENHANCED Q-LEARNING HYPERPARAMETER TUNING")
    print("=" * 80)
    
    print("\n[1/6] Loading data...")
    train_features = load_data("train.xlsx")
    val_features = load_data("validate.xlsx")

    # Base configuration
    base = QConfig(
        episodes=250,
        eval_every=20,
        seed=0,
        track_visits=False,
        track_td_error=True,
        reward_shaping_enabled=False,
    )

    # Parameter grids - expanded for better coverage
    gammas = [0.90, 0.95, 0.99, 0.995, 0.999]
    alphas = [0.5, 0.1, 0.05, 0.01, 0.005]
    episode_grid = [50, 100, 150, 200, 250]

    # Create output directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    outdir = RESULTS_DIR / f"tune_enhanced_{timestamp}"
    ensure_dir(outdir)
    print(f"\nSaving outputs to: {outdir}")

    all_runs = {}

    # ========================================================================
    # GAMMA SWEEP
    # ========================================================================
    print("\n[2/6] Running gamma sweep...")
    gamma_runs = []
    for i, g in enumerate(gammas, 1):
        cfg = replace(base, gamma=g)
        print(f"  [{i}/{len(gammas)}] gamma={g:.3f}, alpha={cfg.alpha}, episodes={cfg.episodes}")
        run = run_one(train_features, val_features, cfg)
        gamma_runs.append({"label": f"γ={g:.3f}", **run})

    all_runs['gamma_sweep'] = gamma_runs
    
    print("  Generating gamma sweep visualizations...")
    plot_learning_curves_with_stats(
        gamma_runs,
        f"Gamma Sweep: Learning Curves (α={base.alpha}, episodes={base.episodes})",
        outdir / "gamma_sweep_learning_curves.png"
    )
    plot_convergence_analysis(
        gamma_runs,
        f"Gamma Sweep: Convergence Analysis",
        outdir / "gamma_sweep_convergence.png"
    )
    plot_metric_distributions(
        gamma_runs,
        f"Gamma Sweep: Performance Metrics Distribution",
        outdir / "gamma_sweep_distributions.png"
    )
    plot_comparison_table(
        gamma_runs,
        f"Gamma Sweep: Detailed Comparison",
        outdir / "gamma_sweep_table.png"
    )

    # ========================================================================
    # ALPHA SWEEP
    # ========================================================================
    print("\n[3/6] Running alpha sweep...")
    alpha_runs = []
    for i, a in enumerate(alphas, 1):
        cfg = replace(base, alpha=a)
        print(f"  [{i}/{len(alphas)}] alpha={a:.3f}, gamma={cfg.gamma}, episodes={cfg.episodes}")
        run = run_one(train_features, val_features, cfg)
        alpha_runs.append({"label": f"α={a:.3f}", **run})

    all_runs['alpha_sweep'] = alpha_runs
    
    print("  Generating alpha sweep visualizations...")
    plot_learning_curves_with_stats(
        alpha_runs,
        f"Alpha Sweep: Learning Curves (γ={base.gamma}, episodes={base.episodes})",
        outdir / "alpha_sweep_learning_curves.png"
    )
    plot_convergence_analysis(
        alpha_runs,
        f"Alpha Sweep: Convergence Analysis",
        outdir / "alpha_sweep_convergence.png"
    )
    plot_metric_distributions(
        alpha_runs,
        f"Alpha Sweep: Performance Metrics Distribution",
        outdir / "alpha_sweep_distributions.png"
    )
    plot_comparison_table(
        alpha_runs,
        f"Alpha Sweep: Detailed Comparison",
        outdir / "alpha_sweep_table.png"
    )

    # ========================================================================
    # EPISODES SWEEP
    # ========================================================================
    print("\n[4/6] Running episodes sweep...")
    episode_runs = []
    for i, E in enumerate(episode_grid, 1):
        cfg = replace(base, episodes=E, eval_every=pick_eval_every(E, frac=0.05, min_every=25))
        print(f"  [{i}/{len(episode_grid)}] episodes={E}, eval_every={cfg.eval_every}")
        run = run_one(train_features, val_features, cfg)
        episode_runs.append({"label": f"ep={E}", **run})

    all_runs['episodes_sweep'] = episode_runs
    
    print("  Generating episodes sweep visualizations...")
    plot_learning_curves_with_stats(
        episode_runs,
        f"Episodes Sweep: Learning Curves (α={base.alpha}, γ={base.gamma})",
        outdir / "episodes_sweep_learning_curves.png"
    )
    plot_convergence_analysis(
        episode_runs,
        f"Episodes Sweep: Convergence Analysis",
        outdir / "episodes_sweep_convergence.png"
    )
    plot_metric_distributions(
        episode_runs,
        f"Episodes Sweep: Performance Metrics Distribution",
        outdir / "episodes_sweep_distributions.png"
    )
    plot_comparison_table(
        episode_runs,
        f"Episodes Sweep: Detailed Comparison",
        outdir / "episodes_sweep_table.png"
    )

    # ========================================================================
    # 2D GRID SEARCH: ALPHA vs GAMMA
    # ========================================================================
    print("\n[5/6] Running 2D grid search (alpha vs gamma)...")
    
    # Use smaller grids for 2D search to keep runtime reasonable
    alpha_grid_2d = [0.05, 0.10, 0.15, 0.20, 0.30]
    gamma_grid_2d = [0.93, 0.95, 0.97, 0.99]
    
    grid_results_final = np.zeros((len(gamma_grid_2d), len(alpha_grid_2d)))
    grid_results_best = np.zeros((len(gamma_grid_2d), len(alpha_grid_2d)))
    grid_results_stability = np.zeros((len(gamma_grid_2d), len(alpha_grid_2d)))
    
    grid_runs = []
    total_runs = len(alpha_grid_2d) * len(gamma_grid_2d)
    run_count = 0
    
    for i, g in enumerate(gamma_grid_2d):
        for j, a in enumerate(alpha_grid_2d):
            run_count += 1
            cfg = replace(base, alpha=a, gamma=g)
            print(f"  [{run_count}/{total_runs}] alpha={a:.3f}, gamma={g:.3f}")
            
            run = run_one(train_features, val_features, cfg)
            grid_runs.append({"label": f"α={a:.2f},γ={g:.3f}", **run})
            
            grid_results_final[i, j] = run["summary"]["final_val_pnl"]
            grid_results_best[i, j] = run["summary"]["best_val_pnl_during_training"]
            grid_results_stability[i, j] = run["summary"]["last_10_val_std"] if run["summary"]["last_10_val_std"] is not None else 0
    
    all_runs['grid_2d_alpha_gamma'] = grid_runs
    
    print("  Generating 2D grid visualizations...")
    plot_2d_heatmap(
        "Alpha (α)", alpha_grid_2d,
        "Gamma (γ)", gamma_grid_2d,
        grid_results_final,
        "Final Validation PnL",
        "2D Grid Search: Final Validation PnL",
        outdir / "grid_2d_final_pnl.png"
    )
    plot_2d_heatmap(
        "Alpha (α)", alpha_grid_2d,
        "Gamma (γ)", gamma_grid_2d,
        grid_results_best,
        "Best Validation PnL",
        "2D Grid Search: Best Validation PnL",
        outdir / "grid_2d_best_pnl.png"
    )
    plot_2d_heatmap(
        "Alpha (α)", alpha_grid_2d,
        "Gamma (γ)", gamma_grid_2d,
        grid_results_stability,
        "Stability (Std Dev)",
        "2D Grid Search: Learning Stability",
        outdir / "grid_2d_stability.png"
    )

    # ========================================================================
    # GENERATE RECOMMENDATIONS
    # ========================================================================
    print("\n[6/6] Generating recommendations and summary...")
    recommendations = generate_recommendation_report(
        all_runs,
        outdir / "RECOMMENDATIONS.txt"
    )

    # ========================================================================
    # SAVE COMPREHENSIVE JSON SUMMARY
    # ========================================================================
    def make_json_serializable(obj):
        """Convert numpy types to native Python types."""
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
        "base_config": base.__dict__,
        "parameter_grids": {
            "gammas": gammas,
            "alphas": alphas,
            "episodes": episode_grid,
            "alpha_grid_2d": alpha_grid_2d,
            "gamma_grid_2d": gamma_grid_2d,
        },
        "sweeps": {},
        "recommendations": make_json_serializable(recommendations),
    }
    
    for sweep_name, runs in all_runs.items():
        summary["sweeps"][sweep_name] = [
            {
                "label": r["label"],
                "config": r["config"].__dict__,
                "summary": make_json_serializable(r["summary"]),
            }
            for r in runs
        ]
    
    with open(outdir / "tuning_summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    # ========================================================================
    # FINAL SUMMARY
    # ========================================================================
    print("\n" + "=" * 80)
    print("TUNING COMPLETE!")
    print("=" * 80)
    print(f"\nAll results saved to: {outdir}")
    print("\nGenerated visualizations:")
    for p in sorted(outdir.glob("*.png")):
        print(f"  ✓ {p.name}")
    print(f"\n  ✓ {(outdir / 'tuning_summary.json').name}")
    print(f"  ✓ {(outdir / 'RECOMMENDATIONS.txt').name}")
    print("\n" + "=" * 80)


if __name__ == "__main__":
    main()