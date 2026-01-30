"""Visualize Q-table heatmaps and training curves."""

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import ListedColormap

# Slice dimension names
SEASONS = ["winter", "spring", "summer", "fall"]
PERIODS = ["night", "morning", "midday", "evening", "late"]
DAY_TYPES = ["weekday", "weekend"]

# Action labels
ACTION_NAMES = {0: "idle", 1: "sell", 2: "buy"}


def load_experiment(exp_dir: Path) -> dict:
    """Load all data from an experiment directory."""
    data = {}

    # Load Q-table (prefer best, fall back to final or legacy)
    qtable_path = exp_dir / "qtable_best.npz"
    if not qtable_path.exists():
        qtable_path = exp_dir / "qtable_final.npz"
    if not qtable_path.exists():
        qtable_path = exp_dir / "qtable.npz"
    if qtable_path.exists():
        data["Q"] = np.load(qtable_path)["Q"]

    # Load visit counts
    visits_path = exp_dir / "visit_counts.npz"
    if visits_path.exists():
        data["visits"] = np.load(visits_path)["visits"]

    # Load config
    config_path = exp_dir / "config.json"
    if config_path.exists():
        with open(config_path) as f:
            data["config"] = json.load(f)

    # Load training metrics
    metrics_path = exp_dir / "training_metrics.json"
    if metrics_path.exists():
        with open(metrics_path) as f:
            data["training_metrics"] = json.load(f)

    # Load eval metrics
    eval_path = exp_dir / "eval_metrics.json"
    if eval_path.exists():
        with open(eval_path) as f:
            data["eval_metrics"] = json.load(f)

    # Load bins
    storage_bins_path = exp_dir / "storage_bins.npy"
    if storage_bins_path.exists():
        data["storage_bins"] = np.load(storage_bins_path)

    price_bins_path = exp_dir / "price_bins.npy"
    if price_bins_path.exists():
        data["price_bins"] = np.load(price_bins_path)

    return data


def get_slice_indices(season: str, period: str, day_type: str) -> tuple[int, int, int]:
    """Convert slice names to indices."""
    season_idx = SEASONS.index(season)
    period_idx = PERIODS.index(period)
    day_idx = DAY_TYPES.index(day_type)
    return period_idx, day_idx, season_idx


def plot_policy_heatmap(
    Q: np.ndarray,
    slice_indices: tuple[int, int, int],
    storage_bins: np.ndarray | None = None,
    price_bins: np.ndarray | None = None,
    ax: plt.Axes | None = None,
    title: str | None = None,
) -> plt.Axes:
    """Plot optimal policy as heatmap (storage vs price)."""
    period_idx, day_idx, season_idx = slice_indices

    # Extract 2D slice: Q[storage, price, period, day_type, season, action]
    q_slice = Q[:, :, period_idx, day_idx, season_idx, :]

    # Get optimal action for each (storage, price) cell
    policy = np.argmax(q_slice, axis=-1)

    if ax is None:
        fig, ax = plt.subplots(figsize=(8, 6))

    # Custom colormap: gray=idle, green=sell, red=buy
    cmap = ListedColormap(["gray", "green", "red"])

    im = ax.imshow(policy.T, origin="lower", cmap=cmap, vmin=0, vmax=2, aspect="auto")

    # Labels - place ticks at cell boundaries (edges), not centers
    n_storage, n_price = policy.shape

    if storage_bins is not None:
        storage_labels = [f"{b:.2f}" for b in storage_bins]
        ax.set_xticks([i - 0.5 for i in range(n_storage + 1)])
        ax.set_xticklabels(storage_labels, rotation=45, ha="right")
    if price_bins is not None:
        price_labels = [f"{b:.2f}" for b in price_bins]
        ax.set_yticks([i - 0.5 for i in range(n_price + 1)])
        ax.set_yticklabels(price_labels)

    ax.set_xlabel("Storage Level")
    ax.set_ylabel("Price (normalized)")
    ax.set_xlim(-0.5, n_storage - 0.5)
    ax.set_ylim(-0.5, n_price - 0.5)

    if title:
        ax.set_title(title)

    # Colorbar with action labels
    cbar = plt.colorbar(im, ax=ax, ticks=[0, 1, 2])
    cbar.ax.set_yticklabels(["idle", "sell", "buy"])

    return ax


def plot_visit_heatmap(
    visits: np.ndarray,
    slice_indices: tuple[int, int, int],
    storage_bins: np.ndarray | None = None,
    price_bins: np.ndarray | None = None,
    ax: plt.Axes | None = None,
    title: str | None = None,
) -> plt.Axes:
    """Plot visit counts as heatmap (storage vs price, summed over actions)."""
    period_idx, day_idx, season_idx = slice_indices

    # Extract 2D slice and sum over actions
    visit_slice = visits[:, :, period_idx, day_idx, season_idx, :].sum(axis=-1)

    if ax is None:
        fig, ax = plt.subplots(figsize=(8, 6))

    im = ax.imshow(visit_slice.T, origin="lower", cmap="viridis", aspect="auto")

    # Labels - place ticks at cell boundaries (edges), not centers
    n_storage, n_price = visit_slice.shape

    if storage_bins is not None:
        storage_labels = [f"{b:.2f}" for b in storage_bins]
        ax.set_xticks([i - 0.5 for i in range(n_storage + 1)])
        ax.set_xticklabels(storage_labels, rotation=45, ha="right")
    if price_bins is not None:
        price_labels = [f"{b:.2f}" for b in price_bins]
        ax.set_yticks([i - 0.5 for i in range(n_price + 1)])
        ax.set_yticklabels(price_labels)

    ax.set_xlabel("Storage Level")
    ax.set_ylabel("Price (normalized)")
    ax.set_xlim(-0.5, n_storage - 0.5)
    ax.set_ylim(-0.5, n_price - 0.5)

    if title:
        ax.set_title(title)

    plt.colorbar(im, ax=ax, label="Visit Count")

    return ax


def plot_qvalue_heatmaps(
    Q: np.ndarray,
    slice_indices: tuple[int, int, int],
    storage_bins: np.ndarray | None = None,
    price_bins: np.ndarray | None = None,
    title_prefix: str = "",
) -> plt.Figure:
    """Plot Q-values as 3 heatmaps (one per action)."""
    period_idx, day_idx, season_idx = slice_indices

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    for action in range(3):
        q_slice = Q[:, :, period_idx, day_idx, season_idx, action]

        im = axes[action].imshow(q_slice.T, origin="lower", cmap="RdYlGn", aspect="auto")

        if storage_bins is not None:
            storage_labels = [f"{b:.1f}" for b in storage_bins[:-1]]
            axes[action].set_xticks(range(len(storage_labels)))
            axes[action].set_xticklabels(storage_labels, rotation=45, ha="right")
        if price_bins is not None:
            price_labels = [f"{b:.0f}" for b in price_bins[:-1]]
            axes[action].set_yticks(range(len(price_labels)))
            axes[action].set_yticklabels(price_labels)

        axes[action].set_xlabel("Storage Level")
        axes[action].set_ylabel("Price")
        axes[action].set_title(f"{title_prefix}Q-values: {ACTION_NAMES[action]}")

        plt.colorbar(im, ax=axes[action])

    plt.tight_layout()
    return fig


def plot_training_curves(
    episode_rewards: list[float],
    epsilon_history: list[float],
    title: str = "Training Curves",
) -> plt.Figure:
    """Plot episode rewards and epsilon decay."""
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8), sharex=True)

    episodes = range(1, len(episode_rewards) + 1)

    # Episode rewards
    ax1.plot(episodes, episode_rewards, alpha=0.7, label="Episode Reward")
    # Moving average
    window = min(20, len(episode_rewards) // 5) or 1
    if len(episode_rewards) >= window:
        moving_avg = np.convolve(episode_rewards, np.ones(window) / window, mode="valid")
        ax1.plot(range(window, len(episode_rewards) + 1), moving_avg, "r-", linewidth=2, label=f"MA({window})")
    ax1.set_ylabel("Episode Reward")
    ax1.legend()
    ax1.set_title(title)
    ax1.grid(True, alpha=0.3)

    # Epsilon decay
    ax2.plot(episodes, epsilon_history, "g-", linewidth=2)
    ax2.set_xlabel("Episode")
    ax2.set_ylabel("Epsilon")
    ax2.set_title("Epsilon Decay")
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    return fig


def plot_qvalue_single_heatmap(
    Q: np.ndarray,
    slice_indices: tuple[int, int, int],
    storage_bins: np.ndarray | None = None,
    price_bins: np.ndarray | None = None,
    ax: plt.Axes | None = None,
    title: str | None = None,
) -> plt.Axes:
    """Plot max Q-value as heatmap (storage vs price)."""
    period_idx, day_idx, season_idx = slice_indices

    # Extract 2D slice and take max over actions
    q_slice = Q[:, :, period_idx, day_idx, season_idx, :]
    q_max = np.max(q_slice, axis=-1)

    if ax is None:
        fig, ax = plt.subplots(figsize=(8, 6))

    im = ax.imshow(q_max.T, origin="lower", cmap="viridis", aspect="auto")

    # Labels - place ticks at cell boundaries (edges), not centers
    n_storage, n_price = q_max.shape

    if storage_bins is not None:
        storage_labels = [f"{b:.2f}" for b in storage_bins]
        ax.set_xticks([i - 0.5 for i in range(n_storage + 1)])
        ax.set_xticklabels(storage_labels, rotation=45, ha="right")
    if price_bins is not None:
        price_labels = [f"{b:.2f}" for b in price_bins]
        ax.set_yticks([i - 0.5 for i in range(n_price + 1)])
        ax.set_yticklabels(price_labels)

    ax.set_xlabel("Storage Level")
    ax.set_ylabel("Price (normalized)")
    ax.set_xlim(-0.5, n_storage - 0.5)
    ax.set_ylim(-0.5, n_price - 0.5)

    if title:
        ax.set_title(title)

    plt.colorbar(im, ax=ax, label="Max Q-value")

    return ax


def generate_all_slices(
    Q: np.ndarray,
    visits: np.ndarray | None,
    storage_bins: np.ndarray | None,
    price_bins: np.ndarray | None,
    output_dir: Path,
) -> None:
    """Generate policy, visit, and Q-value heatmaps for all 40 slices in subdirectories."""
    # Create subdirectories
    policies_dir = output_dir / "policies"
    visits_dir = output_dir / "visits"
    qvalues_dir = output_dir / "qvalues"

    policies_dir.mkdir(parents=True, exist_ok=True)
    if visits is not None:
        visits_dir.mkdir(parents=True, exist_ok=True)
    qvalues_dir.mkdir(parents=True, exist_ok=True)

    for season in SEASONS:
        for period in PERIODS:
            for day_type in DAY_TYPES:
                slice_indices = get_slice_indices(season, period, day_type)
                slice_name = f"{season}_{period}_{day_type}"

                # Policy heatmap
                fig, ax = plt.subplots(figsize=(8, 6))
                plot_policy_heatmap(
                    Q, slice_indices, storage_bins, price_bins, ax=ax,
                    title=f"Policy: {season}, {period}, {day_type}"
                )
                fig.savefig(policies_dir / f"{slice_name}.png", dpi=150, bbox_inches="tight")
                plt.close(fig)

                # Visit heatmap (if available)
                if visits is not None:
                    fig, ax = plt.subplots(figsize=(8, 6))
                    plot_visit_heatmap(
                        visits, slice_indices, storage_bins, price_bins, ax=ax,
                        title=f"Visits: {season}, {period}, {day_type}"
                    )
                    fig.savefig(visits_dir / f"{slice_name}.png", dpi=150, bbox_inches="tight")
                    plt.close(fig)

                # Q-value heatmap
                fig, ax = plt.subplots(figsize=(8, 6))
                plot_qvalue_single_heatmap(
                    Q, slice_indices, storage_bins, price_bins, ax=ax,
                    title=f"Q-values: {season}, {period}, {day_type}"
                )
                fig.savefig(qvalues_dir / f"{slice_name}.png", dpi=150, bbox_inches="tight")
                plt.close(fig)

    print(f"Generated 40 slice plots in:")
    print(f"  - {policies_dir}")
    if visits is not None:
        print(f"  - {visits_dir}")
    print(f"  - {qvalues_dir}")


def main():
    parser = argparse.ArgumentParser(
        description="Visualize Q-table from Q-learning experiment",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("exp_dir", type=Path, help="Path to experiment directory")
    parser.add_argument(
        "--slice",
        nargs=3,
        metavar=("SEASON", "PERIOD", "DAY_TYPE"),
        default=["summer", "midday", "weekday"],
        help="Slice to visualize: season period day_type",
    )
    parser.add_argument(
        "--all-slices",
        action="store_true",
        help="Generate all 40 slices",
    )
    parser.add_argument(
        "--output",
        type=Path,
        help="Output directory for plots (default: exp_dir/plots)",
    )
    parser.add_argument(
        "--show",
        action="store_true",
        help="Show plots interactively instead of saving",
    )

    args = parser.parse_args()

    # Load experiment data
    print(f"Loading experiment from {args.exp_dir}")
    data = load_experiment(args.exp_dir)

    if "Q" not in data:
        print("Error: No Q-table found in experiment directory")
        return

    Q = data["Q"]
    visits = data.get("visits")
    storage_bins = data.get("storage_bins")
    price_bins = data.get("price_bins")
    training_metrics = data.get("training_metrics", {})

    output_dir = args.output or args.exp_dir / "plots"

    # Generate all slices if requested
    if args.all_slices:
        generate_all_slices(Q, visits, storage_bins, price_bins, output_dir)
        return

    # Parse slice specification
    try:
        season, period, day_type = args.slice
        slice_indices = get_slice_indices(season, period, day_type)
    except ValueError as e:
        print(f"Error: Invalid slice specification: {e}")
        print(f"  Seasons: {SEASONS}")
        print(f"  Periods: {PERIODS}")
        print(f"  Day types: {DAY_TYPES}")
        return

    slice_name = f"{season}, {period}, {day_type}"

    if args.show:
        # Interactive display
        fig, axes = plt.subplots(1, 2, figsize=(14, 6))

        plot_policy_heatmap(
            Q, slice_indices, storage_bins, price_bins, ax=axes[0],
            title=f"Policy: {slice_name}"
        )

        if visits is not None:
            plot_visit_heatmap(
                visits, slice_indices, storage_bins, price_bins, ax=axes[1],
                title=f"Visit Counts: {slice_name}"
            )
        else:
            axes[1].text(0.5, 0.5, "No visit counts available", ha="center", va="center")
            axes[1].set_title("Visit Counts")

        plt.tight_layout()

        # Q-value heatmaps
        plot_qvalue_heatmaps(Q, slice_indices, storage_bins, price_bins, title_prefix=f"{slice_name}: ")

        # Training curves
        if training_metrics:
            plot_training_curves(
                training_metrics.get("episode_rewards", []),
                training_metrics.get("epsilon_history", []),
            )

        plt.show()
    else:
        # Save to files
        output_dir.mkdir(parents=True, exist_ok=True)

        # Policy + visits side by side
        fig, axes = plt.subplots(1, 2, figsize=(14, 6))
        plot_policy_heatmap(
            Q, slice_indices, storage_bins, price_bins, ax=axes[0],
            title=f"Policy: {slice_name}"
        )
        if visits is not None:
            plot_visit_heatmap(
                visits, slice_indices, storage_bins, price_bins, ax=axes[1],
                title=f"Visit Counts: {slice_name}"
            )
        else:
            axes[1].text(0.5, 0.5, "No visit counts available", ha="center", va="center")
        plt.tight_layout()
        fig.savefig(output_dir / "policy_visits.png", dpi=150, bbox_inches="tight")
        plt.close(fig)

        # Q-value heatmaps
        fig = plot_qvalue_heatmaps(Q, slice_indices, storage_bins, price_bins)
        fig.savefig(output_dir / "qvalues.png", dpi=150, bbox_inches="tight")
        plt.close(fig)

        # Training curves
        if training_metrics:
            fig = plot_training_curves(
                training_metrics.get("episode_rewards", []),
                training_metrics.get("epsilon_history", []),
            )
            fig.savefig(output_dir / "training_curves.png", dpi=150, bbox_inches="tight")
            plt.close(fig)

        print(f"Plots saved to {output_dir}")


if __name__ == "__main__":
    main()
