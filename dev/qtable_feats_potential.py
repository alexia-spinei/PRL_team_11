"""Q-learning with full feature set for dam energy trading."""

import argparse
import json
import sys
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path

# Add project root to path for imports
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import pandas as pd
from src.environment import DamEnvGym, DamConfig

# add near imports (top of file)
import matplotlib.pyplot as plt


def plot_training_diagnostics(training_metrics: dict, output_dir: Path) -> None:
    """Save training diagnostic plots to output_dir."""
    output_dir.mkdir(parents=True, exist_ok=True)

    rewards = training_metrics.get("episode_rewards", [])
    eps = training_metrics.get("epsilon_history", [])
    td = training_metrics.get("td_error_history", [])
    val_hist = training_metrics.get(
        "val_pnl_history", []
    )  # list of [episode, pnl] or (episode, pnl)

    # 1) Episode reward curve
    if rewards:
        plt.figure()
        plt.plot(rewards)
        plt.xlabel("Episode")
        plt.ylabel("Total episode reward")
        plt.title("Training reward per episode")
        plt.tight_layout()
        plt.savefig(output_dir / "reward_curve.png", dpi=200)
        plt.close()

    # 2) Epsilon decay curve
    if eps:
        plt.figure()
        plt.plot(eps)
        plt.xlabel("Episode")
        plt.ylabel("Epsilon")
        plt.title("Epsilon decay")
        plt.tight_layout()
        plt.savefig(output_dir / "epsilon_decay.png", dpi=200)
        plt.close()

    # 3) TD-error curve (mean abs TD error per episode)
    if td:
        plt.figure()
        plt.plot(td)
        plt.xlabel("Episode")
        plt.ylabel("Mean |TD error|")
        plt.title("TD error per episode")
        plt.tight_layout()
        plt.savefig(output_dir / "td_error_curve.png", dpi=200)
        plt.close()

    # 4) Validation PnL during training (greedy eval)
    if val_hist:
        # accept list of tuples or list of lists
        xs = [int(v[0]) for v in val_hist]
        ys = [float(v[1]) for v in val_hist]
        plt.figure()
        plt.plot(xs, ys, marker="o")
        plt.xlabel("Episode")
        plt.ylabel("Greedy validation PnL")
        plt.title("Validation PnL during training")
        plt.tight_layout()
        plt.savefig(output_dir / "val_pnl_during_training.png", dpi=200)
        plt.close()


def plot_validation_rollout_24h(
    records: dict, output_dir: Path, H: int = 24, start: int = 40
) -> None:
    """Save a 24h window plot from eval rollout records."""
    if not records:
        return

    price = records.get("price", [])
    water = records.get("water", [])
    action = records.get("action", [])
    pnl = records.get("pnl", [])

    if not (price and water and action and pnl):
        return

    end = min(start + H, len(price))
    t = np.arange(end - start)

    price_24 = np.array(price[start:end], dtype=float)
    water_24 = np.array(water[start:end], dtype=float)
    action_24 = np.array(action[start:end], dtype=int)
    pnl_24 = np.array(pnl[start:end], dtype=float)

    fig, axes = plt.subplots(4, 1, figsize=(14, 10), sharex=True)

    axes[0].plot(t, price_24)
    axes[0].set_ylabel("Price")

    axes[1].plot(t, water_24)
    axes[1].set_ylabel("Water (as recorded)")

    axes[2].step(t, action_24, where="mid")
    axes[2].set_yticks([0, 1, 2])
    axes[2].set_yticklabels(["idle", "sell", "buy"])
    axes[2].set_ylabel("Action")

    axes[3].plot(t, pnl_24)
    axes[3].set_ylabel("Cumulative PnL")
    axes[3].set_xlabel("Hour in 24h window")

    fig.suptitle("Q-learning greedy policy (validation) — 24h window")
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    fig.savefig(output_dir / "validation_rollout_24h.png", dpi=200)
    plt.close(fig)


from src.features import precompute_features, create_bins


@dataclass
class QConfig:
    """Configuration for Q-learning agent."""

    # Hyperparameters
    alpha: float = 0.1
    gamma: float = 0.9
    epsilon: float = 1.0
    epsilon_decay: float = 0.995
    epsilon_min: float = 0.01
    episodes: int = 250

    # Discretization
    n_storage: int = 6
    n_price: int = 6
    price_window: int = 168

    # Fixed by design
    n_hour_period: int = 5
    n_weekend: int = 2
    n_seasons: int = 4
    n_actions: int = 3

    # Reward shaping: potential-based (credit/debit based on storage value changes)
    # When buying: gain storage → add positive credit (reduces cost)
    # When selling: lose storage → add negative credit (reduces profit)
    # This flattens the reward landscape and works even with gamma=0
    reward_shaping_enabled: bool = True
    potential_scale_factor: float = 50.0  # multiplier for storage_change × price
    use_current_price: bool = (
        True  # if True, use current price; if False, use normalized average
    )

    # Reproducibility & tracking
    seed: int | None = None
    track_visits: bool = True
    label: str = ""

    # Diagnostics / evaluation during training
    eval_every: int = 20  # run greedy validation every N episodes (0 disables)
    track_td_error: bool = True  # store mean abs TD error per episode

    def q_shape(self) -> tuple:
        """Return shape of Q-table."""
        return (
            self.n_storage,
            self.n_price,
            self.n_hour_period,
            self.n_weekend,
            self.n_seasons,
            self.n_actions,
        )

    @classmethod
    def from_json(cls, path: Path) -> "QConfig":
        """Load config from JSON file."""
        with open(path) as f:
            data = json.load(f)
        return cls(**data)

    def to_json(self, path: Path) -> None:
        """Save config to JSON file."""
        data = asdict(self)
        with open(path, "w") as f:
            json.dump(data, f, indent=2)

    def reward_shaping_description(self) -> str:
        """Return human-readable description of reward shaping config."""
        if not self.reward_shaping_enabled:
            return "disabled"
        price_type = "current" if self.use_current_price else "normalized_avg"
        return (
            f"potential-based: scale={self.potential_scale_factor}, "
            f"price_type={price_type}"
        )


def load_data(filename: str) -> dict:
    """Load price data and precompute features."""
    data_path = PROJECT_ROOT / "data" / filename
    price_df = pd.read_excel(data_path, header=0, parse_dates=[0])
    price_df = price_df.set_index(price_df.columns[0]).rename_axis("date")
    price_df.columns = [
        f"{hour:02d}:00" for hour in range(1, len(price_df.columns) + 1)
    ]
    return precompute_features(price_df)


def create_dam_config(
    features: dict,
    storage_bins=None,
    price_bins=None,
    price_window: int | None = None,
) -> DamConfig:
    """Create DamConfig from precomputed features."""
    return DamConfig(
        prices=features["prices"],
        hour_period=features["hour_period"],
        is_weekend=features["is_weekend"],
        season=features["season"],
        price_window=price_window
        if price_window is not None
        else DamConfig.price_window,
        storage_bins=storage_bins,
        price_bins=price_bins,
    )


def compute_bin_coverage(
    visit_counts: np.ndarray, well_visited_threshold: int = 100
) -> dict:
    """
    Compute coverage statistics for full state space.

    Args:
        visit_counts: array of shape (n_storage, n_price, n_hour_period, n_weekend, n_seasons, n_actions)
        well_visited_threshold: minimum visits to consider a state "well-visited"

    Returns:
        dict with coverage statistics for full states
    """
    # Collapse action dimension to get per-state visit counts.
    state_visits = visit_counts.sum(axis=-1)
    n_states = state_visits.size
    states_visited = int(np.sum(state_visits > 0))
    states_well_visited = int(np.sum(state_visits > well_visited_threshold))

    return {
        "states": {
            "total": n_states,
            "visited": states_visited,
            "well_visited": states_well_visited,
            "pct_visited": 100 * states_visited / n_states,
            "pct_well_visited": 100 * states_well_visited / n_states,
        },
        "threshold": well_visited_threshold,
    }


def print_bin_coverage(coverage: dict) -> None:
    """Print a readable summary of full state coverage."""
    print("\n=== State Coverage ===")
    print(f"  threshold for 'well-visited': {coverage['threshold']} visits\n")

    s = coverage["states"]
    print(
        f"  states  : {s['visited']}/{s['total']} visited ({s['pct_visited']:5.1f}%), "
        f"{s['well_visited']}/{s['total']} well-visited ({s['pct_well_visited']:5.1f}%)"
    )


@dataclass
class TrainingResult:
    """Results from Q-learning training."""

    Q: np.ndarray
    visit_counts: np.ndarray | None
    storage_bins: np.ndarray
    price_bins: np.ndarray
    episode_rewards: list[float]
    epsilon_history: list[float]
    td_error_history: list[float]  # NEW
    val_pnl_history: list[tuple[int, float]]  # NEW: (episode_idx, pnl)


def train_q_learning(
    train_features: dict, config: QConfig, val_features: dict | None = None
) -> TrainingResult:
    """Train Q-learning agent with full feature set."""

    td_error_history: list[float] = []
    val_pnl_history: list[tuple[int, float]] = []

    if config.seed is not None:
        np.random.seed(config.seed)

    storage_bins, price_bins = create_bins(
        train_features["prices"], n_storage=config.n_storage, n_price=config.n_price
    )

    Q = np.zeros(config.q_shape())

    # Initialize visit counts if tracking
    visit_counts = (
        np.zeros(config.q_shape(), dtype=np.int32) if config.track_visits else None
    )

    episode_rewards: list[float] = []
    epsilon_history: list[float] = []

    alpha = config.alpha
    gamma = config.gamma
    epsilon = config.epsilon

    dam_config = create_dam_config(
        train_features, storage_bins, price_bins, price_window=config.price_window
    )
    env = DamEnvGym(dam_config)

    for ep in range(config.episodes):
        obs, _ = env.reset()
        state = env.discretize(obs)

        done = False
        total_episode_reward = 0.0
        td_errors_ep: list[float] = []

        while not done:
            if np.random.rand() < epsilon:
                action = np.random.randint(
                    config.n_actions
                )  # Use numpy RNG for reproducibility
            else:
                action = int(np.argmax(Q[state]))

            if visit_counts is not None:
                visit_counts[state + (action,)] += 1

            obs2, reward, terminated, truncated, _ = env.step(action)

            # Potential-based reward shaping
            if config.reward_shaping_enabled:
                old_storage = obs[0]  # storage before action
                new_storage = obs2[0] if not (terminated or truncated) else old_storage
                storage_change = new_storage - old_storage

                if config.use_current_price:
                    price_factor = obs[1]  # current normalized price
                else:
                    price_factor = 0.5  # normalized average (0-1 range, midpoint)

                # When buying: storage_change > 0 → positive credit (reduces cost)
                # When selling: storage_change < 0 → negative credit (reduces profit)
                potential_reward = (
                    storage_change * price_factor * config.potential_scale_factor
                )
                reward += potential_reward

            total_episode_reward += reward
            done = terminated or truncated  # truncated should not be here

            if not done:
                next_state = env.discretize(obs2)
                td_target = reward + gamma * np.max(Q[next_state])
            else:
                next_state = None
                td_target = reward

            td_error = td_target - Q[state + (action,)]
            Q[state + (action,)] += alpha * td_error

            if config.track_td_error:
                td_errors_ep.append(abs(float(td_error)))

            if not done:
                state = next_state

        # ---- per-episode updates (IMPORTANT: inside for-loop) ----
        epsilon = max(config.epsilon_min, epsilon * config.epsilon_decay)
        episode_rewards.append(total_episode_reward)
        epsilon_history.append(epsilon)

        if config.track_td_error:
            td_error_history.append(
                float(np.mean(td_errors_ep)) if td_errors_ep else 0.0
            )

        if (
            val_features is not None
            and config.eval_every > 0
            and (ep + 1) % config.eval_every == 0
        ):
            val_pnl = greedy_validation_pnl(
                Q,
                val_features,
                storage_bins,
                price_bins,
                price_window=config.price_window,
            )
            val_pnl_history.append((ep + 1, float(val_pnl)))
            print(f"[eval] episode {ep + 1}: greedy validation PnL = {val_pnl:.2f}")

        if (ep + 1) % 50 == 0:
            print(
                f"Episode {ep + 1}/{config.episodes}, Reward: {total_episode_reward:.2f}, Epsilon: {epsilon:.4f}"
            )

    return TrainingResult(
        Q=Q,
        visit_counts=visit_counts,
        storage_bins=storage_bins,
        price_bins=price_bins,
        episode_rewards=episode_rewards,
        epsilon_history=epsilon_history,
        td_error_history=td_error_history,
        val_pnl_history=val_pnl_history,
    )


@dataclass
class EvalResult:
    total_pnl: float
    action_counts: dict[str, int]
    records: dict[str, list] | None = None  # NEW


def evaluate(
    Q: np.ndarray,
    val_features: dict,
    storage_bins,
    price_bins,
    price_window: int,
    record: bool = False,
) -> EvalResult:
    """Evaluate Q-table on validation data."""
    config = create_dam_config(
        val_features, storage_bins, price_bins, price_window=price_window
    )
    env = DamEnvGym(config)

    obs, _ = env.reset()
    state = env.discretize(obs)

    done = False
    total_reward = 0.0
    action_counts = {0: 0, 1: 0, 2: 0}  # idle, sell, buy
    records = (
        {"water": [], "price": [], "action": [], "reward": [], "pnl": []}
        if record
        else None
    )

    while not done:
        # record CURRENT state BEFORE choosing/stepping
        if record and records is not None:
            try:
                records["water"].append(float(obs[0]))
                records["price"].append(float(obs[1]))
            except Exception:
                records["water"].append(np.nan)
                records["price"].append(np.nan)

        action = int(np.argmax(Q[state]))
        action_counts[action] += 1
        if record and records is not None:
            records["action"].append(action)

        obs, reward, terminated, truncated, _ = env.step(action)
        total_reward += reward
        done = terminated or truncated

        if record and records is not None:
            records["reward"].append(float(reward))
            records["pnl"].append(float(total_reward))

        if not done:
            state = env.discretize(obs)

    return EvalResult(
        total_pnl=float(total_reward),
        action_counts={
            "idle": action_counts[0],
            "sell": action_counts[1],
            "buy": action_counts[2],
        },
        records=records,
    )


def greedy_validation_pnl(
    Q: np.ndarray,
    val_features: dict,
    storage_bins,
    price_bins,
    price_window: int,
) -> float:
    """Fast greedy eval on validation (no logging, just total PnL)."""
    cfg = create_dam_config(
        val_features, storage_bins, price_bins, price_window=price_window
    )
    env = DamEnvGym(cfg)
    obs, _ = env.reset()
    state = env.discretize(obs)

    done = False
    total_reward = 0.0
    while not done:
        action = int(np.argmax(Q[state]))
        obs, reward, terminated, truncated, _ = env.step(action)
        total_reward += reward
        done = terminated or truncated
        if not done:
            state = env.discretize(obs)

    return float(total_reward)


def save_results(
    config: QConfig,
    result: TrainingResult,
    eval_result: EvalResult,
    output_dir: Path,
) -> None:
    """Save all results to output directory."""
    output_dir.mkdir(parents=True, exist_ok=True)

    # Save config
    config.to_json(output_dir / "config.json")

    # Save training metrics
    training_metrics = {
        "episode_rewards": result.episode_rewards,
        "epsilon_history": result.epsilon_history,
        "td_error_history": result.td_error_history,
        "val_pnl_history": result.val_pnl_history,
    }

    if eval_result.records is not None:
        with open(output_dir / "eval_records.json", "w") as f:
            json.dump(eval_result.records, f)

    with open(output_dir / "training_metrics.json", "w") as f:
        json.dump(training_metrics, f, indent=2)

    # Save training diagnostic plots
    plot_training_diagnostics(training_metrics, output_dir)

    if eval_result.records is not None:
        plot_validation_rollout_24h(eval_result.records, output_dir, H=24, start=40)

    # Save eval metrics
    eval_metrics = {
        "validation_pnl": eval_result.total_pnl,
        "action_counts": eval_result.action_counts,
    }
    with open(output_dir / "eval_metrics.json", "w") as f:
        json.dump(eval_metrics, f, indent=2)

    # Save reward shaping info (for reproducibility)
    reward_shaping_info = {
        "enabled": config.reward_shaping_enabled,
        "description": config.reward_shaping_description(),
        "parameters": {
            "potential_scale_factor": config.potential_scale_factor,
            "use_current_price": config.use_current_price,
        },
        # Store actual code for full reproducibility
        "code_snippet": """
# Potential-based reward shaping
if config.reward_shaping_enabled:
    old_storage = obs[0]
    new_storage = obs2[0] if not (terminated or truncated) else old_storage
    storage_change = new_storage - old_storage
    price_factor = obs[1] if config.use_current_price else 0.5
    potential_reward = storage_change * price_factor * config.potential_scale_factor
    reward += potential_reward
""".strip(),
    }
    with open(output_dir / "reward_shaping.json", "w") as f:
        json.dump(reward_shaping_info, f, indent=2)

    # Save Q-table
    np.savez_compressed(output_dir / "qtable.npz", Q=result.Q)

    # Save visit counts if tracked
    if result.visit_counts is not None:
        np.savez_compressed(output_dir / "visit_counts.npz", visits=result.visit_counts)

    # Save bins
    np.save(output_dir / "storage_bins.npy", result.storage_bins)
    np.save(output_dir / "price_bins.npy", result.price_bins)

    print(f"Results saved to: {output_dir}")


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Train Q-learning agent for dam energy trading",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # Config file (loaded first, CLI overrides)
    parser.add_argument(
        "--config",
        type=Path,
        help="Path to JSON config file (CLI args override JSON values)",
    )

    # Hyperparameters
    parser.add_argument("--alpha", type=float, help="Learning rate")
    parser.add_argument("--gamma", type=float, help="Discount factor")
    parser.add_argument("--epsilon", type=float, help="Initial exploration rate")
    parser.add_argument(
        "--epsilon-decay", type=float, help="Epsilon decay rate per episode"
    )
    parser.add_argument("--epsilon-min", type=float, help="Minimum epsilon value")
    parser.add_argument("--episodes", type=int, help="Number of training episodes")

    # Discretization
    parser.add_argument("--n-storage", type=int, help="Number of storage bins")
    parser.add_argument("--n-price", type=int, help="Number of price bins")
    parser.add_argument(
        "--price-window", type=int, help="Rolling window length for price normalization"
    )

    # Reward shaping
    parser.add_argument(
        "--reward-shaping",
        action="store_true",
        help="Enable potential-based reward shaping (credits storage value changes)",
    )
    parser.add_argument(
        "--potential-scale",
        type=float,
        help="Scale factor for potential-based reward (default: 10.0)",
    )
    parser.add_argument(
        "--use-avg-price",
        action="store_true",
        help="Use normalized average price (0.5) instead of current price for potential calculation",
    )

    # Reproducibility & tracking
    parser.add_argument("--seed", type=int, help="Random seed for reproducibility")
    parser.add_argument(
        "--no-track-visits", action="store_true", help="Disable visit count tracking"
    )
    parser.add_argument(
        "--label",
        type=str,
        default="",
        help="Label for experiment (used in output directory name)",
    )

    # Output control
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=PROJECT_ROOT / "results" / "qlearning",
        help="Base directory for saving results",
    )
    parser.add_argument("--no-save", action="store_true", help="Don't save results")

    return parser.parse_args()


def build_config(args: argparse.Namespace) -> QConfig:
    """Build QConfig from args, loading JSON first if specified."""
    # Start with default config or load from JSON
    if args.config is not None:
        config = QConfig.from_json(args.config)
    else:
        config = QConfig()

    # Override with CLI arguments (only if explicitly provided)
    if args.alpha is not None:
        config.alpha = args.alpha
    if args.gamma is not None:
        config.gamma = args.gamma
    if args.epsilon is not None:
        config.epsilon = args.epsilon
    if args.epsilon_decay is not None:
        config.epsilon_decay = args.epsilon_decay
    if args.epsilon_min is not None:
        config.epsilon_min = args.epsilon_min
    if args.episodes is not None:
        config.episodes = args.episodes
    if args.n_storage is not None:
        config.n_storage = args.n_storage
    if args.n_price is not None:
        config.n_price = args.n_price
    if args.price_window is not None:
        config.price_window = args.price_window
    if args.seed is not None:
        config.seed = args.seed
    if args.no_track_visits:
        config.track_visits = False
    if args.label:
        config.label = args.label

    # Reward shaping args
    if args.reward_shaping:
        config.reward_shaping_enabled = True
    if args.potential_scale is not None:
        config.potential_scale_factor = args.potential_scale
    if args.use_avg_price:
        config.use_current_price = False

    return config


def main():
    args = parse_args()
    config = build_config(args)

    print("Loading data...")
    train_features = load_data("train.xlsx")
    val_features = load_data("validate.xlsx")

    print(f"Training: {len(train_features['prices'])} hours")
    print(f"Validation: {len(val_features['prices'])} hours")
    print(
        f"\nConfig: alpha={config.alpha}, gamma={config.gamma}, epsilon={config.epsilon}"
    )
    print(f"        episodes={config.episodes}, seed={config.seed}")
    print(f"        Q-table shape: {config.q_shape()}")
    print(f"        reward_shaping: {config.reward_shaping_description()}")

    print("\nTraining Q-learning agent...")
    result = train_q_learning(train_features, config, val_features=val_features)
    if result.visit_counts is not None:
        coverage = compute_bin_coverage(
            result.visit_counts, well_visited_threshold=1000
        )
        print_bin_coverage(coverage)

    print("\nEvaluating on validation set...")
    eval_result = evaluate(
        result.Q,
        val_features,
        result.storage_bins,
        result.price_bins,
        price_window=config.price_window,
        record=True,
    )

    print(f"Validation PnL: {eval_result.total_pnl:.2f}")
    print(f"Action counts: {eval_result.action_counts}")

    # Save results
    if not args.no_save:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        label_suffix = f"_{config.label}" if config.label else ""
        exp_dir = args.output_dir / f"{timestamp}{label_suffix}"
        save_results(config, result, eval_result, exp_dir)


if __name__ == "__main__":
    main()
