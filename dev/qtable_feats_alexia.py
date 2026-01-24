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
    episodes: int = 500

    # Discretization
    n_storage: int = 6
    n_price: int = 6

    # Fixed by design
    n_hour_period: int = 5
    n_weekend: int = 2
    n_seasons: int = 4
    n_actions: int = 3

    # Reward shaping: penalize entering peak periods with low storage
    # (encourages agent to have storage ready for high-price selling opportunities)
    reward_shaping_enabled: bool = False
    peak_low_storage_penalty: float = -1.0  # penalty when in peak with low storage
    low_storage_threshold: int = 1  # storage bins <= this get penalty (0-indexed)
    peak_periods: tuple[int, ...] = (2, 3)  # hour_period indices considered peak (2=Midday, 3=EveningPeak)

    # Reproducibility & tracking
    seed: int | None = None
    track_visits: bool = True
    label: str = ""

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
        # Convert list back to tuple for peak_periods
        if "peak_periods" in data and isinstance(data["peak_periods"], list):
            data["peak_periods"] = tuple(data["peak_periods"])
        return cls(**data)

    def to_json(self, path: Path) -> None:
        """Save config to JSON file."""
        data = asdict(self)
        # Convert tuple to list for JSON serialization
        data["peak_periods"] = list(data["peak_periods"])
        with open(path, "w") as f:
            json.dump(data, f, indent=2)

    def reward_shaping_description(self) -> str:
        """Return human-readable description of reward shaping config."""
        if not self.reward_shaping_enabled:
            return "disabled"
        period_names = {0: "Night", 1: "MorningRush", 2: "Midday", 3: "EveningPeak", 4: "LateNight"}
        peak_names = [period_names.get(p, str(p)) for p in self.peak_periods]
        return (
            f"penalty={self.peak_low_storage_penalty} when storage_bin<={self.low_storage_threshold} "
            f"during {peak_names}"
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


def create_dam_config(features: dict, storage_bins=None, price_bins=None) -> DamConfig:
    """Create DamConfig from precomputed features."""
    return DamConfig(
        prices=features["prices"],
        hour_period=features["hour_period"],
        is_weekend=features["is_weekend"],
        season=features["season"],
        storage_bins=storage_bins,
        price_bins=price_bins,
    )


def compute_bin_coverage(visit_counts: np.ndarray, well_visited_threshold: int = 100) -> dict:
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
    print(f"  states  : {s['visited']}/{s['total']} visited ({s['pct_visited']:5.1f}%), "
          f"{s['well_visited']}/{s['total']} well-visited ({s['pct_well_visited']:5.1f}%)")


@dataclass
class TrainingResult:
    """Results from Q-learning training."""

    Q: np.ndarray
    visit_counts: np.ndarray | None
    storage_bins: np.ndarray
    price_bins: np.ndarray
    episode_rewards: list[float]
    epsilon_history: list[float]


def train_q_learning(train_features: dict, config: QConfig) -> TrainingResult:
    """Train Q-learning agent with full feature set."""
    # Set random seed for reproducibility
    if config.seed is not None:
        np.random.seed(config.seed)

    # Create bins from training prices
    storage_bins, price_bins = create_bins(
        train_features["prices"], n_storage=config.n_storage, n_price=config.n_price
    )

    # Initialize Q-table
    Q = np.zeros(config.q_shape())

    # Initialize visit counts if tracking
    visit_counts = np.zeros(config.q_shape(), dtype=np.int32) if config.track_visits else None

    # Track metrics
    episode_rewards = []
    epsilon_history = []

    # Training hyperparameters
    alpha = config.alpha
    gamma = config.gamma
    epsilon = config.epsilon

    # creating the environment
    dam_config = create_dam_config(train_features, storage_bins, price_bins)
    env = DamEnvGym(dam_config)

    for ep in range(config.episodes):
        obs, _ = env.reset()
        state = env.discretize(obs)

        done = False
        total_episode_reward = 0.0

        while not done:
            # Epsilon-greedy action selection
            if np.random.rand() < epsilon:
                action = np.random.randint(config.n_actions)  # Use numpy RNG for reproducibility
            else:
                action = np.argmax(Q[state])

            # Track visits
            if visit_counts is not None:
                visit_counts[state + (action,)] += 1

            obs2, reward, terminated, truncated, _ = env.step(action)

            # Reward shaping: penalize being in peak period with low storage
            if config.reward_shaping_enabled:
                storage_bin, _, hour_period, _, _ = state
                if hour_period in config.peak_periods and storage_bin <= config.low_storage_threshold:
                    reward += config.peak_low_storage_penalty

            total_episode_reward += reward
            done = terminated or truncated # truncated should not be here

            if not done:
                next_state = env.discretize(obs2)
                td_target = reward + gamma * np.max(Q[next_state])
            else:
                next_state = None
                td_target = reward

            Q[state + (action,)] += alpha * (td_target - Q[state + (action,)])

            if not done:
                state = next_state

        epsilon = max(config.epsilon_min, epsilon * config.epsilon_decay)

        # Track metrics
        episode_rewards.append(total_episode_reward)
        epsilon_history.append(epsilon)

        if (ep + 1) % 50 == 0:
            print(f"Episode {ep + 1}/{config.episodes}, Reward: {total_episode_reward:.2f}, Epsilon: {epsilon:.4f}")

    return TrainingResult(
        Q=Q,
        visit_counts=visit_counts,
        storage_bins=storage_bins,
        price_bins=price_bins,
        episode_rewards=episode_rewards,
        epsilon_history=epsilon_history,
    )


@dataclass
class EvalResult:
    """Results from Q-table evaluation."""

    total_pnl: float
    action_counts: dict[str, int]


def evaluate(Q: np.ndarray, val_features: dict, storage_bins, price_bins) -> EvalResult:
    """Evaluate Q-table on validation data."""
    config = create_dam_config(val_features, storage_bins, price_bins)
    env = DamEnvGym(config)
    obs, _ = env.reset()
    state = env.discretize(obs)

    done = False
    total_reward = 0.0
    action_counts = {0: 0, 1: 0, 2: 0}  # idle, sell, buy

    while not done:
        action = np.argmax(Q[state])
        action_counts[action] += 1
        obs, reward, terminated, truncated, _ = env.step(action)
        total_reward += reward
        done = terminated or truncated
        if not done:
            state = env.discretize(obs)

    return EvalResult(
        total_pnl=total_reward,
        action_counts={"idle": action_counts[0], "sell": action_counts[1], "buy": action_counts[2]},
    )


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
    }
    with open(output_dir / "training_metrics.json", "w") as f:
        json.dump(training_metrics, f, indent=2)

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
            "peak_low_storage_penalty": config.peak_low_storage_penalty,
            "low_storage_threshold": config.low_storage_threshold,
            "peak_periods": list(config.peak_periods),
        },
        # Store actual code for full reproducibility
        "code_snippet": """
# Reward shaping: penalize being in peak period with low storage
if config.reward_shaping_enabled:
    storage_bin, _, hour_period, _, _ = state
    if hour_period in config.peak_periods and storage_bin <= config.low_storage_threshold:
        reward += config.peak_low_storage_penalty
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
    parser.add_argument("--epsilon-decay", type=float, help="Epsilon decay rate per episode")
    parser.add_argument("--epsilon-min", type=float, help="Minimum epsilon value")
    parser.add_argument("--episodes", type=int, help="Number of training episodes")

    # Discretization
    parser.add_argument("--n-storage", type=int, help="Number of storage bins")
    parser.add_argument("--n-price", type=int, help="Number of price bins")

    # Reward shaping
    parser.add_argument(
        "--reward-shaping",
        action="store_true",
        help="Enable reward shaping (penalize low storage in peak periods)",
    )
    parser.add_argument(
        "--peak-penalty",
        type=float,
        help="Penalty for being in peak period with low storage (default: -5.0)",
    )
    parser.add_argument(
        "--low-storage-threshold",
        type=int,
        help="Storage bins at or below this value get penalty (0-indexed, default: 1)",
    )
    parser.add_argument(
        "--peak-periods",
        type=str,
        help="Comma-separated hour_period indices considered peak (default: '3' for EveningPeak)",
    )

    # Reproducibility & tracking
    parser.add_argument("--seed", type=int, help="Random seed for reproducibility")
    parser.add_argument("--no-track-visits", action="store_true", help="Disable visit count tracking")
    parser.add_argument("--label", type=str, default="", help="Label for experiment (used in output directory name)")

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
    if args.seed is not None:
        config.seed = args.seed
    if args.no_track_visits:
        config.track_visits = False
    if args.label:
        config.label = args.label

    # Reward shaping args
    if args.reward_shaping:
        config.reward_shaping_enabled = True
    if args.peak_penalty is not None:
        config.peak_low_storage_penalty = args.peak_penalty
    if args.low_storage_threshold is not None:
        config.low_storage_threshold = args.low_storage_threshold
    if args.peak_periods is not None:
        config.peak_periods = tuple(int(p.strip()) for p in args.peak_periods.split(","))

    return config


def main():
    args = parse_args()
    config = build_config(args)

    print("Loading data...")
    train_features = load_data("train.xlsx")
    val_features = load_data("validate.xlsx")

    print(f"Training: {len(train_features['prices'])} hours")
    print(f"Validation: {len(val_features['prices'])} hours")
    print(f"\nConfig: alpha={config.alpha}, gamma={config.gamma}, epsilon={config.epsilon}")
    print(f"        episodes={config.episodes}, seed={config.seed}")
    print(f"        Q-table shape: {config.q_shape()}")
    print(f"        reward_shaping: {config.reward_shaping_description()}")

    print("\nTraining Q-learning agent...")
    result = train_q_learning(train_features, config)
    if result.visit_counts is not None:
        coverage = compute_bin_coverage(result.visit_counts, well_visited_threshold=1000)
        print_bin_coverage(coverage)

    print("\nEvaluating on validation set...")
    eval_result = evaluate(result.Q, val_features, result.storage_bins, result.price_bins)
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
