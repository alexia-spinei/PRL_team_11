"""Train Q-learning and save the BEST validation checkpoint (not just final).

This script:
1. Reads config from an existing bestmodel experiment directory
2. Trains with the same parameters, tracking best validation Q-table
3. Saves best Q-table, bins, and generates validation rollout plots
"""

import argparse
import json
import sys
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from src.environment import DamEnvGym, DamConfig
from src.features import precompute_features, create_bins


@dataclass
class QConfig:
    """Configuration for Q-learning agent."""
    alpha: float = 0.1
    gamma: float = 0.9
    epsilon: float = 1.0
    epsilon_decay: float = 0.995
    epsilon_min: float = 0.01
    episodes: int = 500
    n_storage: int = 6
    n_price: int = 6
    price_window: int = 168
    n_hour_period: int = 5
    n_weekend: int = 2
    n_seasons: int = 4
    n_actions: int = 3
    reward_shaping_enabled: bool = False
    potential_scale_factor: float = 10.0
    use_current_price: bool = True
    seed: int | None = None
    track_visits: bool = True
    label: str = ""
    eval_every: int = 20
    track_td_error: bool = True

    def q_shape(self) -> tuple:
        return (
            self.n_storage, self.n_price, self.n_hour_period,
            self.n_weekend, self.n_seasons, self.n_actions,
        )

    @classmethod
    def from_json(cls, path: Path) -> "QConfig":
        with open(path) as f:
            data = json.load(f)
        # Filter to only known fields
        known_fields = {f.name for f in cls.__dataclass_fields__.values()}
        filtered = {k: v for k, v in data.items() if k in known_fields}
        return cls(**filtered)

    def to_json(self, path: Path) -> None:
        with open(path, "w") as f:
            json.dump(asdict(self), f, indent=2)


def load_data(filename: str) -> dict:
    data_path = PROJECT_ROOT / "data" / filename
    price_df = pd.read_excel(data_path, header=0, parse_dates=[0])
    price_df = price_df.set_index(price_df.columns[0]).rename_axis("date")
    price_df.columns = [f"{hour:02d}:00" for hour in range(1, len(price_df.columns) + 1)]
    return precompute_features(price_df)


def create_dam_config(features: dict, storage_bins=None, price_bins=None, price_window=None) -> DamConfig:
    return DamConfig(
        prices=features["prices"],
        hour_period=features["hour_period"],
        is_weekend=features["is_weekend"],
        season=features["season"],
        price_window=price_window if price_window is not None else DamConfig.price_window,
        storage_bins=storage_bins,
        price_bins=price_bins,
    )


def greedy_eval(Q, features, storage_bins, price_bins, price_window, record=False):
    """Run greedy evaluation, optionally recording trajectory."""
    cfg = create_dam_config(features, storage_bins, price_bins, price_window=price_window)
    env = DamEnvGym(cfg)
    obs, _ = env.reset()
    state = env.discretize(obs)

    done = False
    total_reward = 0.0
    action_counts = {0: 0, 1: 0, 2: 0}
    records = {"water": [], "price": [], "action": [], "reward": [], "pnl": []} if record else None

    while not done:
        if record:
            records["water"].append(float(obs[0]))
            records["price"].append(float(obs[1]))

        action = int(np.argmax(Q[state]))
        action_counts[action] += 1

        if record:
            records["action"].append(action)

        obs, reward, terminated, truncated, _ = env.step(action)
        total_reward += reward
        done = terminated or truncated

        if record:
            records["reward"].append(float(reward))
            records["pnl"].append(float(total_reward))

        if not done:
            state = env.discretize(obs)

    return total_reward, action_counts, records


def train_with_best_checkpoint(train_features, val_features, config):
    """Train and track best validation checkpoint."""
    if config.seed is not None:
        np.random.seed(config.seed)

    storage_bins, price_bins = create_bins(
        train_features["prices"], n_storage=config.n_storage, n_price=config.n_price
    )

    Q = np.zeros(config.q_shape())
    best_Q = None
    best_val_pnl = float("-inf")
    best_episode = 0

    alpha = config.alpha
    gamma = config.gamma
    epsilon = config.epsilon

    dam_config = create_dam_config(train_features, storage_bins, price_bins, price_window=config.price_window)
    env = DamEnvGym(dam_config)

    val_pnl_history = []

    for ep in range(config.episodes):
        obs, _ = env.reset()
        state = env.discretize(obs)
        done = False

        while not done:
            if np.random.rand() < epsilon:
                action = np.random.randint(config.n_actions)
            else:
                action = int(np.argmax(Q[state]))

            obs2, reward, terminated, truncated, _ = env.step(action)

            # Potential-based reward shaping
            if config.reward_shaping_enabled:
                old_storage = obs[0]
                new_storage = obs2[0] if not (terminated or truncated) else old_storage
                storage_change = new_storage - old_storage
                price_factor = obs[1] if config.use_current_price else 0.5
                reward += storage_change * price_factor * config.potential_scale_factor

            done = terminated or truncated

            if not done:
                next_state = env.discretize(obs2)
                td_target = reward + gamma * np.max(Q[next_state])
            else:
                td_target = reward

            Q[state + (action,)] += alpha * (td_target - Q[state + (action,)])

            if not done:
                state = next_state

        epsilon = max(config.epsilon_min, epsilon * config.epsilon_decay)

        # Validation checkpoint
        if config.eval_every > 0 and (ep + 1) % config.eval_every == 0:
            val_pnl, _, _ = greedy_eval(Q, val_features, storage_bins, price_bins, config.price_window)
            val_pnl_history.append((ep + 1, val_pnl))
            print(f"[eval] episode {ep + 1}: validation PnL = {val_pnl:.2f}")

            if val_pnl > best_val_pnl:
                best_val_pnl = val_pnl
                best_Q = Q.copy()
                best_episode = ep + 1
                print(f"       ^^^ NEW BEST ^^^")

        if (ep + 1) % 50 == 0:
            print(f"Episode {ep + 1}/{config.episodes}, Epsilon: {epsilon:.4f}")

    # If no validation was run, use final
    if best_Q is None:
        best_Q = Q.copy()
        best_episode = config.episodes
        best_val_pnl, _, _ = greedy_eval(Q, val_features, storage_bins, price_bins, config.price_window)

    return {
        "best_Q": best_Q,
        "final_Q": Q,
        "best_episode": best_episode,
        "best_val_pnl": best_val_pnl,
        "storage_bins": storage_bins,
        "price_bins": price_bins,
        "val_pnl_history": val_pnl_history,
    }


def plot_validation_rollout(records, output_path, title_suffix=""):
    """Plot full validation rollout."""
    if not records:
        return

    n = len(records["price"])
    t = np.arange(n)

    fig, axes = plt.subplots(4, 1, figsize=(16, 10), sharex=True)

    axes[0].plot(t, records["price"])
    axes[0].set_ylabel("Price (normalized)")
    axes[0].set_title(f"Validation Rollout{title_suffix}")

    axes[1].plot(t, records["water"])
    axes[1].set_ylabel("Water Level")

    action_colors = {0: "gray", 1: "red", 2: "green"}
    action_labels = {0: "idle", 1: "sell", 2: "buy"}
    for i, a in enumerate(records["action"]):
        axes[2].axvline(i, color=action_colors[a], alpha=0.3, linewidth=0.5)
    axes[2].step(t, records["action"], where="mid", color="black")
    axes[2].set_yticks([0, 1, 2])
    axes[2].set_yticklabels(["idle", "sell", "buy"])
    axes[2].set_ylabel("Action")

    axes[3].plot(t, records["pnl"])
    axes[3].set_ylabel("Cumulative PnL")
    axes[3].set_xlabel("Hour")
    axes[3].axhline(0, color="gray", linestyle="--", alpha=0.5)

    fig.tight_layout()
    fig.savefig(output_path, dpi=200)
    plt.close(fig)


def find_bestmodel_config(results_dir: Path) -> Path | None:
    """Find the first bestmodel config.json in results directory."""
    for exp_dir in sorted(results_dir.iterdir(), reverse=True):  # newest first
        if not exp_dir.is_dir():
            continue
        config_path = exp_dir / "config.json"
        if not config_path.exists():
            continue
        try:
            config = json.load(open(config_path))
            if "bestmodel" in config.get("label", ""):
                return config_path
        except Exception:
            continue
    return None


def main():
    parser = argparse.ArgumentParser(description="Train with best validation checkpoint saving")
    parser.add_argument("--from-config", type=Path, help="Path to config.json from existing experiment")
    parser.add_argument("--auto-find", action="store_true", help="Auto-find a bestmodel config from results")
    parser.add_argument("--seed", type=int, help="Override seed (for running multiple seeds)")
    parser.add_argument("--output-dir", type=Path, default=PROJECT_ROOT / "results" / "best_checkpoints",
                        help="Output directory for best checkpoints")
    args = parser.parse_args()

    # Find config
    if args.from_config:
        config_path = args.from_config
    elif args.auto_find:
        results_dir = PROJECT_ROOT / "results" / "qlearning"
        config_path = find_bestmodel_config(results_dir)
        if config_path is None:
            print("ERROR: No bestmodel config found in results/qlearning/")
            print("Run run_bestmodel.sh first or specify --from-config")
            sys.exit(1)
        print(f"Auto-found config: {config_path}")
    else:
        print("ERROR: Specify --from-config <path> or use --auto-find")
        sys.exit(1)

    config = QConfig.from_json(config_path)

    # Override seed if specified
    if args.seed is not None:
        config.seed = args.seed

    print(f"Loading data...")
    train_features = load_data("train.xlsx")
    val_features = load_data("validate.xlsx")

    print(f"\nConfig: gamma={config.gamma}, alpha={config.alpha}, episodes={config.episodes}")
    print(f"        seed={config.seed}, reward_shaping={config.reward_shaping_enabled}")
    print(f"        potential_scale={config.potential_scale_factor}")

    print(f"\nTraining with best checkpoint tracking...")
    result = train_with_best_checkpoint(train_features, val_features, config)

    print(f"\n{'='*60}")
    print(f"BEST CHECKPOINT: Episode {result['best_episode']}")
    print(f"Best Validation PnL: {result['best_val_pnl']:,.2f}")
    print(f"{'='*60}")

    # Save outputs
    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    seed_label = f"seed{config.seed}" if config.seed is not None else "noseed"
    prefix = f"bestmodel_{seed_label}"

    # Save best Q-table
    qtable_path = output_dir / f"{prefix}_qtable.npz"
    np.savez_compressed(qtable_path, Q=result["best_Q"])
    print(f"Saved best Q-table: {qtable_path}")

    # Save bins
    np.save(output_dir / f"{prefix}_storage_bins.npy", result["storage_bins"])
    np.save(output_dir / f"{prefix}_price_bins.npy", result["price_bins"])

    # Save config with best episode info
    config_out = asdict(config)
    config_out["best_episode"] = result["best_episode"]
    config_out["best_val_pnl"] = result["best_val_pnl"]
    with open(output_dir / f"{prefix}_config.json", "w") as f:
        json.dump(config_out, f, indent=2)

    # Generate validation rollout plot for best checkpoint
    print(f"\nGenerating validation rollout plot for best checkpoint...")
    _, action_counts, records = greedy_eval(
        result["best_Q"], val_features, result["storage_bins"],
        result["price_bins"], config.price_window, record=True
    )

    plot_path = output_dir / f"{prefix}_validation_rollout.png"
    plot_validation_rollout(
        records, plot_path,
        title_suffix=f" (Best @ Episode {result['best_episode']}, PnL: {result['best_val_pnl']:,.0f})"
    )
    print(f"Saved validation plot: {plot_path}")

    # Summary
    print(f"\nAction counts: idle={action_counts[0]}, sell={action_counts[1]}, buy={action_counts[2]}")


if __name__ == "__main__":
    main()
