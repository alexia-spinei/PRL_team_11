"""Q-learning with full feature set for dam energy trading."""

import sys
from pathlib import Path

# Add project root to path for imports
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import pandas as pd
from src.environment import DamEnvGym, DamConfig
from src.features import precompute_features, create_bins


def load_data(filename: str) -> dict:
    """Load price data and precompute features."""
    data_path = PROJECT_ROOT / "data" / filename
    price_df = pd.read_excel(data_path, header=0, parse_dates=[0])
    price_df = price_df.set_index(price_df.columns[0]).rename_axis("date")
    price_df.columns = [f"{hour:02d}:00" for hour in range(1, len(price_df.columns) + 1)]
    return precompute_features(price_df)


def create_config(features: dict, storage_bins=None, price_bins=None) -> DamConfig:
    """Create DamConfig from precomputed features."""
    return DamConfig(
        prices=features["prices"],
        hour_period=features["hour_period"],
        is_weekend=features["is_weekend"],
        season=features["season"],
        storage_bins=storage_bins,
        price_bins=price_bins,
    )


# Q-table dimensions
N_STORAGE = 8
N_PRICE = 4
N_HOUR_PERIOD = 5
N_WEEKEND = 2
N_SEASONS = 4
N_ACTIONS = 3


def train_q_learning(train_features: dict, episodes: int = 200):
    """Train Q-learning agent with full feature set."""
    # Create bins from training prices
    storage_bins, price_bins = create_bins(
        train_features["prices"], n_storage=N_STORAGE, n_price=N_PRICE
    )

    # Initialize Q-table
    Q = np.zeros((N_STORAGE, N_PRICE, N_HOUR_PERIOD, N_WEEKEND, N_SEASONS, N_ACTIONS))

    # Hyperparameters
    alpha = 0.1
    gamma = 0.9
    epsilon = 0.05
    eps_decay = 0.99
    eps_min = 0.01

    for ep in range(episodes):
        config = create_config(train_features, storage_bins, price_bins)
        env = DamEnvGym(config)
        obs, _ = env.reset()
        state = env.discretize(obs)

        done = False
        while not done:
            # Epsilon-greedy action selection
            if np.random.rand() < epsilon:
                action = env.action_space.sample()
            else:
                action = np.argmax(Q[state])

            obs2, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated

            if not done:
                next_state = env.discretize(obs2)
                td_target = reward + gamma * np.max(Q[next_state])
            else:
                next_state = None
                td_target = reward

            Q[state + (action,)] += alpha * (td_target - Q[state + (action,)])

            if not done:
                state = next_state

        epsilon = max(eps_min, epsilon * eps_decay)

        if (ep + 1) % 50 == 0:
            print(f"Episode {ep + 1}/{episodes}")

    return Q, storage_bins, price_bins


def evaluate(Q, val_features: dict, storage_bins, price_bins):
    """Evaluate Q-table on validation data."""
    config = create_config(val_features, storage_bins, price_bins)
    env = DamEnvGym(config)
    obs, _ = env.reset()
    state = env.discretize(obs)

    done = False
    total_reward = 0.0

    while not done:
        action = np.argmax(Q[state])
        obs, reward, terminated, truncated, _ = env.step(action)
        total_reward += reward
        done = terminated or truncated
        if not done:
            state = env.discretize(obs)

    return total_reward


def main():
    print("Loading data...")
    train_features = load_data("train.xlsx")
    val_features = load_data("validate.xlsx")

    print(f"Training: {len(train_features['prices'])} hours")
    print(f"Validation: {len(val_features['prices'])} hours")

    print("\nTraining Q-learning agent...")
    Q, storage_bins, price_bins = train_q_learning(train_features)

    print("\nEvaluating on validation set...")
    val_pnl = evaluate(Q, val_features, storage_bins, price_bins)
    print(f"Validation PnL (Q-learning): {val_pnl:.2f}")


if __name__ == "__main__":
    main()
