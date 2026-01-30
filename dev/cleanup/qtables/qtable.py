"""Q-learning experiments for dam energy trading."""

import sys
from pathlib import Path

# Add project root to path for imports
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import pandas as pd
from src.environment import DamEnvGym, DamConfig


def load_prices(filename: str) -> np.ndarray:
    """Load price data from Excel file."""
    data_path = PROJECT_ROOT / "data" / filename
    price_list = pd.read_excel(data_path, header=0, parse_dates=[0])
    price_list = price_list.set_index(price_list.columns[0]).rename_axis("date")
    price_list.columns = [f"{hour:02d}:00" for hour in range(1, len(price_list.columns) + 1)]
    return price_list.to_numpy().ravel()


# Discretization parameters
N_STORAGE = 10
N_PRICE = 10
N_ACTIONS = 3


def create_bins(training_prices: np.ndarray):
    """Create discretization bins based on training data."""
    storage_bins = np.linspace(0.0, 1.0, N_STORAGE + 1)
    price_bins = np.quantile(training_prices, np.linspace(0, 1, N_PRICE + 1))
    return storage_bins, price_bins


def discretize(obs, storage_bins, price_bins):
    """Convert continuous observation to discrete state."""
    water_ratio, price = obs
    s_bin = np.digitize(water_ratio, storage_bins[1:-1])
    p_bin = np.digitize(price, price_bins[1:-1])
    return s_bin, p_bin


def train_q_learning(training_prices: np.ndarray, episodes: int = 200):
    """Train Q-learning agent."""
    storage_bins, price_bins = create_bins(training_prices)
    Q = np.zeros((N_STORAGE, N_PRICE, N_ACTIONS))

    alpha = 0.1      # learning rate
    gamma = 0.9      # discount factor
    epsilon = 0.05
    eps_decay = 0.99
    eps_min = 0.01

    for ep in range(episodes):
        env = DamEnvGym(DamConfig(prices=training_prices))
        obs, _ = env.reset()
        state = discretize(obs, storage_bins, price_bins)

        done = False
        while not done:
            # epsilon-greedy action
            if np.random.rand() < epsilon:
                action = env.action_space.sample()
            else:
                action = np.argmax(Q[state])

            obs2, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated

            if not done:
                next_state = discretize(obs2, storage_bins, price_bins)
                td_target = reward + gamma * np.max(Q[next_state])
            else:
                td_target = reward

            Q[state + (action,)] += alpha * (td_target - Q[state + (action,)])

            if not done:
                state = next_state

        epsilon = max(eps_min, epsilon * eps_decay)

        if (ep + 1) % 50 == 0:
            print(f"Episode {ep + 1}/{episodes}")

    return Q, storage_bins, price_bins


def evaluate(Q, prices: np.ndarray, storage_bins, price_bins):
    """Evaluate Q-table on given prices."""
    env = DamEnvGym(DamConfig(prices=prices))
    obs, _ = env.reset()
    state = discretize(obs, storage_bins, price_bins)

    done = False
    total_reward = 0.0

    while not done:
        action = np.argmax(Q[state])
        obs, reward, terminated, truncated, _ = env.step(action)
        total_reward += reward
        done = terminated or truncated
        if not done:
            state = discretize(obs, storage_bins, price_bins)

    return total_reward


def main():
    print("Loading data...")
    training_prices = load_prices("train.xlsx")
    validation_prices = load_prices("validate.xlsx")

    print(f"Training prices: {len(training_prices)} hours")
    print(f"Validation prices: {len(validation_prices)} hours")

    print("\nTraining Q-learning agent...")
    Q, storage_bins, price_bins = train_q_learning(training_prices)

    print("\nEvaluating on validation set...")
    val_pnl = evaluate(Q, validation_prices, storage_bins, price_bins)
    print(f"Validation PnL (Q-learning): {val_pnl:.2f}")


if __name__ == "__main__":
    main()
