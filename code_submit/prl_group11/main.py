from TestEnv import HydroElectric_Test
import argparse
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt


# Feature Engineering Module (maps TestEnv observations to our Q-table state space)


class RollingPercentileNormalizer:
    """Normalize price to [0,1] using rolling percentile rank."""

    def __init__(self, window: int = 168):
        self.window = window
        self._buffer = np.empty(window, dtype=np.float64)
        self._count = 0
        self._pos = 0

    def update(self, value: float) -> float:
        # Add value to buffer, keeping only the last `window` values
        if self._count < self.window:
            self._buffer[self._count] = value
            self._count += 1
            window = self._buffer[: self._count]
        else:
            # Buffer full: overwrite oldest entry
            self._buffer[self._pos] = value
            self._pos = (self._pos + 1) % self.window
            window = self._buffer

        n = window.size
        if n == 1:
            return 0.5

        # Compute percentile rank: fraction of values below current value
        less = np.count_nonzero(window < value)
        equal = np.count_nonzero(window == value)
        if less == 0:
            return 0.0
        if less + equal == n:
            return 1.0
        # Handle ties by averaging their ranks
        return (less + 0.5 * (equal - 1)) / (n - 1)


# Map hour (1-24) to period category (0-4)
def get_hour_period(hour: int) -> int:
    if hour <= 6:
        return 0  # Night
    elif hour <= 9:
        return 1  # Morning Rush
    elif hour <= 16:
        return 2  # Midday
    elif hour <= 21:
        return 3  # Evening Peak
    else:
        return 4  # Late Night


# Map month (1-12) to season (0-3)
def get_season(month: int) -> int:
    if month in (12, 1, 2):
        return 0  # Winter
    elif month in (3, 4, 5):
        return 1  # Spring
    elif month in (6, 7, 8):
        return 2  # Summer
    else:
        return 3  # Fall


class QLearningAgent:
    """Agent that uses a pre-trained Q-table."""

    def __init__(self, qtable_path: str):
        self.Q = np.load(qtable_path)["Q"]

        # Discretization: 6 bins each for storage and price (evenly spaced 0-1)
        self.storage_bins = np.linspace(0, 1, 7)
        self.price_bins = np.linspace(0, 1, 7)

        self.max_volume = 100000.0
        self.price_normalizer = RollingPercentileNormalizer(window=168)

    def _discretize(
        self,
        storage_ratio: float,
        norm_price: float,
        hour_period: int,
        is_weekend: int,
        season: int,
    ) -> tuple:
        # Discretize continuous features (storage, price) into bin indices
        storage_idx = int(
            np.clip(
                np.digitize(storage_ratio, self.storage_bins[1:-1]),
                0,
                len(self.storage_bins) - 2,
            )
        )
        price_idx = int(
            np.clip(
                np.digitize(norm_price, self.price_bins[1:-1]),
                0,
                len(self.price_bins) - 2,
            )
        )
        return (storage_idx, price_idx, hour_period, is_weekend, season)

    def act(self, observation: np.ndarray) -> float:
        """Select action given TestEnv observation.

        Args:
            observation: [volume, price, hour_of_day, day_of_week, day_of_year, month, year]

        Returns:
            Action in range [-1, 1] for TestEnv
        """
        volume, price, hour, day_of_week, _, month, _ = observation

        # Convert to our feature space
        storage_ratio = volume / self.max_volume
        norm_price = self.price_normalizer.update(price)
        hour_period = get_hour_period(int(hour))
        is_weekend = 1 if day_of_week >= 5 else 0
        season = get_season(int(month))

        # Get discrete state
        state = self._discretize(
            storage_ratio, norm_price, hour_period, is_weekend, season
        )

        # Select best action from Q-table
        discrete_action = int(np.argmax(self.Q[state]))

        # Map to TestEnv action format: 0=idle→0, 1=sell→-1, 2=buy→+1
        if discrete_action == 0:
            return 0.0  # idle
        elif discrete_action == 1:
            return -1.0  # sell (release water)
        else:
            return 1.0  # buy (pump water)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--excel_file", type=str, default="data/validate.xlsx")
    args = parser.parse_args()

    # Initialize environment and agent
    script_dir = Path(__file__).parent
    env = HydroElectric_Test(path_to_test_data=args.excel_file)
    agent = QLearningAgent(qtable_path=str(script_dir / "qtable_best.npz"))

    total_reward = []
    cumulative_reward = []

    observation = env.observation()
    for i in range(730 * 24 - 1):
        action = agent.act(observation)
        next_observation, reward, terminated, truncated, info = env.step(action)
        total_reward.append(reward)
        cumulative_reward.append(sum(total_reward))

        done = terminated or truncated
        observation = next_observation

        if done:
            print("Total reward:", sum(total_reward))
            # Plot the cumulative reward over time
            plt.plot(cumulative_reward)
            plt.xlabel("Time (Hours)")
            plt.show()
            break


if __name__ == "__main__":
    main()
