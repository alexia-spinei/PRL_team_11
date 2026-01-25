import numpy as np
import gymnasium as gym
from gymnasium import spaces
from dataclasses import dataclass
from typing import Optional, Dict, Any


@dataclass
class DamConfig:
    """Configuration for the Dam environment.

    Arrays (prices, hour_period, is_weekend, season) must all be the same length.
    """

    prices: np.ndarray
    hour_period: np.ndarray  # 0-4: Night/MorningRush/Midday/EveningPeak/LateNight
    is_weekend: np.ndarray  # 0 or 1
    season: np.ndarray  # 0-3: Winter/Spring/Summer/Fall
    price_window: int = 168  # rolling window length for price normalization
    Wmax: float = 100000.0
    Vmax: float = 18000.0
    W_init: float = 50000.0
    g: float = 9.81
    rho: float = 1000.0
    h: float = 30.0
    eta_turbine: float = 0.9
    eta_pump: float = 0.8

    # Discretization bins (set after creation with training data)
    storage_bins: Optional[np.ndarray] = None
    price_bins: Optional[np.ndarray] = None


class RollingPercentileNormalizer:
    """Online rolling percentile rank normalizer for streaming prices."""
    # example: it will return 0 when teh current price is smaller than all historical prices
    #          and 1 when it is larger than all historical prices
    #          for a window of prices no larger than the window size

    def __init__(self, window: int):
        if window <= 0:
            raise ValueError("window must be positive")
        self.window = int(window)
        self._buffer = np.empty(self.window, dtype=np.float64)
        self._count = 0
        self._pos = 0

    def reset(self) -> None:
        self._count = 0
        self._pos = 0

    def update(self, value: float) -> float:
        """Add value and return its percentile rank in the current window."""
        if self._count < self.window:
            self._buffer[self._count] = value
            self._count += 1
            window = self._buffer[:self._count]
        else:
            self._buffer[self._pos] = value
            self._pos = (self._pos + 1) % self.window
            window = self._buffer

        n = window.size
        # special case when this is the first value we normalize and there is no history yet
        if n == 1:
            return 0.5

        less = np.count_nonzero(window < value)
        equal = np.count_nonzero(window == value)
        if less == 0:
            return 0.0
        if less + equal == n:
            return 1.0

        return (less + 0.5 * (equal - 1)) / (n - 1)


class DamEnvGym(gym.Env):
    """Dam energy trading environment.

    Observation: (storage_ratio, normalized_price, hour_period, is_weekend, season)
    Actions: 0=idle, 1=sell, 2=buy
    """

    metadata = {"render_modes": []}

    def __init__(self, config: DamConfig):
        super().__init__()
        self.cfg = config

        # --- action & observation spaces ---
        self.action_space = spaces.Discrete(3)  # idle, sell, buy

        # Observation: [storage_ratio, normalized_price, hour_period, is_weekend, season]
        self.observation_space = spaces.Box(
            low=np.array([0.0, 0.0, 0, 0, 0], dtype=np.float32),
            high=np.array([1.0, 1.0, 4, 1, 3], dtype=np.float32),
            dtype=np.float32,
        )

        self._price_norm = RollingPercentileNormalizer(self.cfg.price_window)
        self._norm_price = 0.0
        self._norm_t = -1 # no normalization yet
        self.reset()

    def _potential_energy_mwh(self, volume_m3: float) -> float:
        joules = self.cfg.rho * self.cfg.g * self.cfg.h * volume_m3
        return joules / 3.6e9

    def _observation(self):
        """Return current state as tuple for discretization."""
        if self._norm_t != self.t:
            self._norm_price = self._price_norm.update(self.cfg.prices[self.t])
            self._norm_t = self.t
        return (
            self.W_t / self.cfg.Wmax,  # storage ratio (0-1)
            self._norm_price,  # normalized price in [0, 1]
            self.cfg.hour_period[self.t],  # 0-4
            self.cfg.is_weekend[self.t],  # 0-1
            self.cfg.season[self.t],  # 0-3
        )

    def reset(self, *, seed: Optional[int] = None, options: Optional[dict] = None):
        super().reset(seed=seed)
        self.t = 0
        self.W_t = self.cfg.W_init
        self.pnl = 0.0
        self._price_norm.reset()
        self._norm_t = -1
        return self._observation(), {}

    def step(self, action: int):
        price = self.cfg.prices[self.t]

        if action == 1:  # sell
            V = min(self.cfg.Vmax, self.W_t)
        elif action == 2:  # buy
            V = min(self.cfg.Vmax, self.cfg.Wmax - self.W_t)
        else:  # idle
            V = 0.0

        E_pot = self._potential_energy_mwh(V)

        # removed V > 0 condition because it doesn't change outcome
        if action == 1:  # sell
            reward = price * (self.cfg.eta_turbine * E_pot)
            self.W_t -= V
        elif action == 2:  # buy
            reward = -price * (E_pot / self.cfg.eta_pump)
            self.W_t += V
        else:
            reward = 0.0

        self.pnl += reward

        self.t += 1

        terminated = self.t >= len(self.cfg.prices)
        truncated = False

        obs = self._observation() if not terminated else None
        info: Dict[str, Any] = {"pnl": self.pnl, "volume": V}

        return obs, reward, terminated, truncated, info

    def discretize(self, obs) -> tuple:
        """Convert continuous observation to discrete state indices.

        Args:
            obs: (storage_ratio, normalized_price, hour_period, is_weekend, season)

        Returns:
            Tuple of discrete indices for Q-table indexing.
        """
        if self.cfg.storage_bins is None or self.cfg.price_bins is None:
            raise ValueError("Bins not set in config. Call set_bins() first.")

        storage_idx = int(np.clip(
            np.digitize(obs[0], self.cfg.storage_bins[1:-1]),
            0, len(self.cfg.storage_bins) - 2
        ))
        price_idx = int(np.clip(
            np.digitize(obs[1], self.cfg.price_bins[1:-1]),
            0, len(self.cfg.price_bins) - 2
        ))
        # Temporal features are already discrete
        return (storage_idx, price_idx, int(obs[2]), int(obs[3]), int(obs[4]))
