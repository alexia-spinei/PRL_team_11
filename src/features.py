"""Feature preprocessing for dam environment."""

import numpy as np
import pandas as pd


def get_hour_period(hour_of_day: int) -> int:
    hour = hour_of_day + 1  # convert index (0-23) to hour (1-24)
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


def get_season(month: int) -> int:
    """
    Map month (1-12) to season (0-3).
    """
    if month in (12, 1, 2):
        return 0  # Winter
    elif month in (3, 4, 5):
        return 1  # Spring
    elif month in (6, 7, 8):
        return 2  # Summer
    else:
        return 3  # Fall


def precompute_features(price_df: pd.DataFrame) -> dict:
    """
    Precompute all temporal features for the entire dataset.

    Args:
        price_df: DataFrame with DatetimeIndex and 24 hourly price columns

    Returns:
        dict with arrays all of length (n_days * 24):
            - prices: flattened price array
            - hour_period: 0-4 for each hour
            - is_weekend: 0 or 1
            - season: 0-3
    """
    prices = price_df.to_numpy().ravel()
    n_days = len(price_df)
    n_total = n_days * 24

    # Initialize arrays
    hour_period = np.zeros(n_total, dtype=np.int32)
    is_weekend = np.zeros(n_total, dtype=np.int32)
    season = np.zeros(n_total, dtype=np.int32)

    # Fill arrays by iterating through days
    for day_idx, date in enumerate(price_df.index):
        start_idx = day_idx * 24
        end_idx = start_idx + 24

        # Weekend: Sat=5, Sun=6 in pandas dayofweek
        if date.dayofweek >= 5:
            is_weekend[start_idx:end_idx] = 1

        # Season from month
        season[start_idx:end_idx] = get_season(date.month)

        # Hour period for each hour in the day
        for hour_idx in range(24):
            hour_period[start_idx + hour_idx] = get_hour_period(hour_idx)

    return {
        "prices": prices,
        "hour_period": hour_period,
        "is_weekend": is_weekend,
        "season": season,
    }


def create_bins(training_prices: np.ndarray, n_storage: int = 8, n_price: int = 6):
    """
    Create discretization bins.

    Args:
        training_prices: array of all training prices (unused; kept for compatibility)
        n_storage: number of storage bins
        n_price: number of price bins

    Returns:
        storage_bins, price_bins: arrays of bin edges
    """
    storage_bins = np.linspace(0.0, 1.0, n_storage + 1)
    price_bins = np.linspace(0.0, 1.0, n_price + 1)
    return storage_bins, price_bins
