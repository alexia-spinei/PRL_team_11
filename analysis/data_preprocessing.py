"""Data Preprocessing Steps

Reshapes the original train.xlsx into long format and calculates hour statistics.
"""

import pandas as pd
import numpy as np
import os

# Load original Excel file
df = pd.read_excel('data/train.xlsx')

print(f"Original shape: {df.shape}")
print(f"\nOriginal columns: {df.columns.tolist()}")
print(f"\nFirst 3 rows:")
print(df.head(3))

# Convert date column
df['PRICES'] = pd.to_datetime(df['PRICES'])

# Get hour columns
hour_cols = [col for col in df.columns if col.startswith('Hour')]
print(f"Hour columns: {len(hour_cols)}")

# Reshape: wide to long format
df_long = df.melt(id_vars=['PRICES'],
                  value_vars=hour_cols,
                  var_name='hour',
                  value_name='price')

# Extract hour number from 'Hour 01' -> 1
df_long['hour_num'] = df_long['hour'].str.extract(r'(\d+)').astype(int)

# Add temporal features
df_long['year'] = df_long['PRICES'].dt.year
df_long['month'] = df_long['PRICES'].dt.month
df_long['day_of_week'] = df_long['PRICES'].dt.day_name()
df_long['date'] = df_long['PRICES'].dt.date

# Add season
def get_season(month):
    if month in [12, 1, 2]:
        return 'Winter'
    elif month in [3, 4, 5]:
        return 'Spring'
    elif month in [6, 7, 8]:
        return 'Summer'
    else:
        return 'Fall'

df_long['season'] = df_long['month'].apply(get_season)

# Add weekend flag
df_long['is_weekend'] = df_long['day_of_week'].isin(['Saturday', 'Sunday'])

# Add hour period classification
def classify_hour_period(hour):
    if hour in [1, 2, 3, 4, 5, 6]:
        return 'Night (1-6)'
    elif hour in [7, 8, 9]:
        return 'Morning Rush (7-9)'
    elif hour in [10, 11, 12, 13, 14, 15, 16]:
        return 'Midday (10-16)'
    elif hour in [17, 18, 19, 20, 21]:
        return 'Evening Peak (17-21)'
    else:
        return 'Late Night (22-24)'

df_long['hour_period'] = df_long['hour_num'].apply(classify_hour_period)

print(f"\nLong format shape: {df_long.shape}")
print(f"Columns: {df_long.columns.tolist()}")
print(f"\nFirst 5 rows:")
print(df_long.head())

# Verify transformation
original_records = len(df)
hours_per_day = len(hour_cols)
expected_long_records = original_records * hours_per_day
print(f"\nVerification:")
print(f"  Original days: {original_records}")
print(f"  Hours per day: {hours_per_day}")
print(f"  Expected long records: {expected_long_records}")
print(f"  Actual long records: {len(df_long)}")
print(f"  Match: {len(df_long) == expected_long_records}")

# Calculate comprehensive statistics per hour
hour_stats = df_long.groupby('hour_num')['price'].agg([
    'mean',
    'std',
    'min',
    'max',
    ('q25', lambda x: x.quantile(0.25)),
    ('q75', lambda x: x.quantile(0.75)),
    ('iqr', lambda x: x.quantile(0.75) - x.quantile(0.25))
]).round(2)

# Add price level classification
hour_stats['price_level'] = pd.cut(hour_stats['mean'], bins=3, labels=['Low', 'Medium', 'High'])

# Add variance level classification
hour_stats['variance_level'] = pd.cut(hour_stats['std'], bins=3, labels=['Stable', 'Moderate', 'Volatile'])

# Combined group
hour_stats['hour_group'] = hour_stats['price_level'].astype(str) + '_' + hour_stats['variance_level'].astype(str)

print("Hour statistics shape:", hour_stats.shape)
print("\nFirst 10 hours:")
print(hour_stats.head(10))

# Verify: each hour should have data from all days
records_per_hour = df_long.groupby('hour_num').size()
print(f"\nRecords per hour (should all be {len(df)}):")
print(records_per_hour.head(10))
print(f"All hours have same count: {records_per_hour.nunique() == 1}")

# Save to CSV
df_long.to_csv('data/train_long_format.csv', index=False)
hour_stats.to_csv('data/hour_statistics.csv')

print("\nFiles saved:")
print("  - data/train_long_format.csv")
print("  - data/hour_statistics.csv")

# Show file sizes for verification
long_size = os.path.getsize('data/train_long_format.csv') / 1024 / 1024
stats_size = os.path.getsize('data/hour_statistics.csv') / 1024
print(f"\nFile sizes:")
print(f"  train_long_format.csv: {long_size:.2f} MB")
print(f"  hour_statistics.csv: {stats_size:.2f} KB")

# Reload to verify
df_long_check = pd.read_csv('data/train_long_format.csv')
hour_stats_check = pd.read_csv('data/hour_statistics.csv', index_col=0)

print("\nVerification after reload:")
print(f"  df_long shape: {df_long_check.shape}")
print(f"  hour_stats shape: {hour_stats_check.shape}")

# Check date range preserved
df_long_check['PRICES'] = pd.to_datetime(df_long_check['PRICES'])
print(f"\nDate range in long format:")
print(f"  Min: {df_long_check['PRICES'].min()}")
print(f"  Max: {df_long_check['PRICES'].max()}")

# Check price statistics preserved
print(f"\nPrice statistics in long format:")
print(f"  Mean: {df_long_check['price'].mean():.2f}")
print(f"  Std: {df_long_check['price'].std():.2f}")
print(f"  Min: {df_long_check['price'].min():.2f}")
print(f"  Max: {df_long_check['price'].max():.2f}")

# Verify against original
print(f"\nOriginal data (all hours flattened):")
original_prices = df[hour_cols].values.flatten()
print(f"  Mean: {original_prices.mean():.2f}")
print(f"  Std: {original_prices.std():.2f}")
print(f"  Min: {original_prices.min():.2f}")
print(f"  Max: {original_prices.max():.2f}")

print(f"\nData integrity check: {'PASSED' if np.allclose(original_prices.mean(), df_long_check['price'].mean()) else 'FAILED'}")
