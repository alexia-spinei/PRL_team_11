"""Electricity Price Pattern Analysis

Explores patterns in electricity price data focusing on:
- Hour-based patterns and variance groupings
- Year-over-year trends
- Seasonal variations
- Day-of-week effects
- Interactions between temporal factors
"""

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import os

# Create results directory if it doesn't exist
os.makedirs('results', exist_ok=True)

# Load preprocessed data
df_long = pd.read_csv('data/train_long_format.csv')
hour_stats = pd.read_csv('data/hour_statistics.csv', index_col=0)

print(f"Data loaded: {len(df_long)} records")
print(f"Date range: {df_long['PRICES'].min()} to {df_long['PRICES'].max()}")

# =============================================================================
# Key Finding 1: Extreme Variance in Evening Hours
# =============================================================================
print("\n" + "="*60)
print("KEY FINDING 1: Extreme Variance in Evening Hours")
print("="*60)

high_var_hours = hour_stats.nlargest(5, 'std')[['mean', 'std', 'max']]
print("Hours with highest price variance:")
print(high_var_hours)
print("\nHour 19 has a max price of €1762.54!")
print("Hour 21 has a max price of €2500.00!")

# =============================================================================
# Key Finding 2: Clear Hour Groupings
# =============================================================================
print("\n" + "="*60)
print("KEY FINDING 2: Clear Hour Groupings")
print("="*60)

print("Hour classification by price and variance:")
print(hour_stats[['mean', 'std', 'price_level', 'variance_level', 'hour_group']])

print("""
Period Definitions:
| Period       | Hours | Characteristics                                    |
|--------------|-------|---------------------------------------------------|
| Night        | 1-6   | Lowest prices (24-39€), stable variance            |
| Morning Rush | 7-9   | Rising prices (37-57€), increasing variance        |
| Midday       | 10-16 | Highest average prices (54-71€), moderate variance |
| Evening Peak | 17-21 | High prices (54-67€), EXTREME variance             |
| Late Night   | 22-24 | Declining prices (45-50€), stable                  |
""")

# =============================================================================
# Key Finding 3: Dramatic Year-Over-Year Differences
# =============================================================================
print("\n" + "="*60)
print("KEY FINDING 3: Dramatic Year-Over-Year Differences")
print("="*60)

df_long['year'] = pd.to_datetime(df_long['PRICES']).dt.year

yearly_summary = df_long.groupby('year')['price'].agg([
    'mean', 'std', 'min', 'max'
]).round(2)
print("Price statistics by year:")
print(yearly_summary)

print("\nYear-over-year price change:")
print(yearly_summary['mean'].pct_change() * 100)

print("""
Interpretation:
2008 average price (€70.61) is:
- 69% higher than 2007 (€41.78)
- 79% higher than 2009 (€39.36)
This corresponds to the 2008 global energy crisis.
""")

# =============================================================================
# Key Finding 4: Seasonal Patterns Interact with Hour Periods
# =============================================================================
print("\n" + "="*60)
print("KEY FINDING 4: Seasonal Patterns Interact with Hour Periods")
print("="*60)

seasonal_hour = df_long.groupby(['season', 'hour_period'])['price'].mean().round(2)
seasonal_hour_pivot = seasonal_hour.unstack()

print("Average price by season and hour period:")
print(seasonal_hour_pivot)

for period in ['Night (1-6)', 'Morning Rush (7-9)', 'Midday (10-16)',
               'Evening Peak (17-21)', 'Late Night (22-24)']:
    if period in seasonal_hour_pivot.columns:
        period_data = seasonal_hour_pivot[period]
        print(f"\n{period}:")
        print(f"  Range: €{period_data.min():.2f} - €{period_data.max():.2f}")
        print(f"  Highest: {period_data.idxmax()} (€{period_data.max():.2f})")
        print(f"  Lowest: {period_data.idxmin()} (€{period_data.min():.2f})")

# =============================================================================
# Key Finding 5: Weekend Effect Varies by Hour
# =============================================================================
print("\n" + "="*60)
print("KEY FINDING 5: Weekend Effect Varies by Hour")
print("="*60)

df_long['is_weekend'] = pd.to_datetime(df_long['PRICES']).dt.day_name().isin(['Saturday', 'Sunday'])

weekend_comparison = df_long.groupby(['is_weekend', 'hour_num'])['price'].mean()
weekday_prices = weekend_comparison[False]
weekend_prices = weekend_comparison[True]
weekend_effect = ((weekend_prices - weekday_prices) / weekday_prices * 100).round(2)

print("Weekend price change vs weekday by hour:")
print(weekend_effect.sort_values())
print(f"\nLargest discount: Hour {weekend_effect.idxmin()} ({weekend_effect.min()}%)")
print(f"Largest premium: Hour {weekend_effect.idxmax()} ({weekend_effect.max()}%)")

# =============================================================================
# Key Finding 6: Day-of-Week Patterns with Tuesday Peak
# =============================================================================
print("\n" + "="*60)
print("KEY FINDING 6: Day-of-Week Patterns with Tuesday Peak")
print("="*60)

day_order = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']

daily_avg = df_long.groupby(pd.to_datetime(df_long['PRICES']).dt.day_name())['price'].agg([
    'mean', 'std', 'min', 'max'
]).reindex(day_order).round(2)

print("Price statistics by day of week:")
print(daily_avg)

weekly_avg = df_long['price'].mean()
daily_avg['rel_to_avg'] = ((daily_avg['mean'] - weekly_avg) / weekly_avg * 100).round(2)
print("\nRelative to weekly average:")
print(daily_avg[['mean', 'rel_to_avg']])

# =============================================================================
# Key Finding 7: Volatility Hotspots
# =============================================================================
print("\n" + "="*60)
print("KEY FINDING 7: Volatility Hotspots")
print("="*60)

variance_analysis = df_long.groupby(['season', 'hour_period', 'is_weekend'])['price'].agg([
    'mean', 'std', 'count'
]).round(2)

variance_analysis['cv'] = (variance_analysis['std'] / variance_analysis['mean'] * 100).round(2)
variance_analysis = variance_analysis.sort_values('cv', ascending=False)

print("Top 10 most volatile period combinations (by coefficient of variation %):")
print(variance_analysis.head(10)[['mean', 'std', 'cv']])

print("""
Critical Insight:
Fall weekday evening peak has a CV of ~129% - the standard deviation
is larger than the mean! This indicates extreme price unpredictability.
""")

# =============================================================================
# Key Finding 8: Year × Season Interaction
# =============================================================================
print("\n" + "="*60)
print("KEY FINDING 8: Year × Season Interaction")
print("="*60)

year_season = df_long.groupby(
    [pd.to_datetime(df_long['PRICES']).dt.year, 'season']
)['price'].mean().round(2)

year_season_pivot = year_season.unstack()
print("Average price by year and season:")
print(year_season_pivot)

print("\nSeasonal price ranking by year:")
for year in year_season_pivot.index:
    season_rank = year_season_pivot.loc[year].sort_values(ascending=False)
    print(f"{year}: {' > '.join(season_rank.index.tolist())}")

# =============================================================================
# VISUALIZATIONS
# =============================================================================
print("\n" + "="*60)
print("GENERATING VISUALIZATIONS...")
print("="*60)

# Comprehensive pattern analysis (3x3 grid - matching original)
fig, axes = plt.subplots(3, 3, figsize=(18, 14))

# 1: Average Price by Hour with Variance (shaded)
ax = axes[0, 0]
hourly_mean = df_long.groupby('hour_num')['price'].mean()
hourly_std = df_long.groupby('hour_num')['price'].std()
ax.plot(hourly_mean.index, hourly_mean.values, 'b-', label='Mean', linewidth=2)
ax.fill_between(hourly_mean.index, hourly_mean - hourly_std, hourly_mean + hourly_std, alpha=0.3, label='± 1 Std Dev')
ax.set_xlabel('Hour')
ax.set_ylabel('Price (€)')
ax.set_title('Average Price by Hour with Variance')
ax.legend()
ax.set_xlim(1, 24)

# 2: Price Distribution by Hour (boxplot)
ax = axes[0, 1]
df_long.boxplot(column='price', by='hour_num', ax=ax, flierprops=dict(marker='.', markersize=2))
ax.set_xlabel('Hour')
ax.set_ylabel('Price (€)')
ax.set_title('Price Distribution by Hour')
plt.suptitle('')

# 3: Hourly Patterns by Year
ax = axes[0, 2]
colors_year = {'2007': 'blue', '2008': 'red', '2009': 'green'}
for year in df_long['year'].unique():
    year_data = df_long[df_long['year'] == year].groupby('hour_num')['price'].mean()
    ax.plot(year_data.index, year_data.values, 'o-', label=str(year), color=colors_year.get(str(year), 'gray'))
ax.set_xlabel('Hour')
ax.set_ylabel('Price (€)')
ax.set_title('Hourly Patterns by Year')
ax.legend()
ax.set_xlim(1, 24)

# 4: Hourly Patterns by Season
ax = axes[1, 0]
season_colors = {'Winter': 'blue', 'Spring': 'green', 'Summer': 'orange', 'Fall': 'red'}
for season in ['Winter', 'Spring', 'Summer', 'Fall']:
    season_data = df_long[df_long['season'] == season].groupby('hour_num')['price'].mean()
    ax.plot(season_data.index, season_data.values, 'o-', label=season, color=season_colors[season])
ax.set_xlabel('Hour')
ax.set_ylabel('Price (€)')
ax.set_title('Hourly Patterns by Season')
ax.legend()
ax.set_xlim(1, 24)

# 5: Weekday vs Weekend Hourly Patterns
ax = axes[1, 1]
weekday_hourly = df_long[~df_long['is_weekend']].groupby('hour_num')['price'].mean()
weekend_hourly = df_long[df_long['is_weekend']].groupby('hour_num')['price'].mean()
ax.plot(weekday_hourly.index, weekday_hourly.values, 'o-', label='Weekday', color='blue')
ax.plot(weekend_hourly.index, weekend_hourly.values, 'o-', label='Weekend', color='orange')
ax.set_xlabel('Hour')
ax.set_ylabel('Price (€)')
ax.set_title('Weekday vs Weekend Hourly Patterns')
ax.legend()
ax.set_xlim(1, 24)

# 6: Price Heatmap: Day vs Hour
ax = axes[1, 2]
day_order = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
df_long['day_of_week'] = pd.to_datetime(df_long['PRICES']).dt.day_name()
day_hour_pivot = df_long.groupby(['day_of_week', 'hour_num'])['price'].mean().unstack()
day_hour_pivot = day_hour_pivot.reindex(day_order)
sns.heatmap(day_hour_pivot, cmap='YlOrRd', ax=ax)
ax.set_title('Price Heatmap: Day vs Hour')
ax.set_xlabel('Hour')
ax.set_ylabel('')

# 7: Price Heatmap: Season vs Hour Period
ax = axes[2, 0]
season_order = ['Winter', 'Spring', 'Summer', 'Fall']
period_order = ['Night (1-6)', 'Morning Rush (7-9)', 'Midday (10-16)', 'Evening Peak (17-21)', 'Late Night (22-24)']
season_period_pivot = df_long.groupby(['season', 'hour_period'])['price'].mean().unstack()
season_period_pivot = season_period_pivot.reindex(season_order)
if all(p in season_period_pivot.columns for p in period_order):
    season_period_pivot = season_period_pivot[period_order]
sns.heatmap(season_period_pivot, annot=True, fmt='.1f', cmap='YlOrRd', ax=ax)
ax.set_title('Price Heatmap: Season vs Hour Period')
ax.set_xlabel('Hour Period')
ax.set_ylabel('')

# 8: Yearly Trends by Season
ax = axes[2, 1]
for season in ['Winter', 'Spring', 'Summer', 'Fall']:
    season_yearly = df_long[df_long['season'] == season].groupby('year')['price'].mean()
    ax.plot(season_yearly.index, season_yearly.values, 'o-', label=season, color=season_colors[season])
ax.set_xlabel('Year')
ax.set_ylabel('Price (€)')
ax.set_title('Yearly Trends by Season')
ax.legend()

# 9: Price Volatility by Hour (CV%)
ax = axes[2, 2]
cv_by_hour = df_long.groupby('hour_num')['price'].agg(lambda x: x.std()/x.mean()*100)
ax.bar(cv_by_hour.index, cv_by_hour.values, color='orange')
ax.set_xlabel('Hour')
ax.set_ylabel('Coefficient of Variation (%)')
ax.set_title('Price Volatility by Hour')
ax.set_xlim(0.5, 24.5)

plt.tight_layout()
plt.savefig('results/comprehensive_pattern_analysis.png', dpi=150, bbox_inches='tight')
print("Saved: results/comprehensive_pattern_analysis.png")
plt.close()

print("\nAll visualizations saved to results/")

# =============================================================================
# INDIVIDUAL PLOTS (for slides)
# =============================================================================
print("Saving individual plots...")

# 1: Hourly price with variance
fig, ax = plt.subplots(figsize=(10, 6))
ax.plot(hourly_mean.index, hourly_mean.values, 'b-', label='Mean', linewidth=2)
ax.fill_between(hourly_mean.index, hourly_mean - hourly_std, hourly_mean + hourly_std, alpha=0.3, label='± 1 Std Dev')
ax.set_xlabel('Hour')
ax.set_ylabel('Price (€)')
ax.set_title('Average Price by Hour with Variance')
ax.legend()
ax.set_xlim(1, 24)
plt.tight_layout()
plt.savefig('results/01_hourly_price_variance.png', dpi=150, bbox_inches='tight')
plt.close()

# 2: Hourly patterns by year
fig, ax = plt.subplots(figsize=(10, 6))
for year in sorted(df_long['year'].unique()):
    year_data = df_long[df_long['year'] == year].groupby('hour_num')['price'].mean()
    ax.plot(year_data.index, year_data.values, 'o-', label=str(year), color=colors_year.get(str(year), 'gray'))
ax.set_xlabel('Hour')
ax.set_ylabel('Price (€)')
ax.set_title('Hourly Patterns by Year')
ax.legend()
ax.set_xlim(1, 24)
plt.tight_layout()
plt.savefig('results/02_hourly_by_year.png', dpi=150, bbox_inches='tight')
plt.close()

# 3: Hourly patterns by season
fig, ax = plt.subplots(figsize=(10, 6))
for season in ['Winter', 'Spring', 'Summer', 'Fall']:
    season_data = df_long[df_long['season'] == season].groupby('hour_num')['price'].mean()
    ax.plot(season_data.index, season_data.values, 'o-', label=season, color=season_colors[season])
ax.set_xlabel('Hour')
ax.set_ylabel('Price (€)')
ax.set_title('Hourly Patterns by Season')
ax.legend()
ax.set_xlim(1, 24)
plt.tight_layout()
plt.savefig('results/03_hourly_by_season.png', dpi=150, bbox_inches='tight')
plt.close()

# 4: Weekday vs weekend
fig, ax = plt.subplots(figsize=(10, 6))
ax.plot(weekday_hourly.index, weekday_hourly.values, 'o-', label='Weekday', color='blue', linewidth=2)
ax.plot(weekend_hourly.index, weekend_hourly.values, 'o-', label='Weekend', color='orange', linewidth=2)
ax.set_xlabel('Hour')
ax.set_ylabel('Price (€)')
ax.set_title('Weekday vs Weekend Hourly Patterns')
ax.legend()
ax.set_xlim(1, 24)
plt.tight_layout()
plt.savefig('results/04_weekday_vs_weekend.png', dpi=150, bbox_inches='tight')
plt.close()

# 5: Day vs Hour heatmap
fig, ax = plt.subplots(figsize=(12, 6))
sns.heatmap(day_hour_pivot, cmap='YlOrRd', ax=ax)
ax.set_title('Price Heatmap: Day vs Hour')
ax.set_xlabel('Hour')
ax.set_ylabel('')
plt.tight_layout()
plt.savefig('results/05_heatmap_day_hour.png', dpi=150, bbox_inches='tight')
plt.close()

# 6: Season vs Hour Period heatmap
fig, ax = plt.subplots(figsize=(10, 6))
sns.heatmap(season_period_pivot, annot=True, fmt='.1f', cmap='YlOrRd', ax=ax)
ax.set_title('Price Heatmap: Season vs Hour Period')
ax.set_xlabel('Hour Period')
ax.set_ylabel('')
plt.tight_layout()
plt.savefig('results/06_heatmap_season_period.png', dpi=150, bbox_inches='tight')
plt.close()

# 7: Yearly trends by season
fig, ax = plt.subplots(figsize=(10, 6))
for season in ['Winter', 'Spring', 'Summer', 'Fall']:
    season_yearly = df_long[df_long['season'] == season].groupby('year')['price'].mean()
    ax.plot(season_yearly.index, season_yearly.values, 'o-', label=season, color=season_colors[season], linewidth=2, markersize=8)
ax.set_xlabel('Year')
ax.set_ylabel('Price (€)')
ax.set_title('Yearly Trends by Season')
ax.legend()
plt.tight_layout()
plt.savefig('results/07_yearly_by_season.png', dpi=150, bbox_inches='tight')
plt.close()

# 8: Volatility by hour
fig, ax = plt.subplots(figsize=(10, 6))
ax.bar(cv_by_hour.index, cv_by_hour.values, color='orange')
ax.set_xlabel('Hour')
ax.set_ylabel('Coefficient of Variation (%)')
ax.set_title('Price Volatility by Hour')
ax.set_xlim(0.5, 24.5)
plt.tight_layout()
plt.savefig('results/08_volatility_by_hour.png', dpi=150, bbox_inches='tight')
plt.close()

print("Individual plots saved:")
print("  results/01_hourly_price_variance.png")
print("  results/02_hourly_by_year.png")
print("  results/03_hourly_by_season.png")
print("  results/04_weekday_vs_weekend.png")
print("  results/05_heatmap_day_hour.png")
print("  results/06_heatmap_season_period.png")
print("  results/07_yearly_by_season.png")
print("  results/08_volatility_by_hour.png")

# =============================================================================
# Summary
# =============================================================================
print("\n" + "="*60)
print("SUMMARY OF ACTIONABLE INSIGHTS FOR HEURISTIC MODEL")
print("="*60)

print("""
Feature Engineering Recommendations:

1. Hour Periods (5 categories) instead of 24 hours
   - Night (1-6), Morning Rush (7-9), Midday (10-16),
   - Evening Peak (17-21), Late Night (22-24)

2. Volatility Flags
   - high_volatility_hour: hours 19, 21
   - moderate_volatility_hour: hours 10-12, 15-18, 20

3. Temporal Features
   - Year (or year regime: crisis/normal)
   - Season (Winter/Spring/Summer/Fall)
   - is_weekend (Boolean)
   - day_of_week (especially flag Tuesday, Sunday)

4. Interaction Terms
   - season × hour_period
   - is_weekend × hour_period
   - year × season

Price Level Expectations:

| Condition                   | Expected Price Range | Volatility |
|-----------------------------|----------------------|------------|
| Summer Night Weekend        | €20-30               | Low        |
| Weekday Midday (any season) | €60-80               | Moderate   |
| Fall Evening Peak Weekday   | €40-120+             | EXTREME    |
| Sunday (any hour)           | €25-45               | Low        |
| Tuesday Midday              | €70-90               | Moderate   |

Risk Areas for Model:
- Fall weekday evenings (hours 17-21): CV > 100%
- 2008 data: structurally different, consider separate handling
- Hours 19 & 21: potential for extreme outliers (>€1000)

Stable Patterns (Easy to Model):
- Night hours (1-6): consistently low, stable
- Weekend effect on midday hours: predictable discount
- Summer overall: lower variance and prices
- Late night (22-24): stable across seasons
""")
