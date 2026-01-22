"""Generate individual plots: boxplot and fixed yearly trends.

- Boxplot: Shows full distribution including outliers
- Yearly trends: Fixed x-axis to show only integer years
"""

import pandas as pd
import matplotlib.pyplot as plt
import os

os.makedirs('results', exist_ok=True)

df_long = pd.read_csv('data/train_long_format.csv')
df_long['year'] = pd.to_datetime(df_long['PRICES']).dt.year

# 1: Boxplot - Price Distribution by Hour (side by side: zoomed + full)
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

# Left: Zoomed view (typical range)
df_long.boxplot(column='price', by='hour_num', ax=ax1, flierprops=dict(marker='.', markersize=2))
ax1.set_xlabel('Hour')
ax1.set_ylabel('Price (€)')
ax1.set_title('Typical Range (0-150€)')
ax1.set_ylim(0, 150)

# Right: Full range with outliers
df_long.boxplot(column='price', by='hour_num', ax=ax2, flierprops=dict(marker='.', markersize=2))
ax2.set_xlabel('Hour')
ax2.set_ylabel('Price (€)')
ax2.set_title('Full Range (includes outliers up to €2500)')

plt.suptitle('Price Distribution by Hour', fontsize=14)
plt.tight_layout()
plt.savefig('results/boxplot_price_by_hour.png', dpi=150, bbox_inches='tight')
print("Saved: results/boxplot_price_by_hour.png")
plt.close()

# 2: Seasonal Trends by Year (years as lines, seasons on x-axis)
# Uses same color scheme as "Hourly by Year": blue=2007, red=2008, green=2009
year_colors = {2007: 'blue', 2008: 'red', 2009: 'green'}
season_order = ['Winter', 'Spring', 'Summer', 'Fall']

fig, ax = plt.subplots(figsize=(10, 6))

for year in sorted(df_long['year'].unique()):
    year_data = df_long[df_long['year'] == year].groupby('season')['price'].mean().reindex(season_order)
    ax.plot(season_order, year_data.values, 'o-', label=str(year), color=year_colors[year], linewidth=2, markersize=8)

ax.set_xlabel('Season')
ax.set_ylabel('Price (€)')
ax.set_title('Seasonal Trends by Year')
ax.legend()
plt.tight_layout()
plt.savefig('results/seasonal_by_year.png', dpi=150, bbox_inches='tight')
print("Saved: results/seasonal_by_year.png")
plt.close()

# 3: Seasonal Trends by Year (with season-specific markers)
# Same as above but with distinct marker shapes per season for extra distinction
season_markers = {'Winter': 'o', 'Spring': 's', 'Summer': '^', 'Fall': 'D'}

fig, ax = plt.subplots(figsize=(10, 6))

for year in sorted(df_long['year'].unique()):
    year_data = df_long[df_long['year'] == year].groupby('season')['price'].mean().reindex(season_order)
    # Draw line
    ax.plot(season_order, year_data.values, '-', color=year_colors[year], linewidth=2)
    # Draw markers with season-specific shapes
    for season in season_order:
        ax.scatter(season, year_data[season], marker=season_markers[season],
                  color=year_colors[year], s=100, zorder=2, edgecolors='black', linewidths=0.5)

# Legend for years only (shapes are redundant with x-axis labels)
from matplotlib.lines import Line2D
year_handles = [Line2D([0], [0], marker='o', color=year_colors[y], markerfacecolor=year_colors[y],
                       markersize=8, linestyle='-', linewidth=2, label=str(y)) for y in sorted(df_long['year'].unique())]
ax.legend(handles=year_handles)
ax.set_xlabel('Season')
ax.set_ylabel('Price (€)')
ax.set_title('Seasonal Trends by Year')
plt.tight_layout()
plt.savefig('results/seasonal_by_year_markers.png', dpi=150, bbox_inches='tight')
print("Saved: results/seasonal_by_year_markers.png")
plt.close()
