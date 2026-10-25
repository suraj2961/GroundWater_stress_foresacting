"""
Groundwater Forecast Visualization
Creates two figures:
1. Historical + forecasted GRACE anomaly, one subplot per district
2. Stress category summary (bar chart) across all districts

Run this from D:\research\scripts. Saves PNGs to ../outputs/
"""

import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import os

os.makedirs('../outputs', exist_ok=True)

DISTRICTS = ['Bengaluru Urban', 'Mysuru', 'Ballari', 'Uttara Kannada',
             'Kalaburagi', 'Vijayapura']

CATEGORY_COLORS = {
    'Safe': '#2ecc71',
    'Semi-Critical': '#f1c40f',
    'Critical': '#e67e22',
    'Over-Exploited': '#e74c3c',
}

print("=== Loading data ===")
historical = pd.read_csv('../data/processed/panel_dataset.csv', parse_dates=['date'])
forecast = pd.read_csv('../data/processed/multi_district_forecast_24month.csv', parse_dates=['date'])

print("\n=== Figure 1: Historical + Forecast per district ===")
fig, axes = plt.subplots(3, 2, figsize=(14, 12), sharex=False)
axes = axes.flatten()

for i, district in enumerate(DISTRICTS):
    ax = axes[i]

    hist = historical[historical['district'] == district].sort_values('date')
    hist_recent = hist[hist['date'] >= '2021-01-01']
    ax.plot(hist_recent['date'], hist_recent['GRACE_TWS_anomaly_cm'],
            color='#34495e', linewidth=1.5, label='Historical')

    fc = forecast[forecast['district'] == district].sort_values('date')
    ax.plot(fc['date'], fc['predicted_GRACE_TWS_anomaly_cm'],
            color='#3498db', linewidth=1.5, linestyle='--', label='Forecast')

    ax.axvspan(fc['date'].min(), fc['date'].max(), alpha=0.08, color='#3498db')

    ax.axhline(0, color='gray', linewidth=0.5, linestyle=':')
    ax.set_title(district, fontsize=11, fontweight='bold')
    ax.set_ylabel('GRACE TWS\nAnomaly (cm)', fontsize=9)
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
    ax.tick_params(axis='x', rotation=45, labelsize=8)
    ax.legend(fontsize=8, loc='upper right')
    ax.grid(alpha=0.3)

plt.suptitle('Groundwater Storage: Historical (2021-2026) and 24-Month Forecast\n'
             'Karnataka Districts', fontsize=14, fontweight='bold', y=1.00)
plt.tight_layout()
plt.savefig('../outputs/forecast_by_district.png', dpi=150, bbox_inches='tight')
print("Saved: ../outputs/forecast_by_district.png")
plt.close()


print("\n=== Figure 2: Stress category summary ===")
category_order = ['Safe', 'Semi-Critical', 'Critical', 'Over-Exploited']
summary = forecast.groupby(['district', 'stress_category']).size().unstack(fill_value=0)
summary = summary.reindex(columns=category_order, fill_value=0)
summary = summary.reindex(DISTRICTS)

fig, ax = plt.subplots(figsize=(12, 6))
bottom = pd.Series([0] * len(summary), index=summary.index)

for cat in category_order:
    ax.bar(summary.index, summary[cat], bottom=bottom,
           label=cat, color=CATEGORY_COLORS[cat])
    bottom += summary[cat]

ax.set_ylabel('Number of Forecasted Months (out of 24)', fontsize=10)
ax.set_title('Forecasted Groundwater Stress Category Distribution\n'
              '(June 2026 - May 2028)', fontsize=13, fontweight='bold')
ax.legend(title='Stress Category', bbox_to_anchor=(1.02, 1), loc='upper left')
plt.xticks(rotation=20, ha='right')
plt.tight_layout()
plt.savefig('../outputs/stress_category_summary.png', dpi=150, bbox_inches='tight')
print("Saved: ../outputs/stress_category_summary.png")
plt.close()

print("\nAll visualizations complete. Check the ../outputs/ folder.")
