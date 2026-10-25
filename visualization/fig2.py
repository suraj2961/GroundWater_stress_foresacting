"""generate_fig2.py - Forecasted stress category distribution, 600+ DPI"""
import pandas as pd
import matplotlib.pyplot as plt

DISTRICTS = ['Bengaluru Urban', 'Mysuru', 'Ballari', 'Uttara Kannada', 'Kalaburagi', 'Vijayapura']
CATEGORY_COLORS = {'Safe': '#2ecc71', 'Semi-Critical': '#f1c40f', 'Critical': '#e67e22', 'Over-Exploited': '#e74c3c'}
category_order = ['Safe', 'Semi-Critical', 'Critical', 'Over-Exploited']

forecast = pd.read_csv('../data/processed/multi_district_forecast_24month.csv', parse_dates=['date'])

summary = forecast.groupby(['district', 'stress_category']).size().unstack(fill_value=0)
summary = summary.reindex(columns=category_order, fill_value=0).reindex(DISTRICTS)

fig, ax = plt.subplots(figsize=(9, 5))
bottom = pd.Series([0] * len(summary), index=summary.index)
for cat in category_order:
    ax.bar(summary.index, summary[cat], bottom=bottom, label=cat, color=CATEGORY_COLORS[cat])
    bottom += summary[cat]

ax.set_ylabel('Number of Forecasted Months (out of 24)', fontsize=10)
ax.set_title('Forecasted Groundwater Stress Category Distribution\n(June 2026 - May 2028)',
              fontsize=13, fontweight='bold')
ax.legend(title='Stress Category', bbox_to_anchor=(1.02, 1), loc='upper left')
plt.xticks(rotation=20, ha='right')
plt.tight_layout()
plt.savefig('fig2_stress_summary.png', dpi=600, bbox_inches='tight', facecolor='white')
print("Saved fig2_stress_summary.png at 600 DPI")