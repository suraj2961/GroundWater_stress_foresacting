"""generate_district_figs.py - One chart per district, 600+ DPI each"""
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import re

DISTRICTS = ['Bengaluru Urban', 'Mysuru', 'Ballari', 'Uttara Kannada', 'Kalaburagi', 'Vijayapura']

historical = pd.read_csv('../data/processed/panel_dataset.csv', parse_dates=['date'])
forecast = pd.read_csv('../data/processed/multi_district_forecast_24month.csv', parse_dates=['date'])

for district in DISTRICTS:
    fig, ax = plt.subplots(figsize=(4.2, 2.6))

    hist = historical[historical['district'] == district].sort_values('date')
    hist_recent = hist[hist['date'] >= '2021-01-01']
    ax.plot(hist_recent['date'], hist_recent['GRACE_TWS_anomaly_cm'],
            color='#34495e', linewidth=1.4, label='Historical')

    fc = forecast[forecast['district'] == district].sort_values('date')
    ax.plot(fc['date'], fc['predicted_GRACE_TWS_anomaly_cm'],
            color='#3498db', linewidth=1.4, linestyle='--', label='Forecast')
    ax.axvspan(fc['date'].min(), fc['date'].max(), alpha=0.08, color='#3498db')
    ax.axhline(0, color='gray', linewidth=0.5, linestyle=':')

    ax.set_title(district, fontsize=11, fontweight='bold')
    ax.set_ylabel('GRACE TWS Anomaly (cm)', fontsize=8.5)
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
    ax.tick_params(axis='x', rotation=45, labelsize=7.5)
    ax.tick_params(axis='y', labelsize=7.5)
    ax.legend(fontsize=7.5, loc='upper right')
    ax.grid(alpha=0.3)

    plt.tight_layout()
    fname = 'fig_' + re.sub(r'[^a-z]', '', district.lower()) + '.png'
    plt.savefig(fname, dpi=600, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"Saved {fname} at 600 DPI")

print("Done - 6 files generated")