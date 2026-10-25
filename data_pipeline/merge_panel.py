"""
Multi-District Panel Merge
Combines GRACE, ERA5, NDVI/NDBI, and Population data for all 6 Karnataka
districts into one panel dataset (district x date rows).

Run this from D:\research\scripts after all 4 multi-district extraction
scripts have completed successfully.
"""

import pandas as pd
import numpy as np

FULL_RANGE = pd.date_range('2002-01-01', '2026-05-01', freq='MS')
DISTRICTS = ['Bengaluru Urban', 'Mysuru', 'Ballari', 'Uttara Kannada',
             'Kalaburagi', 'Vijayapura']

print("=== STEP 1: Clean GRACE (multi-district) ===")
grace = pd.read_csv('../data/raw/multi_district_grace.csv')
grace['date'] = pd.to_datetime(grace['date'])
grace['month'] = grace['date'].dt.to_period('M').dt.to_timestamp()

grace_panel = {}
for district in DISTRICTS:
    sub = grace[grace['district'] == district]
    monthly = sub.groupby('month')['GRACE_TWS_anomaly_cm'].mean()
    monthly = monthly.reindex(FULL_RANGE)
    grace_panel[district] = monthly
    print(f"{district}: {monthly.notna().sum()} / {len(FULL_RANGE)} months have data")


print("\n=== STEP 2: Clean NDVI/NDBI (combine Bengaluru file + multi-district file) ===")
ndvi_bengaluru = pd.read_csv('../data/raw/bengaluru_ndvi_ndbi.csv')
ndvi_bengaluru['district'] = 'Bengaluru Urban'
ndvi_others = pd.read_csv('../data/raw/multi_district_ndvi_ndbi.csv')
ndvi_all = pd.concat([ndvi_bengaluru, ndvi_others], ignore_index=True)
ndvi_all['date'] = pd.to_datetime(ndvi_all['date'])

ndvi_panel = {}
for district in DISTRICTS:
    sub = ndvi_all[ndvi_all['district'] == district].set_index('date')[['NDVI', 'NDBI']]
    sub = sub.reindex(FULL_RANGE)
    sub_filled = sub.interpolate(method='linear', limit_direction='both')
    ndvi_panel[district] = sub_filled
    print(f"{district}: {sub['NDVI'].notna().sum()} / {len(FULL_RANGE)} months had real NDVI data")


print("\n=== STEP 3: Clean ERA5 (multi-district) ===")
era5 = pd.read_csv('../data/raw/multi_district_era5.csv')
era5['date'] = pd.to_datetime(era5['date'])

era5_panel = {}
for district in DISTRICTS:
    sub = era5[era5['district'] == district].set_index('date')
    sub = sub.reindex(FULL_RANGE)[['precipitation_mm', 'temperature_C', 'potential_evaporation_mm']]
    era5_panel[district] = sub
    print(f"{district}: {sub['temperature_C'].notna().sum()} / {len(FULL_RANGE)} months")


print("\n=== STEP 4: Population density (multi-district, yearly -> repeat monthly) ===")
pop = pd.read_csv('../data/raw/multi_district_population.csv')
pop_panel = {}
for district in DISTRICTS:
    val = pop[pop['district'] == district]['mean_density'].iloc[0]
    pop_panel[district] = pd.Series(val, index=FULL_RANGE)
    print(f"{district}: {val:.2f} people/km^2")


print("\n=== STEP 5: Build panel dataframe (district x date rows) ===")
rows = []
for district in DISTRICTS:
    df = pd.DataFrame(index=FULL_RANGE)
    df['district'] = district
    df['GRACE_TWS_anomaly_cm'] = grace_panel[district]
    df['NDVI'] = ndvi_panel[district]['NDVI']
    df['NDBI'] = ndvi_panel[district]['NDBI']
    df['precipitation_mm'] = era5_panel[district]['precipitation_mm']
    df['temperature_C'] = era5_panel[district]['temperature_C']
    df['potential_evaporation_mm'] = era5_panel[district]['potential_evaporation_mm']
    df['population_density'] = pop_panel[district]
    rows.append(df)

panel = pd.concat(rows)
panel.index.name = 'date'
panel = panel.reset_index()

print(f"\nPanel shape: {panel.shape} ({len(DISTRICTS)} districts x {len(FULL_RANGE)} months)")
print("\nMissing values per column:")
print(panel.isna().sum())


print("\n=== STEP 6: Fill GRACE gaps using seasonal pattern (PER DISTRICT) ===")
panel['month_num'] = panel['date'].dt.month
panel['seasonal_avg'] = panel.groupby(['district', 'month_num'])['GRACE_TWS_anomaly_cm'].transform('mean')
panel['GRACE_TWS_anomaly_cm'] = panel['GRACE_TWS_anomaly_cm'].fillna(panel['seasonal_avg'])
panel = panel.drop(columns=['month_num', 'seasonal_avg'])
print(f"GRACE gaps remaining after fill: {panel['GRACE_TWS_anomaly_cm'].isna().sum()}")

panel.to_csv('../data/processed/panel_dataset.csv', index=False)
print(f"\nSaved: ../data/processed/panel_dataset.csv ({len(panel)} total rows)")
print("Multi-district panel dataset COMPLETE, ready for modeling")