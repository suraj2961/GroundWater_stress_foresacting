"""
Groundwater Forecasting - Data Cleaning & Merge Script
Combines GRACE, ERA5, NDVI/NDBI, and Population Density into one
monthly master dataset for Bengaluru Urban district.

Run this from D:\research (the folder with all your data files).
"""

import pandas as pd
import xarray as xr
import numpy as np

# A common monthly index we'll align everything to
FULL_RANGE = pd.date_range('2002-01-01', '2026-05-01', freq='MS')  # MS = Month Start

print("=== STEP 1: Clean GRACE ===")
grace = pd.read_csv('bengaluru_grace.csv')
grace['date'] = pd.to_datetime(grace['date'], format='%d-%m-%Y %H:%M')
grace = grace[['date', 'GRACE_TWS_anomaly_cm']].sort_values('date')

# Snap each reading to the 1st of its month, then average duplicates in the same month
grace['month'] = grace['date'].dt.to_period('M').dt.to_timestamp()
grace_monthly = grace.groupby('month')['GRACE_TWS_anomaly_cm'].mean()
grace_monthly = grace_monthly.reindex(FULL_RANGE)  # fills missing months with NaN
grace_monthly.index.name = 'date'
print(f"GRACE: {grace_monthly.notna().sum()} / {len(FULL_RANGE)} months have data")


print("\n=== STEP 2: Clean NDVI/NDBI ===")
ndvi = pd.read_csv('bengaluru_ndvi_ndbi.csv')
ndvi['date'] = pd.to_datetime(ndvi['date'])
ndvi = ndvi.set_index('date')[['NDVI', 'NDBI']]
ndvi = ndvi.reindex(FULL_RANGE)
ndvi.index.name = 'date'
# NDVI/NDBI change smoothly month to month -> linear interpolation is reasonable
ndvi_filled = ndvi.interpolate(method='linear', limit_direction='both')
print(f"NDVI: {ndvi['NDVI'].notna().sum()} / {len(FULL_RANGE)} months had real data "
      f"(rest interpolated)")


print("\n=== STEP 3: Extract ERA5 (precip, temp, PET) ===")
def load_era5(path):
    ds = xr.open_dataset(path)
    print(f"  Variables in {path}: {list(ds.data_vars)}")
    return ds

era5_a = load_era5('era5_2002_2025_extracted/data_stream-moda.nc')
era5_b = load_era5('era5_2026_extracted/data_stream-moda.nc')
concat_dim = 'valid_time' if 'valid_time' in era5_a.dims else 'time'
era5 = xr.concat([era5_a, era5_b], dim=concat_dim)

# Bengaluru Urban bounding box (same one used for Earth Engine / WorldPop)
lat_min, lat_max = 12.75, 13.15
lon_min, lon_max = 77.35, 77.85

lat_name = 'latitude' if 'latitude' in era5.coords else 'lat'
lon_name = 'longitude' if 'longitude' in era5.coords else 'lon'
time_name = 'valid_time' if 'valid_time' in era5.coords else 'time'

era5_bengaluru = era5.sel(
    {lat_name: slice(lat_max, lat_min), lon_name: slice(lon_min, lon_max)}
).mean(dim=[lat_name, lon_name])

era5_df = era5_bengaluru.to_dataframe().reset_index()
era5_df = era5_df.rename(columns={time_name: 'date'})
era5_df['date'] = pd.to_datetime(era5_df['date']).dt.to_period('M').dt.to_timestamp()
era5_df = era5_df.groupby('date').mean(numeric_only=True)
era5_df = era5_df.reindex(FULL_RANGE)
era5_df.index.name = 'date'
era5_df = era5_df.drop(columns=['number'], errors='ignore')

era5_df = era5_df.rename(columns={
    'tp': 'precipitation_mm',
    't2m': 'temperature_C',
    'pev': 'potential_evaporation_mm',
})
era5_df['precipitation_mm'] = era5_df['precipitation_mm'] * 1000
era5_df['temperature_C'] = era5_df['temperature_C'] - 273.15
era5_df['potential_evaporation_mm'] = era5_df['potential_evaporation_mm'] * 1000

days_in_month = era5_df.index.days_in_month
era5_df['precipitation_mm'] = era5_df['precipitation_mm'] * days_in_month
era5_df['potential_evaporation_mm'] = era5_df['potential_evaporation_mm'] * days_in_month

print(f"ERA5 columns available: {era5_df.columns.tolist()}")

print("\n=== STEP 4: Population density (yearly -> repeat monthly) ===")
pop = pd.read_csv('bengaluru_population_density.csv')
pop_value = pop['mean_density'].iloc[0]
pop_series = pd.Series(pop_value, index=FULL_RANGE, name='population_density')
pop_series.index.name = 'date'
print(f"Using constant population density: {pop_value} people/km^2 for all months")
print("NOTE: this is a simplification - flag it in your methodology/limitations section")


print("\n=== STEP 5: Merge everything ===")
master = pd.DataFrame(index=FULL_RANGE)
master.index.name = 'date'
master['GRACE_TWS_anomaly_cm'] = grace_monthly
master['NDVI'] = ndvi_filled['NDVI']
master['NDBI'] = ndvi_filled['NDBI']
for col in era5_df.columns:
    master[col] = era5_df[col]
master['population_density'] = pop_series

print("\nMaster dataset shape:", master.shape)
print("\nColumn-wise missing value counts:")
print(master.isna().sum())

print("\nFirst 5 rows:")
print(master.head())
print("\n=== STEP 5b: Fill GRACE gaps using seasonal pattern ===")
# For each missing month, use the average value of that same calendar month
# across all other years (e.g. missing June 2017 -> average of all other Junes)
master['month_num'] = master.index.month
seasonal_avg = master.groupby('month_num')['GRACE_TWS_anomaly_cm'].transform('mean')
master['GRACE_TWS_anomaly_cm'] = master['GRACE_TWS_anomaly_cm'].fillna(seasonal_avg)
master = master.drop(columns=['month_num'])
print(f"GRACE gaps remaining after fill: {master['GRACE_TWS_anomaly_cm'].isna().sum()}")
master.to_csv('master_dataset_raw.csv')
print("\nSaved: master_dataset_raw.csv")
print("(GRACE gaps are still NaN on purpose - decide interpolation strategy before modeling)")