"""
Multi-District ERA5 Extraction
Extracts monthly precipitation, temperature, and PET for all 6 Karnataka
districts from the already-downloaded ERA5 NetCDF files (no new download needed).

Run this from D:\research\scripts
"""

import pandas as pd
import xarray as xr

DISTRICTS = {
    'Bengaluru Urban': (12.95, 77.60),
    'Mysuru': (12.30, 76.65),
    'Ballari': (15.14, 76.92),
    'Uttara Kannada': (14.80, 74.50),
    'Kalaburagi': (17.35, 76.85),
    'Vijayapura': (16.83, 75.71),
}

print("=== Loading ERA5 files ===")
era5_a = xr.open_dataset('era5_2002_2025_data/data_stream-moda.nc')
era5_b = xr.open_dataset('era5_2026_data/data_stream-moda.nc')
concat_dim = 'valid_time' if 'valid_time' in era5_a.dims else 'time'
era5 = xr.concat([era5_a, era5_b], dim=concat_dim)

lat_name = 'latitude' if 'latitude' in era5.coords else 'lat'
lon_name = 'longitude' if 'longitude' in era5.coords else 'lon'
time_name = 'valid_time' if 'valid_time' in era5.coords else 'time'

all_rows = []

for district, (lat, lon) in DISTRICTS.items():
    point = era5.sel({lat_name: lat, lon_name: lon}, method='nearest')
    df = point.to_dataframe().reset_index()
    df = df.rename(columns={time_name: 'date'})
    df['date'] = pd.to_datetime(df['date']).dt.to_period('M').dt.to_timestamp()
    df = df.groupby('date').mean(numeric_only=True).reset_index()
    df['district'] = district
    df = df.drop(columns=['number'], errors='ignore')
    df = df.rename(columns={'tp': 'precipitation_mm', 't2m': 'temperature_C', 'pev': 'potential_evaporation_mm'})
    df['precipitation_mm'] = df['precipitation_mm'] * 1000
    df['temperature_C'] = df['temperature_C'] - 273.15
    df['potential_evaporation_mm'] = df['potential_evaporation_mm'] * 1000
    days_in_month = df['date'].dt.days_in_month
    df['precipitation_mm'] = df['precipitation_mm'] * days_in_month
    df['potential_evaporation_mm'] = df['potential_evaporation_mm'] * days_in_month
    all_rows.append(df[['date', 'district', 'precipitation_mm', 'temperature_C', 'potential_evaporation_mm']])
    print(f"{district}: {len(df)} months extracted")

result = pd.concat(all_rows, ignore_index=True)
result.to_csv('../data/raw/multi_district_era5.csv', index=False)
print(f"\nSaved: ../data/raw/multi_district_era5.csv ({len(result)} total rows)")