"""
Multi-District GRACE Extraction
Extracts monthly GRACE TWS anomaly for all 6 Karnataka districts
(one per distinct mascon region) from the raw NASA JPL mascon file.

Run this from D:\research\scripts
"""

import xarray as xr
import pandas as pd

RAW_GRACE_PATH = r'D:\Data\GRCTellus.JPL.200204_202605.GLO.RL06.3M.MSCNv04CRI.nc'

DISTRICTS = {
    'Bengaluru Urban': (12.95, 77.60),
    'Mysuru': (12.30, 76.65),
    'Ballari': (15.14, 76.92),
    'Uttara Kannada': (14.80, 74.50),
    'Kalaburagi': (17.35, 76.85),
    'Vijayapura': (16.83, 75.71),
}

print("=== Loading raw GRACE mascon file ===")
ds = xr.open_dataset(RAW_GRACE_PATH)
print(f"Loaded. Time range: {ds.time.min().values} to {ds.time.max().values}")

all_rows = []

for district, (lat, lon) in DISTRICTS.items():
    sub = ds['lwe_thickness'].sel(lat=lat, lon=lon, method='nearest')
    mascon_id = ds['mascon_ID'].sel(lat=lat, lon=lon, method='nearest')
    df = sub.to_dataframe(name='GRACE_TWS_anomaly_cm').reset_index()
    df['district'] = district
    df['mascon_ID'] = float(mascon_id.values)
    all_rows.append(df[['time', 'district', 'GRACE_TWS_anomaly_cm', 'mascon_ID']])
    print(f"{district}: {len(df)} readings, mascon_ID={float(mascon_id.values):.0f}")

result = pd.concat(all_rows, ignore_index=True)
result = result.rename(columns={'time': 'date'})
result = result.sort_values(['district', 'date'])

result.to_csv('../data/raw/multi_district_grace.csv', index=False)
print(f"\nSaved: ../data/raw/multi_district_grace.csv ({len(result)} total rows)")
print("\nDistinct mascon regions used:", result['mascon_ID'].nunique())