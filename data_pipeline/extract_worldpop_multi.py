"""
Multi-District WorldPop Population Density Extraction
Extracts mean population density for all 6 Karnataka districts from the
already-downloaded raster (no new download needed).

Run this from D:\research\scripts
"""

import rasterio
from rasterio.windows import from_bounds
import numpy as np
import pandas as pd

DISTRICTS = {
    'Bengaluru Urban': (77.35, 12.75, 77.85, 13.15),
    'Mysuru': (76.45, 12.10, 76.85, 12.50),
    'Ballari': (76.70, 14.95, 77.15, 15.35),
    'Uttara Kannada': (74.10, 14.00, 75.10, 15.00),
    'Kalaburagi': (76.60, 17.15, 77.10, 17.55),
    'Vijayapura': (75.50, 16.65, 76.10, 17.05),
}

results = []

with rasterio.open('../data/raw/ind_pd_2020_1km.tif') as src:
    nodata = src.nodata
    for district, bbox in DISTRICTS.items():
        window = from_bounds(*bbox, transform=src.transform)
        data = src.read(1, window=window)
        if nodata is not None:
            data = data[data != nodata]
        mean_density = np.nanmean(data)
        results.append({'district': district, 'year': 2020, 'mean_density': mean_density})
        print(f"{district}: mean density = {mean_density:.2f} people/km^2")

result_df = pd.DataFrame(results)
result_df.to_csv('../data/raw/multi_district_population.csv', index=False)
print(f"\nSaved: ../data/raw/multi_district_population.csv")