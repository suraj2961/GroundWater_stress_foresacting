"""
Multi-District NDVI/NDBI Extraction (Earth Engine)
Extracts monthly NDVI/NDBI for the 5 NEW Karnataka districts
(Bengaluru Urban already done separately).

This is the slow step - expect roughly 15-40 min PER district,
so ~1.5-3 hours total for all 5. Progress prints as it goes.

Run this from D:\research\scripts
"""

import ee
import csv

ee.Initialize(project='groundwater-forecast-504718')

DISTRICTS = {
    'Mysuru': (76.45, 12.10, 76.85, 12.50),
    'Ballari': (76.70, 14.95, 77.15, 15.35),
    'Uttara Kannada': (74.10, 14.00, 75.10, 15.00),
    'Kalaburagi': (76.60, 17.15, 77.10, 17.55),
    'Vijayapura': (75.50, 16.65, 76.10, 17.05),
}


def calc_indices(image, red_band, nir_band, swir_band):
    ndvi = image.normalizedDifference([nir_band, red_band]).rename('NDVI')
    ndbi = image.normalizedDifference([swir_band, nir_band]).rename('NDBI')
    return ndvi.addBands(ndbi)


def monthly_mean(geometry, year, month, satellite):
    start = ee.Date.fromYMD(year, month, 1)
    end = start.advance(1, 'month')

    if satellite == 'L7':
        coll = ee.ImageCollection('LANDSAT/LE07/C02/T1_L2').filterDate(start, end).filterBounds(geometry)
        red, nir, swir = 'SR_B3', 'SR_B4', 'SR_B5'
    else:
        coll = ee.ImageCollection('LANDSAT/LC08/C02/T1_L2').filterDate(start, end).filterBounds(geometry)
        red, nir, swir = 'SR_B4', 'SR_B5', 'SR_B6'

    size = coll.size().getInfo()
    if size == 0:
        return None, None

    composite = coll.median()
    indices = calc_indices(composite, red, nir, swir)

    stats = indices.reduceRegion(
        reducer=ee.Reducer.mean(),
        geometry=geometry,
        scale=100,
        maxPixels=1e9
    ).getInfo()

    return stats.get('NDVI'), stats.get('NDBI')


all_results = []

for district, (west, south, east, north) in DISTRICTS.items():
    print(f"\n=== Starting {district} ===")
    geometry = ee.Geometry.Rectangle([west, south, east, north])

    for year in range(2002, 2027):
        for month in range(1, 13):
            if year == 2026 and month > 5:
                break
            satellite = 'L7' if year < 2013 else 'L8'
            try:
                ndvi, ndbi = monthly_mean(geometry, year, month, satellite)
                print(f"  {district} {year}-{month:02d}: NDVI={ndvi}, NDBI={ndbi}")
                all_results.append([f"{year}-{month:02d}-01", district, ndvi, ndbi, satellite])
            except Exception as e:
                print(f"  {district} {year}-{month:02d}: FAILED - {e}")
                all_results.append([f"{year}-{month:02d}-01", district, None, None, satellite])

    print(f"=== Finished {district} ===")

with open('../data/raw/multi_district_ndvi_ndbi.csv', 'w', newline='') as f:
    writer = csv.writer(f)
    writer.writerow(['date', 'district', 'NDVI', 'NDBI', 'satellite'])
    writer.writerows(all_results)

print("\nAll districts done! Saved to ../data/raw/multi_district_ndvi_ndbi.csv")