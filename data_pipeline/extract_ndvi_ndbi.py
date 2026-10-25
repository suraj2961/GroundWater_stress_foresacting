import ee
import csv

ee.Initialize(project='groundwater-forecast-504718')

# Bengaluru Urban district boundary (approx bounding box)
bengaluru = ee.Geometry.Rectangle([77.35, 12.75, 77.85, 13.15])

def get_landsat_collection(start, end):
    # Combine Landsat 7 (2002-2013 gap-filled) and Landsat 8/9 (2013-present)
    l7 = ee.ImageCollection('LANDSAT/LE07/C02/T1_L2') \
        .filterDate(start, end) \
        .filterBounds(bengaluru)
    l8 = ee.ImageCollection('LANDSAT/LC08/C02/T1_L2') \
        .filterDate(start, end) \
        .filterBounds(bengaluru)
    return l7, l8

def calc_indices(image, red_band, nir_band, swir_band):
    ndvi = image.normalizedDifference([nir_band, red_band]).rename('NDVI')
    ndbi = image.normalizedDifference([swir_band, nir_band]).rename('NDBI')
    return ndvi.addBands(ndbi)

def monthly_mean(year, month, satellite):
    start = ee.Date.fromYMD(year, month, 1)
    end = start.advance(1, 'month')

    if satellite == 'L7':
        coll = ee.ImageCollection('LANDSAT/LE07/C02/T1_L2').filterDate(start, end).filterBounds(bengaluru)
        red, nir, swir = 'SR_B3', 'SR_B4', 'SR_B5'
    else:
        coll = ee.ImageCollection('LANDSAT/LC08/C02/T1_L2').filterDate(start, end).filterBounds(bengaluru)
        red, nir, swir = 'SR_B4', 'SR_B5', 'SR_B6'

    size = coll.size().getInfo()
    if size == 0:
        return None, None

    composite = coll.median()
    indices = calc_indices(composite, red, nir, swir)

    stats = indices.reduceRegion(
        reducer=ee.Reducer.mean(),
        geometry=bengaluru,
        scale=100,
        maxPixels=1e9
    ).getInfo()

    return stats.get('NDVI'), stats.get('NDBI')

# Build monthly time series 2002-2026
results = []
for year in range(2002, 2027):
    for month in range(1, 13):
        if year == 2026 and month > 5:
            break
        satellite = 'L7' if year < 2013 else 'L8'
        try:
            ndvi, ndbi = monthly_mean(year, month, satellite)
            print(f"{year}-{month:02d}: NDVI={ndvi}, NDBI={ndbi}")
            results.append([f"{year}-{month:02d}-01", ndvi, ndbi, satellite])
        except Exception as e:
            print(f"{year}-{month:02d}: FAILED - {e}")
            results.append([f"{year}-{month:02d}-01", None, None, satellite])

# Save to CSV
with open('bengaluru_ndvi_ndbi.csv', 'w', newline='') as f:
    writer = csv.writer(f)
    writer.writerow(['date', 'NDVI', 'NDBI', 'satellite'])
    writer.writerows(results)

print("Done! Saved to bengaluru_ndvi_ndbi.csv")