import cdsapi

client = cdsapi.Client()

# Karnataka bounding box: [North, West, South, East]
karnataka_bbox = [18.5, 74.0, 11.5, 78.5]

dataset = "reanalysis-era5-land-monthly-means"

request = {
    "product_type": "monthly_averaged_reanalysis",
    "variable": [
        "total_precipitation",
        "2m_temperature",
        "potential_evaporation"
    ],
    "year": [str(y) for y in range(2002, 2026)],  # 2002-2025 full years
    "month": [f"{m:02d}" for m in range(1, 13)],
    "time": "00:00",
    "area": karnataka_bbox,
    "format": "netcdf"
}

client.retrieve(dataset, request, "era5_karnataka_2002_2025.nc")
print("Download complete: era5_karnataka_2002_2025.nc")