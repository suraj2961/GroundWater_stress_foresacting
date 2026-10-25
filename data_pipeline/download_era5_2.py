import cdsapi

client = cdsapi.Client()

karnataka_bbox = [18.5, 74.0, 11.5, 78.5]

dataset = "reanalysis-era5-land-monthly-means"

request = {
    "product_type": "monthly_averaged_reanalysis",
    "variable": [
        "total_precipitation",
        "2m_temperature",
        "potential_evaporation"
    ],
    "year": ["2026"],
    "month": ["01", "02", "03", "04", "05"],  # match your GRACE data's latest month
    "time": "00:00",
    "area": karnataka_bbox,
    "format": "netcdf"
}

client.retrieve(dataset, request, "era5_karnataka_2026.nc")
print("Download complete: era5_karnataka_2026.nc")