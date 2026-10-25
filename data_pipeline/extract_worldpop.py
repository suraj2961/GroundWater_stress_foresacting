import rasterio
from rasterio.windows import from_bounds
import numpy as np

# Bengaluru Urban district bounding box (same as used for Earth Engine)
# [West, South, East, North]
bbox = (77.35, 12.75, 77.85, 13.15)

with rasterio.open('ind_pd_2020_1km.tif') as src:
    window = from_bounds(*bbox, transform=src.transform)
    data = src.read(1, window=window)
    
    # Remove nodata values before averaging
    nodata = src.nodata
    if nodata is not None:
        data = data[data != nodata]
    
    mean_density = np.nanmean(data)
    max_density = np.nanmax(data)
    
    print(f"Bengaluru Urban - Mean population density (2020): {mean_density:.2f} people/km²")
    print(f"Bengaluru Urban - Max population density (2020): {max_density:.2f} people/km²")

    # Save to a simple CSV for merging later
    with open('bengaluru_population_density.csv', 'w') as f:
        f.write("year,mean_density,max_density\n")
        f.write(f"2020,{mean_density:.2f},{max_density:.2f}\n")

print("Saved to bengaluru_population_density.csv")