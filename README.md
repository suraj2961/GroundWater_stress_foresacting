# GroundWater_stress_foresacting
A machine learning pipeline that forecasts groundwater stress across Karnataka, India, giving planners, researchers and farmers an early warning of water scarcity.



## Overview

Groundwater depletion is a growing threat to agriculture and drinking water in Karnataka. This project combines satellite observations with climate reanalysis data to predict groundwater stress before it becomes critical. It compares several machine learning approaches and presents the forecasts in an interactive dashboard.

## Data Sources

- **GRACE:** satellite-derived terrestrial water storage anomalies
- **ERA5:** climate reanalysis variables such as rainfall and temperature
- **Google Earth Engine:** data extraction, aggregation and preprocessing

## Methodology

1. Collect and preprocess spatial and temporal data using Google Earth Engine.
2. Engineer features from climate variables and water-storage anomalies.
3. Train and compare three models:
   - Random Forest
   - XGBoost
   - Stacked ensemble combining both
4. Evaluate forecast accuracy across different prediction horizons.

## Key Results

- The stacked ensemble was benchmarked against the individual models.
- Forecasts are reliable up to **3 months ahead**.
- An interactive dashboard lets users explore and visualise predicted groundwater stress.

## Tech Stack

Python, scikit-learn, XGBoost, Google Earth Engine, GRACE, ERA5

## Impact

Early forecasts of groundwater stress can support better irrigation planning, crop selection and water management decisions. This is especially valuable for farming communities that depend on groundwater.

## Author

**Suraj**, MCA student with an interest in agri-tech and applications that create social impact.
