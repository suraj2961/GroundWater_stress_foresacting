"""
Multi-District 24-Month Recursive Forecast (XGBoost)
Generates a 24-month groundwater stress forecast for all 6 Karnataka
districts, using the best-performing model from validation (XGBoost).

Run this from D:\research\scripts after merge_panel.py has completed.
"""

import pandas as pd
import numpy as np
from xgboost import XGBRegressor

FORECAST_MONTHS = 24
LAGS = [1, 2, 3, 6, 12]
feature_cols = ['GRACE_TWS_anomaly_cm', 'NDVI', 'NDBI', 'precipitation_mm',
                 'temperature_C', 'potential_evaporation_mm']
DISTRICTS = ['Bengaluru Urban', 'Mysuru', 'Ballari', 'Uttara Kannada',
             'Kalaburagi', 'Vijayapura']

print("=== STEP 1: Load panel dataset ===")
df = pd.read_csv('../data/processed/panel_dataset.csv', parse_dates=['date'])
df = df.sort_values(['district', 'date'])
print(f"Loaded {len(df)} rows across {df['district'].nunique()} districts")


print("\n=== STEP 2: Build features and train final XGBoost model on ALL data ===")
pieces = []
for district, sub in df.groupby('district'):
    sub = sub.sort_values('date').copy()
    for col in feature_cols:
        for lag in LAGS:
            sub[f'{col}_lag{lag}'] = sub[col].shift(lag)
    sub['GRACE_roll3'] = sub['GRACE_TWS_anomaly_cm'].rolling(3).mean()
    sub['target'] = sub['GRACE_TWS_anomaly_cm'].shift(-1)
    pieces.append(sub)

df_feat = pd.concat(pieces, ignore_index=True)
district_dummies = pd.get_dummies(df_feat['district'], prefix='district')
df_feat = pd.concat([df_feat, district_dummies], axis=1)
df_model = df_feat.dropna()

feature_names = [c for c in df_model.columns if c not in
                  ['date', 'district', 'GRACE_TWS_anomaly_cm', 'NDVI', 'NDBI',
                   'precipitation_mm', 'temperature_C', 'potential_evaporation_mm',
                   'population_density', 'target']]

X = df_model[feature_names]
y = df_model['target']

model = XGBRegressor(n_estimators=200, max_depth=4, learning_rate=0.05,
                      subsample=0.8, colsample_bytree=0.8, random_state=42)
model.fit(X, y)
print(f"Model trained on {len(X)} rows using {len(feature_names)} features.")


print("\n=== STEP 3: Recursive 24-month forecast PER DISTRICT ===")
all_forecasts = []

for district in DISTRICTS:
    print(f"\n--- Forecasting {district} ---")
    dist_df = df[df['district'] == district].set_index('date').sort_index()

    dist_df['month_num'] = dist_df.index.month
    climatology = dist_df.groupby('month_num')[['NDVI', 'NDBI', 'precipitation_mm',
                                                   'temperature_C', 'potential_evaporation_mm']].mean()

    history = dist_df[feature_cols].copy()
    last_date = history.index.max()
    future_dates = pd.date_range(last_date + pd.DateOffset(months=1),
                                  periods=FORECAST_MONTHS, freq='MS')

    for future_date in future_dates:
        month_num = future_date.month
        clim = climatology.loc[month_num]

        row = {}
        for col in feature_cols:
            for lag in LAGS:
                lag_date = future_date - pd.DateOffset(months=lag)
                row[f'{col}_lag{lag}'] = history.loc[lag_date, col] if lag_date in history.index else np.nan

        roll_dates = [future_date - pd.DateOffset(months=m) for m in [1, 2, 3]]
        roll_vals = [history.loc[d, 'GRACE_TWS_anomaly_cm'] for d in roll_dates if d in history.index]
        row['GRACE_roll3'] = np.mean(roll_vals) if roll_vals else np.nan

        for d in DISTRICTS:
            row[f'district_{d}'] = 1 if d == district else 0

        X_future = pd.DataFrame([row])[feature_names]
        if X_future.isna().any().any():
            print(f"  WARNING: missing lag data for {future_date}, skipping")
            continue

        pred_grace = model.predict(X_future)[0]
        all_forecasts.append({'district': district, 'date': future_date,
                               'predicted_GRACE_TWS_anomaly_cm': pred_grace})

        history.loc[future_date] = {
            'GRACE_TWS_anomaly_cm': pred_grace,
            'NDVI': clim['NDVI'], 'NDBI': clim['NDBI'],
            'precipitation_mm': clim['precipitation_mm'],
            'temperature_C': clim['temperature_C'],
            'potential_evaporation_mm': clim['potential_evaporation_mm'],
        }

    print(f"  {district}: {FORECAST_MONTHS} months forecast complete")


forecast_df = pd.DataFrame(all_forecasts)


print("\n=== STEP 4: Map to CGWB-style categories (PER-DISTRICT thresholds) ===")
category_rows = []
for district in DISTRICTS:
    hist_vals = df[df['district'] == district]['GRACE_TWS_anomaly_cm'].dropna()
    p25, p50, p75 = hist_vals.quantile([0.25, 0.50, 0.75])

    def categorize(val, p25=p25, p50=p50, p75=p75):
        if val <= p25:
            return 'Over-Exploited'
        elif val <= p50:
            return 'Critical'
        elif val <= p75:
            return 'Semi-Critical'
        else:
            return 'Safe'

    sub = forecast_df[forecast_df['district'] == district].copy()
    sub['stress_category'] = sub['predicted_GRACE_TWS_anomaly_cm'].apply(categorize)
    category_rows.append(sub)

final_forecast = pd.concat(category_rows, ignore_index=True)
print(final_forecast.groupby('district')['stress_category'].value_counts())

final_forecast.to_csv('../data/processed/multi_district_forecast_24month.csv', index=False)
print("\nSaved: ../data/processed/multi_district_forecast_24month.csv")
print("MULTI-DISTRICT FORECAST COMPLETE")