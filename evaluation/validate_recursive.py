

"""
TRUE 24-Month Recursive Forecast Validation
Tests how accurate the RECURSIVE forecasting process actually is, not just
1-month-ahead accuracy. Picks past cutoff dates, forecasts 24 months forward
using ONLY data available before that cutoff (recursively, using predicted
values as inputs - exactly like the real dashboard forecast does), then
compares against the REAL observed values for those 24 months.

Run this from D:\research\scripts after merge_panel.py has completed.
"""

import pandas as pd
import numpy as np
from xgboost import XGBRegressor
from sklearn.metrics import mean_absolute_error, r2_score

LAGS = [1, 2, 3, 6, 12]
feature_cols = ['GRACE_TWS_anomaly_cm', 'NDVI', 'NDBI', 'precipitation_mm',
                 'temperature_C', 'potential_evaporation_mm']
DISTRICTS = ['Bengaluru Urban', 'Mysuru', 'Ballari', 'Uttara Kannada',
             'Kalaburagi', 'Vijayapura']
FORECAST_MONTHS = 24

print("=== STEP 1: Load panel dataset ===")
df = pd.read_csv('../data/processed/panel_dataset.csv', parse_dates=['date'])
df = df.sort_values(['district', 'date'])

# Cutoff points: we need 24 months of REAL data after the cutoff to compare
# against, so cutoffs must be at least 24 months before the last date.
last_date = df['date'].max()
CUTOFFS = [last_date - pd.DateOffset(months=24 + 24*2),
           last_date - pd.DateOffset(months=24 + 24*1),
           last_date - pd.DateOffset(months=24)]
print(f"Testing {len(CUTOFFS)} cutoff origins: {[c.strftime('%Y-%m') for c in CUTOFFS]}")
print(f"(each forecasts 24 months forward, compared against real observed values)")


def build_features(data):
    pieces = []
    for district, sub in data.groupby('district'):
        sub = sub.sort_values('date').copy()
        for col in feature_cols:
            for lag in LAGS:
                sub[f'{col}_lag{lag}'] = sub[col].shift(lag)
        sub['GRACE_roll3'] = sub['GRACE_TWS_anomaly_cm'].rolling(3).mean()
        sub['target'] = sub['GRACE_TWS_anomaly_cm'].shift(-1)
        pieces.append(sub)
    feat = pd.concat(pieces, ignore_index=True)
    dummies = pd.get_dummies(feat['district'], prefix='district')
    feat = pd.concat([feat, dummies], axis=1)
    return feat


all_results = []

for cutoff in CUTOFFS:
    print(f"\n{'='*60}")
    print(f"CUTOFF: {cutoff.strftime('%Y-%m')} (train on data before this, forecast 24mo forward)")
    print(f"{'='*60}")

    train_data = df[df['date'] <= cutoff].copy()
    feat = build_features(train_data)
    feat_model = feat.dropna()

    feature_names = [c for c in feat_model.columns if c not in
                      ['date', 'district', 'GRACE_TWS_anomaly_cm', 'NDVI', 'NDBI',
                       'precipitation_mm', 'temperature_C', 'potential_evaporation_mm',
                       'population_density', 'target']]

    model = XGBRegressor(n_estimators=200, max_depth=4, learning_rate=0.05,
                          subsample=0.8, colsample_bytree=0.8, random_state=42)
    model.fit(feat_model[feature_names], feat_model['target'])

    for district in DISTRICTS:
        dist_train = train_data[train_data['district'] == district].set_index('date').sort_index()
        dist_train['month_num'] = dist_train.index.month
        climatology = dist_train.groupby('month_num')[['NDVI', 'NDBI', 'precipitation_mm',
                                                          'temperature_C', 'potential_evaporation_mm']].mean()

        history = dist_train[feature_cols].copy()
        future_dates = pd.date_range(cutoff + pd.DateOffset(months=1),
                                      periods=FORECAST_MONTHS, freq='MS')

        recursive_preds = []
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
                recursive_preds.append(np.nan)
                continue

            pred = model.predict(X_future)[0]
            recursive_preds.append(pred)

            history.loc[future_date] = {
                'GRACE_TWS_anomaly_cm': pred, 'NDVI': clim['NDVI'], 'NDBI': clim['NDBI'],
                'precipitation_mm': clim['precipitation_mm'], 'temperature_C': clim['temperature_C'],
                'potential_evaporation_mm': clim['potential_evaporation_mm'],
            }

        # Compare against REAL observed values for this same period
        actual = df[(df['district'] == district) &
                     (df['date'].isin(future_dates))].set_index('date')['GRACE_TWS_anomaly_cm']
        actual_aligned = actual.reindex(future_dates)

        valid = ~(actual_aligned.isna() | pd.isna(recursive_preds))
        if valid.sum() > 0:
            mae = mean_absolute_error(actual_aligned[valid], np.array(recursive_preds)[valid])
            r2 = r2_score(actual_aligned[valid], np.array(recursive_preds)[valid])
        else:
            mae, r2 = np.nan, np.nan

        all_results.append({
            'cutoff': cutoff.strftime('%Y-%m'), 'district': district,
            'MAE_24mo_recursive': mae, 'R2_24mo_recursive': r2,
            'valid_months': int(valid.sum())
        })
        print(f"  {district}: 24-month recursive MAE={mae:.2f} cm, R2={r2:.3f} "
              f"({int(valid.sum())} months compared)")

results_df = pd.DataFrame(all_results)
print(f"\n{'='*60}")
print("=== TRUE 24-MONTH RECURSIVE FORECAST ACCURACY (honest) ===")
print(f"{'='*60}")
print(f"Mean MAE across all cutoffs/districts: {results_df['MAE_24mo_recursive'].mean():.2f} cm")
print(f"Mean R2 across all cutoffs/districts:  {results_df['R2_24mo_recursive'].mean():.3f}")

print("\n=== FOR COMPARISON: 1-month-ahead validation (from train_panel_model.py) ===")
print("XGBoost 1-month-ahead -> Mean MAE: 6.15 cm, Mean R2: 0.646")

results_df.to_csv('../data/processed/recursive_24month_validation.csv', index=False)
print("\nSaved: ../data/processed/recursive_24month_validation.csv")
