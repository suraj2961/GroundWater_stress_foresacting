"""
CGWB Stress Category Classification Accuracy Test
Tests how often the model's predicted stress CATEGORY (Safe/Semi-Critical/
Critical/Over-Exploited) matches the ACTUAL category, at each horizon.
This is a different, complementary metric to R2/MAE - category classification
is more forgiving than exact value prediction.

Run this from D:\research\scripts after merge_panel.py has completed.
"""

import pandas as pd
import numpy as np
from xgboost import XGBRegressor
from sklearn.metrics import accuracy_score

LAGS = [1, 2, 3, 6, 12]
feature_cols = ['GRACE_TWS_anomaly_cm', 'NDVI', 'NDBI', 'precipitation_mm',
                 'temperature_C', 'potential_evaporation_mm']
DISTRICTS = ['Bengaluru Urban', 'Mysuru', 'Ballari', 'Uttara Kannada',
             'Kalaburagi', 'Vijayapura']
HORIZONS_TO_TEST = [1, 3, 6, 12, 24]

print("=== STEP 1: Load panel dataset ===")
df = pd.read_csv('../data/processed/panel_dataset.csv', parse_dates=['date'])
df = df.sort_values(['district', 'date'])
last_date = df['date'].max()

CUTOFFS = [last_date - pd.DateOffset(months=24 + 24*2),
           last_date - pd.DateOffset(months=24 + 24*1),
           last_date - pd.DateOffset(months=24)]


def categorize(val, p25, p50, p75):
    if val <= p25: return 'Over-Exploited'
    elif val <= p50: return 'Critical'
    elif val <= p75: return 'Semi-Critical'
    else: return 'Safe'


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


def recursive_forecast(model, feature_names, train_data, district, cutoff, max_horizon):
    dist_train = train_data[train_data['district'] == district].set_index('date').sort_index()
    dist_train['month_num'] = dist_train.index.month
    climatology = dist_train.groupby('month_num')[['NDVI', 'NDBI', 'precipitation_mm',
                                                      'temperature_C', 'potential_evaporation_mm']].mean()
    history = dist_train[feature_cols].copy()
    future_dates = pd.date_range(cutoff + pd.DateOffset(months=1), periods=max_horizon, freq='MS')
    preds = []
    for future_date in future_dates:
        clim = climatology.loc[future_date.month]
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
            preds.append(np.nan)
            continue
        pred = model.predict(X_future)[0]
        preds.append(pred)
        history.loc[future_date] = {
            'GRACE_TWS_anomaly_cm': pred, 'NDVI': clim['NDVI'], 'NDBI': clim['NDBI'],
            'precipitation_mm': clim['precipitation_mm'], 'temperature_C': clim['temperature_C'],
            'potential_evaporation_mm': clim['potential_evaporation_mm'],
        }
    return future_dates, preds


all_results = []

for cutoff in CUTOFFS:
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
        hist_vals = train_data[train_data['district'] == district]['GRACE_TWS_anomaly_cm'].dropna()
        p25, p50, p75 = hist_vals.quantile([0.25, 0.50, 0.75])

        future_dates, preds = recursive_forecast(model, feature_names, train_data,
                                                    district, cutoff, max(HORIZONS_TO_TEST))
        actual = df[(df['district'] == district) &
                     (df['date'].isin(future_dates))].set_index('date')['GRACE_TWS_anomaly_cm']
        actual_aligned = actual.reindex(future_dates)

        for h in HORIZONS_TO_TEST:
            preds_h = preds[:h]
            actual_h = actual_aligned.iloc[:h]
            valid = ~(actual_h.isna() | pd.isna(preds_h))
            if valid.sum() == 0:
                continue
            pred_cats = [categorize(v, p25, p50, p75) for v in np.array(preds_h)[valid]]
            actual_cats = [categorize(v, p25, p50, p75) for v in actual_h[valid]]
            acc = accuracy_score(actual_cats, pred_cats)
            all_results.append({'cutoff': cutoff.strftime('%Y-%m'), 'district': district,
                                 'horizon_months': h, 'category_accuracy': acc, 'n_months': int(valid.sum())})

results_df = pd.DataFrame(all_results)
print("\n=== CATEGORY CLASSIFICATION ACCURACY by horizon (avg across districts/cutoffs) ===")
summary = results_df.groupby('horizon_months')['category_accuracy'].mean().reset_index()
summary['category_accuracy_pct'] = (summary['category_accuracy'] * 100).round(1)
print(summary.to_string(index=False))

results_df.to_csv('../data/processed/category_accuracy_test.csv', index=False)
print("\nSaved: ../data/processed/category_accuracy_test.csv")
print("\nNOTE: This is CATEGORY classification accuracy (4 classes), a different metric")
print("from R2 (continuous value accuracy). Report both, do not conflate them.")

