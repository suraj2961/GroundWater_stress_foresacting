"""
Stacked Ensemble Multi-Horizon Test
Tests whether combining Random Forest + XGBoost + Neural Network (stacked
ensemble, as used in recent published groundwater forecasting research)
improves accuracy at longer forecast horizons compared to XGBoost alone.

Run this from D:\research\scripts after merge_panel.py has completed.
"""

import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, r2_score

LAGS = [1, 2, 3, 6, 12]
feature_cols = ['GRACE_TWS_anomaly_cm', 'NDVI', 'NDBI', 'precipitation_mm',
                 'temperature_C', 'potential_evaporation_mm']
DISTRICTS = ['Bengaluru Urban', 'Mysuru', 'Ballari', 'Uttara Kannada',
             'Kalaburagi', 'Vijayapura']
HORIZONS_TO_TEST = [3, 6, 12, 24]

print("=== STEP 1: Load panel dataset ===")
df = pd.read_csv('../data/processed/panel_dataset.csv', parse_dates=['date'])
df = df.sort_values(['district', 'date'])
last_date = df['date'].max()

CUTOFFS = [last_date - pd.DateOffset(months=24 + 24*2),
           last_date - pd.DateOffset(months=24 + 24*1),
           last_date - pd.DateOffset(months=24)]


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


class StackedEnsemble:
    """RF + XGBoost + MLP, averaged (simple stacking)."""
    def __init__(self):
        self.rf = RandomForestRegressor(n_estimators=200, max_depth=8, random_state=42)
        self.xgb = XGBRegressor(n_estimators=200, max_depth=4, learning_rate=0.05,
                                 subsample=0.8, colsample_bytree=0.8, random_state=42)
        self.mlp = MLPRegressor(hidden_layer_sizes=(64, 32), max_iter=2000,
                                 random_state=42, early_stopping=True)
        self.scaler = StandardScaler()

    def fit(self, X, y):
        self.rf.fit(X, y)
        self.xgb.fit(X, y)
        Xs = self.scaler.fit_transform(X)
        self.mlp.fit(Xs, y)
        return self

    def predict(self, X):
        p_rf = self.rf.predict(X)
        p_xgb = self.xgb.predict(X)
        p_mlp = self.mlp.predict(self.scaler.transform(X))
        return (p_rf + p_xgb + p_mlp) / 3


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
    print(f"\n--- Cutoff {cutoff.strftime('%Y-%m')}: training stacked ensemble ---")
    train_data = df[df['date'] <= cutoff].copy()
    feat = build_features(train_data)
    feat_model = feat.dropna()
    feature_names = [c for c in feat_model.columns if c not in
                      ['date', 'district', 'GRACE_TWS_anomaly_cm', 'NDVI', 'NDBI',
                       'precipitation_mm', 'temperature_C', 'potential_evaporation_mm',
                       'population_density', 'target']]

    model = StackedEnsemble()
    model.fit(feat_model[feature_names], feat_model['target'])

    for district in DISTRICTS:
        future_dates, preds = recursive_forecast(model, feature_names, train_data,
                                                    district, cutoff, max(HORIZONS_TO_TEST))
        actual = df[(df['district'] == district) &
                     (df['date'].isin(future_dates))].set_index('date')['GRACE_TWS_anomaly_cm']
        actual_aligned = actual.reindex(future_dates)

        for h in HORIZONS_TO_TEST:
            preds_h = preds[:h]
            actual_h = actual_aligned.iloc[:h]
            valid = ~(actual_h.isna() | pd.isna(preds_h))
            if valid.sum() < 2:
                continue
            mae = mean_absolute_error(actual_h[valid], np.array(preds_h)[valid])
            r2 = r2_score(actual_h[valid], np.array(preds_h)[valid])
            all_results.append({'cutoff': cutoff.strftime('%Y-%m'), 'district': district,
                                 'horizon_months': h, 'ensemble_MAE': mae, 'ensemble_R2': r2})

results_df = pd.DataFrame(all_results)
print("\n=== Stacked Ensemble accuracy by horizon (avg across districts/cutoffs) ===")
summary = results_df.groupby('horizon_months').agg(
    mean_MAE=('ensemble_MAE', 'mean'), mean_R2=('ensemble_R2', 'mean')).reset_index()
print(summary.to_string(index=False))

print("\n=== FOR COMPARISON: XGBoost-only results (from test_horizons.py) ===")
print(" horizon_months  mean_MAE   mean_R2")
print("              3  8.144120  0.015779")
print("              6 10.471705 -0.104533")
print("             12 13.741893 -0.576481")
print("             24 13.478066 -0.366353")

results_df.to_csv('../data/processed/stacked_ensemble_horizon_test.csv', index=False)
print("\nSaved: ../data/processed/stacked_ensemble_horizon_test.csv")