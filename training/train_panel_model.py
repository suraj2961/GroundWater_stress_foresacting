"""
Multi-District Panel Model Training
Trains and compares Random Forest, XGBoost, and an ensemble of both
on the 6-district panel dataset, using walk-forward validation.

Run this from D:\research\scripts after merge_panel.py has completed.
"""

import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
from sklearn.metrics import mean_absolute_error, r2_score

print("=== STEP 1: Load panel dataset ===")
df = pd.read_csv('../data/processed/panel_dataset.csv', parse_dates=['date'])
df = df.sort_values(['district', 'date'])
print(f"Loaded {len(df)} rows across {df['district'].nunique()} districts")


print("\n=== STEP 2: Feature engineering (PER DISTRICT, so lags don't leak across districts) ===")
LAGS = [1, 2, 3, 6, 12]
feature_cols = ['GRACE_TWS_anomaly_cm', 'NDVI', 'NDBI', 'precipitation_mm',
                 'temperature_C', 'potential_evaporation_mm']

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
print(f"Rows usable for modeling: {len(df_model)} (across all districts combined)")

feature_names = [c for c in df_model.columns if c not in
                  ['date', 'district', 'GRACE_TWS_anomaly_cm', 'NDVI', 'NDBI',
                   'precipitation_mm', 'temperature_C', 'potential_evaporation_mm',
                   'population_density', 'target']]
print(f"Number of features: {len(feature_names)}")


print("\n=== STEP 3: Walk-forward validation (time-based, across all districts) ===")
dates_sorted = sorted(df_model['date'].unique())
N_SPLITS = 4
TEST_MONTHS = 12
n_dates = len(dates_sorted)
min_train_dates = n_dates - N_SPLITS * TEST_MONTHS

results = []

for i in range(N_SPLITS):
    train_end_idx = min_train_dates + i * TEST_MONTHS
    test_end_idx = train_end_idx + TEST_MONTHS

    train_dates = dates_sorted[:train_end_idx]
    test_dates = dates_sorted[train_end_idx:test_end_idx]

    train_mask = df_model['date'].isin(train_dates)
    test_mask = df_model['date'].isin(test_dates)

    X_train, y_train = df_model.loc[train_mask, feature_names], df_model.loc[train_mask, 'target']
    X_test, y_test = df_model.loc[test_mask, feature_names], df_model.loc[test_mask, 'target']

    if len(X_test) == 0:
        continue

    rf = RandomForestRegressor(n_estimators=200, max_depth=8, random_state=42)
    rf.fit(X_train, y_train)
    rf_preds = rf.predict(X_test)

    xgb = XGBRegressor(n_estimators=200, max_depth=4, learning_rate=0.05,
                        subsample=0.8, colsample_bytree=0.8, random_state=42)
    xgb.fit(X_train, y_train)
    xgb_preds = xgb.predict(X_test)

    ensemble_preds = (rf_preds + xgb_preds) / 2

    rf_mae, rf_r2 = mean_absolute_error(y_test, rf_preds), r2_score(y_test, rf_preds)
    xgb_mae, xgb_r2 = mean_absolute_error(y_test, xgb_preds), r2_score(y_test, xgb_preds)
    ens_mae, ens_r2 = mean_absolute_error(y_test, ensemble_preds), r2_score(y_test, ensemble_preds)

    results.append({
        'fold': i+1, 'test_rows': len(X_test),
        'RF_MAE': rf_mae, 'RF_R2': rf_r2,
        'XGB_MAE': xgb_mae, 'XGB_R2': xgb_r2,
        'Ensemble_MAE': ens_mae, 'Ensemble_R2': ens_r2,
    })
    print(f"Fold {i+1} ({test_dates[0].strftime('%Y-%m')} to {test_dates[-1].strftime('%Y-%m')}, "
          f"{len(X_test)} rows):")
    print(f"  RF       -> MAE={rf_mae:.2f}, R2={rf_r2:.3f}")
    print(f"  XGBoost  -> MAE={xgb_mae:.2f}, R2={xgb_r2:.3f}")
    print(f"  Ensemble -> MAE={ens_mae:.2f}, R2={ens_r2:.3f}")

results_df = pd.DataFrame(results)
print("\n=== Average performance across folds (MULTI-DISTRICT PANEL) ===")
print(f"Random Forest -> Mean MAE: {results_df['RF_MAE'].mean():.2f} cm, Mean R2: {results_df['RF_R2'].mean():.3f}")
print(f"XGBoost       -> Mean MAE: {results_df['XGB_MAE'].mean():.2f} cm, Mean R2: {results_df['XGB_R2'].mean():.3f}")
print(f"Ensemble      -> Mean MAE: {results_df['Ensemble_MAE'].mean():.2f} cm, Mean R2: {results_df['Ensemble_R2'].mean():.3f}")

print("\n=== FOR COMPARISON: your original single-district (Bengaluru only) results ===")
print("Random Forest -> Mean MAE: 7.25 cm, Mean R2: 0.117")
print("XGBoost       -> Mean MAE: 7.10 cm, Mean R2: 0.223")

results_df.to_csv('../data/processed/panel_model_comparison.csv', index=False)
print("\nSaved: ../data/processed/panel_model_comparison.csv")