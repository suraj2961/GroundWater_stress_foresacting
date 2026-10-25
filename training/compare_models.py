"""
Groundwater Forecasting - Improved Model (XGBoost + Richer Features)
Compares against the Random Forest baseline using the same walk-forward
validation setup, so results are directly comparable.

Run this from D:\research\scripts after clean_merge.py has produced
../data/processed/master_dataset_raw.csv
"""

import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
from sklearn.metrics import mean_absolute_error, r2_score

print("=== STEP 1: Load master dataset ===")
df = pd.read_csv('../data/processed/master_dataset_raw.csv', parse_dates=['date'])
df = df.set_index('date').sort_index()
print(f"Loaded {len(df)} rows")


print("\n=== STEP 2: Richer feature engineering ===")
LAGS = [1, 2, 3, 4, 6, 8, 12, 18]
feature_cols = ['GRACE_TWS_anomaly_cm', 'NDVI', 'NDBI', 'precipitation_mm',
                 'temperature_C', 'potential_evaporation_mm']

for col in feature_cols:
    for lag in LAGS:
        df[f'{col}_lag{lag}'] = df[col].shift(lag)

df['GRACE_roll3'] = df['GRACE_TWS_anomaly_cm'].rolling(3).mean()
df['GRACE_roll6'] = df['GRACE_TWS_anomaly_cm'].rolling(6).mean()
df['GRACE_roll12_std'] = df['GRACE_TWS_anomaly_cm'].rolling(12).std()

df['precip_x_temp'] = df['precipitation_mm'] * df['temperature_C']

df['month_sin'] = np.sin(2 * np.pi * df.index.month / 12)
df['month_cos'] = np.cos(2 * np.pi * df.index.month / 12)

df['time_index'] = np.arange(len(df))

df['target'] = df['GRACE_TWS_anomaly_cm'].shift(-1)
df_model = df.dropna()
print(f"Rows usable: {len(df_model)}")

feature_names = [c for c in df_model.columns if c not in
                  ['GRACE_TWS_anomaly_cm', 'NDVI', 'NDBI', 'precipitation_mm',
                   'temperature_C', 'potential_evaporation_mm', 'population_density', 'target']]
print(f"Number of features: {len(feature_names)}")

X = df_model[feature_names]
y = df_model['target']


print("\n=== STEP 3: Walk-forward validation - COMPARE both models ===")
N_SPLITS = 4
TEST_SIZE = 12
n = len(df_model)
min_train = n - N_SPLITS * TEST_SIZE

all_results = []

for i in range(N_SPLITS):
    train_end = min_train + i * TEST_SIZE
    test_end = train_end + TEST_SIZE

    X_train, y_train = X.iloc[:train_end], y.iloc[:train_end]
    X_test, y_test = X.iloc[train_end:test_end], y.iloc[train_end:test_end]

    if len(X_test) == 0:
        continue

    rf = RandomForestRegressor(n_estimators=200, max_depth=8, random_state=42)
    rf.fit(X_train, y_train)
    rf_preds = rf.predict(X_test)
    rf_mae = mean_absolute_error(y_test, rf_preds)
    rf_r2 = r2_score(y_test, rf_preds)

    xgb = XGBRegressor(n_estimators=200, max_depth=4, learning_rate=0.05,
                        subsample=0.8, colsample_bytree=0.8, random_state=42)
    xgb.fit(X_train, y_train)
    xgb_preds = xgb.predict(X_test)
    xgb_mae = mean_absolute_error(y_test, xgb_preds)
    xgb_r2 = r2_score(y_test, xgb_preds)

    all_results.append({
        'fold': i+1, 'test_size': len(X_test),
        'RF_MAE': rf_mae, 'RF_R2': rf_r2,
        'XGB_MAE': xgb_mae, 'XGB_R2': xgb_r2,
    })
    print(f"Fold {i+1}: RF  -> MAE={rf_mae:.2f}, R2={rf_r2:.3f}   |   "
          f"XGB -> MAE={xgb_mae:.2f}, R2={xgb_r2:.3f}")

results_df = pd.DataFrame(all_results)
print("\n=== Average performance across folds ===")
print(f"Random Forest -> Mean MAE: {results_df['RF_MAE'].mean():.2f} cm, Mean R2: {results_df['RF_R2'].mean():.3f}")
print(f"XGBoost       -> Mean MAE: {results_df['XGB_MAE'].mean():.2f} cm, Mean R2: {results_df['XGB_R2'].mean():.3f}")

print("\n=== Average performance EXCLUDING Fold 1 (2022 anomaly year) ===")
recent = results_df[results_df['fold'] != 1]
print(f"Random Forest -> Mean MAE: {recent['RF_MAE'].mean():.2f} cm, Mean R2: {recent['RF_R2'].mean():.3f}")
print(f"XGBoost       -> Mean MAE: {recent['XGB_MAE'].mean():.2f} cm, Mean R2: {recent['XGB_R2'].mean():.3f}")

results_df.to_csv('../data/processed/model_comparison.csv', index=False)
print("\nSaved: ../data/processed/model_comparison.csv")