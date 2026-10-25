"""
Groundwater Forecasting - Feature Engineering + Baseline Model
Builds lag features and trains a Random Forest to predict next-month
GRACE TWS anomaly, validated with walk-forward validation.

Run this from D:\research after clean_merge.py has produced master_dataset_raw.csv
"""

import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, r2_score

print("=== STEP 1: Load master dataset ===")
df = pd.read_csv('../data/processed/master_dataset_raw.csv', parse_dates=['date'])
df = df.set_index('date').sort_index()
print(f"Loaded {len(df)} rows, columns: {df.columns.tolist()}")


print("\n=== STEP 2: Feature engineering (lag features) ===")
LAGS = [1, 3, 6, 12]
feature_cols = ['GRACE_TWS_anomaly_cm', 'NDVI', 'NDBI', 'precipitation_mm',
                 'temperature_C', 'potential_evaporation_mm']

for col in feature_cols:
    for lag in LAGS:
        df[f'{col}_lag{lag}'] = df[col].shift(lag)

df['GRACE_roll3'] = df['GRACE_TWS_anomaly_cm'].rolling(3).mean()
df['target'] = df['GRACE_TWS_anomaly_cm'].shift(-1)

df_model = df.dropna()
print(f"Rows usable for modeling after adding lags: {len(df_model)} "
      f"(lost {len(df) - len(df_model)} rows to lag warm-up + missing target)")

feature_names = [c for c in df_model.columns if c not in
                  ['GRACE_TWS_anomaly_cm', 'NDVI', 'NDBI', 'precipitation_mm',
                   'temperature_C', 'potential_evaporation_mm', 'population_density', 'target']]
print(f"Number of features: {len(feature_names)}")


print("\n=== STEP 3: Walk-forward validation ===")
N_SPLITS = 4
TEST_SIZE = 12

X = df_model[feature_names]
y = df_model['target']

results = []
n = len(df_model)
min_train = n - N_SPLITS * TEST_SIZE

for i in range(N_SPLITS):
    train_end = min_train + i * TEST_SIZE
    test_end = train_end + TEST_SIZE

    X_train, y_train = X.iloc[:train_end], y.iloc[:train_end]
    X_test, y_test = X.iloc[train_end:test_end], y.iloc[train_end:test_end]

    if len(X_test) == 0:
        continue

    model = RandomForestRegressor(n_estimators=200, max_depth=8, random_state=42)
    model.fit(X_train, y_train)
    preds = model.predict(X_test)

    mae = mean_absolute_error(y_test, preds)
    r2 = r2_score(y_test, preds)
    results.append({'fold': i+1, 'train_size': len(X_train), 'test_size': len(X_test),
                     'MAE': mae, 'R2': r2})
    print(f"Fold {i+1}: train={len(X_train)} months, test={len(X_test)} months, "
          f"MAE={mae:.2f} cm, R2={r2:.3f}")

results_df = pd.DataFrame(results)
print("\n=== Average performance across folds ===")
print(f"Mean MAE: {results_df['MAE'].mean():.2f} cm")
print(f"Mean R2:  {results_df['R2'].mean():.3f}")


print("\n=== STEP 4: Train final model on ALL data (for forecasting later) ===")
final_model = RandomForestRegressor(n_estimators=200, max_depth=8, random_state=42)
final_model.fit(X, y)
print("Final model trained on full dataset, ready for use in forecasting step.")

importances = pd.Series(final_model.feature_importances_, index=feature_names)
importances = importances.sort_values(ascending=False)
print("\nTop 10 most important features:")
print(importances.head(10))

results_df.to_csv('../data/processed/validation_results.csv', index=False)
importances.to_csv('../data/processed/feature_importance.csv')
print("\nSaved: validation_results.csv, feature_importance.csv")
print("\n=== DIAGNOSTIC: What's going on in Fold 1? ===")
fold1_test_start = min_train
fold1_test_end = min_train + TEST_SIZE
fold1_dates = df_model.index[fold1_test_start:fold1_test_end]
fold1_target = y.iloc[fold1_test_start:fold1_test_end]
print(f"Fold 1 test period: {fold1_dates.min()} to {fold1_dates.max()}")
print(f"Fold 1 target (actual GRACE next-month) stats:")
print(fold1_target.describe())