"""
=============================================================================
Caribbean AI Academy - Sixth Form Capstone Project
Climate Change Impact Predictor — Main Pipeline
=============================================================================
Predicts climate change impacts (flood risk, agricultural damage, tourism
disruption) for Caribbean nations using historical climate data.

Run data_generator.py FIRST to create the dataset!

TODOs: Students must complete the marked sections.

Requirements: pip install numpy scikit-learn
Author: Adrian Dunkley | Caribbean AI Academy
=============================================================================
"""

import numpy as np
import json
import os

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, r2_score

print("=" * 60)
print("  Caribbean Climate Change Impact Predictor")
print("=" * 60)


# =====================================================================
# STEP 1: Load the Data
# =====================================================================
print("\n--- Step 1: Loading Caribbean Climate Data ---\n")

script_dir = os.path.dirname(os.path.abspath(__file__))
data_path = os.path.join(script_dir, 'caribbean_climate_data.json')

if not os.path.exists(data_path):
    print("ERROR: Data file not found! Run data_generator.py first.")
    print(f"Expected path: {data_path}")
    exit(1)

with open(data_path, 'r') as f:
    raw_data = json.load(f)

print(f"Loaded data for {len(raw_data)} Caribbean nations")

# Feature and target column names
FEATURE_COLS = [
    'avg_temperature', 'sea_level_rise_mm', 'annual_rainfall_mm',
    'hurricane_intensity', 'coral_bleaching_pct', 'drought_months'
]
TARGET_COLS = ['flood_risk_index', 'agricultural_impact', 'tourism_impact']


# =====================================================================
# STEP 2: Prepare Feature Matrix and Target Vectors
# =====================================================================
print("--- Step 2: Preparing Data ---\n")

X_all = []
y_all = []
countries = []
years = []

for country, records in raw_data.items():
    for record in records:
        features = [record[col] for col in FEATURE_COLS]
        targets = [record[col] for col in TARGET_COLS]
        X_all.append(features)
        y_all.append(targets)
        countries.append(country)
        years.append(record['year'])

X_all = np.array(X_all)
y_all = np.array(y_all)

print(f"Total samples: {X_all.shape[0]}")
print(f"Features ({len(FEATURE_COLS)}): {FEATURE_COLS}")
print(f"Targets ({len(TARGET_COLS)}): {TARGET_COLS}")


# =====================================================================
# STEP 3: Train/Test Split
# =====================================================================
print("\n--- Step 3: Splitting Data ---\n")

# TODO 1: Split the data into training (80%) and testing (20%) sets.
# Use train_test_split with random_state=42 for reproducibility.
# Hint: X_train, X_test, y_train, y_test = train_test_split(...)
X_train, X_test, y_train, y_test = train_test_split(
    X_all, y_all, test_size=0.2, random_state=42
)

print(f"Training samples: {len(X_train)}")
print(f"Testing samples: {len(X_test)}")


# =====================================================================
# STEP 4: Feature Scaling
# =====================================================================
print("\n--- Step 4: Scaling Features ---\n")

# TODO 2: Create a StandardScaler and fit_transform on training data.
# Then transform the test data (do NOT fit on test data — why?).
# Hint: scaler = StandardScaler()
#       X_train_scaled = scaler.fit_transform(X_train)
#       X_test_scaled = scaler.transform(X_test)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

print("Features scaled using StandardScaler")
print("(Remember: fit on training data ONLY to avoid data leakage!)")


# =====================================================================
# STEP 5: Train Models for Each Target
# =====================================================================
print("\n--- Step 5: Training Prediction Models ---\n")

models = {}
results = {}

for i, target_name in enumerate(TARGET_COLS):
    print(f"\n  Training model for: {target_name}")
    print(f"  {'-' * 40}")

    y_tr = y_train[:, i]
    y_te = y_test[:, i]

    # TODO 3: Train a RandomForestRegressor for this target.
    # Use n_estimators=100, random_state=42.
    # Then predict on the test set and calculate MAE and R2 score.
    # Hint:
    #   model = RandomForestRegressor(n_estimators=100, random_state=42)
    #   model.fit(X_train_scaled, y_tr)
    #   predictions = model.predict(X_test_scaled)
    #   mae = mean_absolute_error(y_te, predictions)
    #   r2 = r2_score(y_te, predictions)

    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X_train_scaled, y_tr)
    predictions = model.predict(X_test_scaled)
    mae = mean_absolute_error(y_te, predictions)
    r2 = r2_score(y_te, predictions)

    models[target_name] = model
    results[target_name] = {'mae': mae, 'r2': r2}

    print(f"  MAE: {mae:.2f}")
    print(f"  R2 Score: {r2:.4f}")

    # Feature importance
    importances = model.feature_importances_
    print(f"  Feature Importance:")
    for fname, imp in sorted(zip(FEATURE_COLS, importances),
                              key=lambda x: -x[1]):
        bar = '#' * int(imp * 40)
        print(f"    {fname:<24} {imp:.3f} {bar}")


# =====================================================================
# STEP 6: Compare with Baseline Models
# =====================================================================
print("\n--- Step 6: Model Comparison ---\n")

# TODO 4: Train a LinearRegression model on the same data and compare.
# Does the Random Forest outperform the simpler model?
# Hint: Follow the same pattern as Step 5 but use LinearRegression()

print(f"{'Target':<24} {'RF MAE':>8} {'RF R2':>8} {'LR MAE':>8} {'LR R2':>8}")
print("-" * 56)

for i, target_name in enumerate(TARGET_COLS):
    y_tr = y_train[:, i]
    y_te = y_test[:, i]

    lr_model = LinearRegression()
    lr_model.fit(X_train_scaled, y_tr)
    lr_pred = lr_model.predict(X_test_scaled)
    lr_mae = mean_absolute_error(y_te, lr_pred)
    lr_r2 = r2_score(y_te, lr_pred)

    rf_mae = results[target_name]['mae']
    rf_r2 = results[target_name]['r2']

    print(f"{target_name:<24} {rf_mae:>8.2f} {rf_r2:>8.3f} "
          f"{lr_mae:>8.2f} {lr_r2:>8.3f}")


# =====================================================================
# STEP 7: Future Predictions (2030-2050)
# =====================================================================
print("\n--- Step 7: Future Climate Impact Predictions ---\n")

# TODO 5: Create synthetic future climate scenarios and predict impacts.
# For each future year, extrapolate trends from the data.
# Students should think about: what assumptions are we making?
# Are these predictions reliable? What are the limitations?

print("Projected Climate Impacts (2030-2050):")
print(f"{'Year':<6} {'Temp(C)':<9} {'SLR(mm)':<9} {'Rain(mm)':<10} "
      f"{'Hurr':<6} {'Flood':<8} {'AgriDmg':<8} {'TourDmg':<8}")
print("-" * 64)

for future_year in range(2030, 2051, 5):
    decade_from_1970 = (future_year - 1970) / 10.0

    # Extrapolate features based on observed trends
    temp = 27.0 + 0.018 * (future_year - 1970)
    slr = 2.0 * decade_from_1970 + 0.15 * decade_from_1970**2
    rainfall = 1800 - 5 * decade_from_1970
    hurr = 2.0 + 0.08 * decade_from_1970
    bleach = min(5 + 8 * decade_from_1970, 80)
    drought = min(int(1.5 + 0.3 * decade_from_1970), 8)

    future_features = np.array([[temp, slr, rainfall, hurr, bleach, drought]])
    future_scaled = scaler.transform(future_features)

    preds = {}
    for target_name, model in models.items():
        preds[target_name] = model.predict(future_scaled)[0]

    print(f"{future_year:<6} {temp:<9.1f} {slr:<9.1f} {rainfall:<10.0f} "
          f"{hurr:<6.1f} {preds['flood_risk_index']:<8.1f} "
          f"{preds['agricultural_impact']:<8.1f} "
          f"{preds['tourism_impact']:<8.1f}")


# =====================================================================
# STEP 8: Country-Specific Analysis
# =====================================================================
print("\n--- Step 8: Country-Specific Vulnerability ---\n")

# TODO 6: For each Caribbean nation, calculate average predicted
# impact scores. Which nations are most vulnerable?
# Think about: Why are some nations more vulnerable?
# What geographic/economic factors contribute?

# Use the latest year's data for each country
print("Most recent year vulnerability by nation:")
print(f"{'Nation':<22} {'FloodRisk':>10} {'AgriImpact':>12} {'TourImpact':>12}")
print("-" * 58)

nation_scores = {}
for country, records in raw_data.items():
    last_record = records[-1]
    features = np.array([[last_record[col] for col in FEATURE_COLS]])
    features_scaled = scaler.transform(features)

    scores = {}
    for target_name, model in models.items():
        scores[target_name] = model.predict(features_scaled)[0]
    nation_scores[country] = scores

# Sort by average vulnerability
sorted_nations = sorted(
    nation_scores.items(),
    key=lambda x: sum(x[1].values()) / len(x[1]),
    reverse=True
)

for country, scores in sorted_nations:
    print(f"{country:<22} {scores['flood_risk_index']:>10.1f} "
          f"{scores['agricultural_impact']:>12.1f} "
          f"{scores['tourism_impact']:>12.1f}")


# =====================================================================
# FINAL: Summary and Reflection
# =====================================================================
print("\n" + "=" * 60)
print("  Project Summary")
print("=" * 60)
print("""
REFLECTION QUESTIONS (discuss with your class):

1. Which Caribbean nations appear most vulnerable to climate change?
   Why might that be? (Hint: think about elevation, size, economy)

2. What are the limitations of using historical data to predict
   future climate impacts? What assumptions are we making?

3. How could this model be improved with real data from Caribbean
   meteorological agencies (CIMH, national weather services)?

4. If you were presenting this to CARICOM leaders, what policy
   recommendations would you make based on the predictions?

5. How does this project connect to AI ethics (Lesson 06)?
   Who benefits from these predictions? Who might be harmed?

EXTENSION CHALLENGES:
- Add a GradientBoostingRegressor and compare performance
- Implement cross-validation instead of a single train/test split
- Create per-country models instead of one regional model
- Add economic features (GDP, tourism revenue) as inputs
- Build an interactive version that lets users explore scenarios
""")

print("Project complete! Big up yuhself for building climate AI!")
print("=" * 60)
