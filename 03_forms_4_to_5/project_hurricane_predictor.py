"""
Caribbean AI Academy — Forms 4-5: Capstone Project
====================================================
PROJECT: Caribbean Hurricane Impact Predictor
Designed by Adrian Dunkley (https://Adriandunkley.net) — FREE!

YOUR MISSION: Build a model that predicts hurricane damage level
for Caribbean islands based on storm and island characteristics.

Complete the TODO sections to finish this project!
"""

import numpy as np

print("""
╔══════════════════════════════════════════════════════════╗
║  CAPSTONE PROJECT: HURRICANE IMPACT PREDICTOR            ║
║  Predict damage levels for Caribbean islands! 🌀         ║
║                                                          ║
║  Complete the TODO sections to finish this project.      ║
╚══════════════════════════════════════════════════════════╝
""")

# ============================================================
# STEP 1: Generate Synthetic Hurricane Data
# ============================================================
# This simulates real hurricane impact data for Caribbean islands

np.random.seed(42)

islands = {
    "Jamaica": {"size_km2": 10991, "infrastructure": 0.65, "population": 2961000},
    "Trinidad": {"size_km2": 5131, "infrastructure": 0.70, "population": 1399000},
    "Barbados": {"size_km2": 431, "infrastructure": 0.75, "population": 287000},
    "Bahamas": {"size_km2": 13878, "infrastructure": 0.68, "population": 393000},
    "Dominica": {"size_km2": 751, "infrastructure": 0.45, "population": 72000},
    "Grenada": {"size_km2": 344, "infrastructure": 0.50, "population": 113000},
    "St. Lucia": {"size_km2": 617, "infrastructure": 0.55, "population": 184000},
    "Antigua": {"size_km2": 442, "infrastructure": 0.60, "population": 98000},
    "St. Vincent": {"size_km2": 389, "infrastructure": 0.48, "population": 111000},
    "Haiti": {"size_km2": 27750, "infrastructure": 0.30, "population": 11400000},
    "Dominican Republic": {"size_km2": 48671, "infrastructure": 0.55, "population": 10850000},
    "Cuba": {"size_km2": 109884, "infrastructure": 0.60, "population": 11330000},
    "Puerto Rico": {"size_km2": 9104, "infrastructure": 0.65, "population": 3222000},
    "Cayman Islands": {"size_km2": 264, "infrastructure": 0.80, "population": 66000},
    "Belize": {"size_km2": 22966, "infrastructure": 0.45, "population": 398000},
    "Guyana": {"size_km2": 214969, "infrastructure": 0.40, "population": 787000},
}

n_samples = 500
data = []

for _ in range(n_samples):
    island_name = np.random.choice(list(islands.keys()))
    island_info = islands[island_name]

    # Hurricane characteristics
    category = np.random.choice([1, 2, 3, 4, 5], p=[0.3, 0.25, 0.2, 0.15, 0.1])
    wind_speed_mph = {1: 80, 2: 100, 3: 120, 4: 145, 5: 165}[category] + np.random.randint(-10, 10)
    distance_km = np.random.uniform(0, 200)  # Distance of eye from island center
    rainfall_mm = np.random.uniform(100, 500) * (category / 3)

    # Island characteristics
    size_km2 = island_info["size_km2"]
    infra_score = island_info["infrastructure"] + np.random.normal(0, 0.05)
    infra_score = np.clip(infra_score, 0, 1)

    # Calculate damage level based on realistic factors
    # Higher category + closer distance + lower infrastructure = more damage
    damage_score = (
        category * 0.3 +
        (200 - distance_km) / 200 * 0.3 +
        (1 - infra_score) * 0.2 +
        rainfall_mm / 500 * 0.1 +
        (1 / np.log(size_km2 + 1)) * 5 * 0.1  # Smaller islands more vulnerable
    )
    damage_score += np.random.normal(0, 0.1)

    # Convert to damage level: 0=low, 1=medium, 2=high
    if damage_score < 0.45:
        damage_level = 0
    elif damage_score < 0.7:
        damage_level = 1
    else:
        damage_level = 2

    data.append({
        "island": island_name,
        "category": category,
        "wind_speed": wind_speed_mph,
        "distance_km": distance_km,
        "rainfall_mm": rainfall_mm,
        "size_km2": size_km2,
        "infrastructure": infra_score,
        "damage_level": damage_level,
    })

print(f"Generated {len(data)} hurricane impact records")
print(f"Islands: {len(islands)} Caribbean nations/territories\n")

# Prepare features and labels
features = np.array([[d["category"], d["wind_speed"], d["distance_km"],
                       d["rainfall_mm"], d["size_km2"], d["infrastructure"]]
                      for d in data])
labels = np.array([d["damage_level"] for d in data])

# Normalize features
feature_means = features.mean(axis=0)
feature_stds = features.std(axis=0)
X_normalized = (features - feature_means) / feature_stds

# Split
split_idx = int(0.8 * len(X_normalized))
X_train, X_test = X_normalized[:split_idx], X_normalized[split_idx:]
y_train, y_test = labels[:split_idx], labels[split_idx:]

print(f"Training set: {len(X_train)} samples")
print(f"Test set: {len(X_test)} samples")
print(f"Damage levels: Low={sum(labels==0)}, Medium={sum(labels==1)}, High={sum(labels==2)}")


# ============================================================
# STEP 2: Build Your Classifier
# ============================================================

print("\n" + "=" * 55)
print("STEP 2: BUILD YOUR CLASSIFIER")
print("=" * 55)

# TODO 1: Import scikit-learn and create a classifier
# =====================================================
# Try different classifiers and compare:
# - from sklearn.tree import DecisionTreeClassifier
# - from sklearn.ensemble import RandomForestClassifier
# - from sklearn.neighbors import KNeighborsClassifier
# - from sklearn.svm import SVC
#
# Example:
#   from sklearn.ensemble import RandomForestClassifier
#   clf = RandomForestClassifier(n_estimators=100, random_state=42)

# UNCOMMENT AND COMPLETE:
# from sklearn._______ import _______
# clf = _______(_____)

# TODO 2: Train the classifier
# ==============================
# clf.fit(X_train, y_train)

# TODO 3: Make predictions
# =========================
# y_pred = clf.predict(X_test)

# TODO 4: Calculate accuracy
# ============================
# from sklearn.metrics import accuracy_score, classification_report
# accuracy = accuracy_score(y_test, y_pred)
# print(f"Accuracy: {accuracy:.1%}")
# print(classification_report(y_test, y_pred, target_names=["Low", "Medium", "High"]))


# ============================================================
# STEP 3: Feature Importance (BONUS)
# ============================================================

print("\n" + "=" * 55)
print("STEP 3: WHICH FACTORS MATTER MOST? (BONUS)")
print("=" * 55)

# TODO 5: If using RandomForest, show feature importance
# =======================================================
# feature_names = ["Category", "Wind Speed", "Distance", "Rainfall", "Island Size", "Infrastructure"]
# importances = clf.feature_importances_
# for name, imp in sorted(zip(feature_names, importances), key=lambda x: -x[1]):
#     bar = "█" * int(imp * 50)
#     print(f"  {name:15s} {imp:.3f} {bar}")


# ============================================================
# STEP 4: Predict for Specific Scenarios
# ============================================================

print("\n" + "=" * 55)
print("STEP 4: PREDICT REAL SCENARIOS")
print("=" * 55)

scenarios = [
    {"desc": "Category 5 hits Dominica directly", "features": [5, 165, 10, 400, 751, 0.45]},
    {"desc": "Category 1 passes 150km from Jamaica", "features": [1, 75, 150, 120, 10991, 0.65]},
    {"desc": "Category 3 hits Bahamas", "features": [3, 120, 30, 250, 13878, 0.68]},
    {"desc": "Category 4 hits Haiti directly", "features": [4, 145, 5, 380, 27750, 0.30]},
    {"desc": "Category 2 passes near Cayman Islands", "features": [2, 100, 80, 200, 264, 0.80]},
]

# TODO 6: Predict each scenario
# ===============================
# for scenario in scenarios:
#     features_normalized = (np.array(scenario["features"]) - feature_means) / feature_stds
#     prediction = clf.predict([features_normalized])[0]
#     damage_labels = {0: "LOW", 1: "MEDIUM", 2: "HIGH"}
#     print(f"  {scenario['desc']}")
#     print(f"    → Predicted damage: {damage_labels[prediction]}\n")

print("""
HINT: When you complete all TODOs, you'll have a working hurricane
impact predictor! Try different classifiers and see which works best.

Think about: Why does infrastructure matter so much?
Haiti and Dominica often suffer worse damage not because of bigger
storms, but because of lower infrastructure scores.

This kind of model could help Caribbean disaster preparedness agencies
allocate resources BEFORE a hurricane hits!
""")
