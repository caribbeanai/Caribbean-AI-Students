"""
Supervised Learning Examples - Caribbean AI Curriculum
Designed by Adrian Dunkley (Adriandunkley.net) | FREE

All examples use synthetic Caribbean data and are fully runnable.
Each example demonstrates a different supervised learning application
relevant to the Caribbean region.

Requirements:
    pip install numpy pandas scikit-learn matplotlib
"""

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import (
    classification_report, confusion_matrix, accuracy_score,
    mean_squared_error, mean_absolute_error, r2_score
)
import warnings
warnings.filterwarnings('ignore')

# ============================================================================
# EXAMPLE 1: Dengue Prediction from Weather Data (Trinidad & Tobago)
# ============================================================================

def dengue_prediction_trinidad():
    """
    Predict dengue outbreaks in Trinidad & Tobago based on weather data.

    Caribbean Context:
        Dengue fever is a major public health concern across the Caribbean.
        Trinidad & Tobago, like many Caribbean nations, experiences seasonal
        dengue outbreaks linked to rainfall, temperature, and humidity —
        conditions that favour the Aedes aegypti mosquito.

    Features:
        - Monthly rainfall (mm)
        - Average temperature (°C)
        - Humidity (%)
        - Previous month's dengue cases
        - Month of year (seasonality)
        - Standing water index (proxy for breeding sites)

    Target:
        - Outbreak (1) or No Outbreak (0) — binary classification
    """
    print("=" * 70)
    print("EXAMPLE 1: Dengue Outbreak Prediction — Trinidad & Tobago")
    print("=" * 70)

    np.random.seed(42)
    n_samples = 500

    # Generate synthetic weather data for Trinidad
    months = np.random.randint(1, 13, n_samples)

    # Rainfall higher in wet season (June-December)
    base_rainfall = np.where(
        (months >= 6) & (months <= 12),
        np.random.normal(250, 60, n_samples),   # Wet season
        np.random.normal(80, 30, n_samples)      # Dry season
    )
    rainfall = np.clip(base_rainfall, 0, 500)

    # Temperature fairly consistent in Trinidad (tropical)
    temperature = np.random.normal(28, 2, n_samples)

    # Humidity correlates with rainfall
    humidity = np.clip(60 + 0.08 * rainfall + np.random.normal(0, 5, n_samples), 50, 100)

    # Previous month's cases
    prev_cases = np.random.poisson(15, n_samples)

    # Standing water index
    standing_water = np.clip(0.3 * rainfall / 100 + np.random.normal(0, 0.2, n_samples), 0, 1)

    # Generate outbreak labels — more likely with high rain, humidity, temp
    outbreak_prob = (
        0.2 * (rainfall / 500) +
        0.2 * (humidity / 100) +
        0.15 * ((temperature - 24) / 8) +
        0.25 * (prev_cases / 50) +
        0.2 * standing_water
    )
    outbreak_prob = np.clip(outbreak_prob, 0.05, 0.95)
    outbreak = (np.random.random(n_samples) < outbreak_prob).astype(int)

    # Create DataFrame
    df = pd.DataFrame({
        'month': months,
        'rainfall_mm': np.round(rainfall, 1),
        'temperature_c': np.round(temperature, 1),
        'humidity_pct': np.round(humidity, 1),
        'prev_month_cases': prev_cases,
        'standing_water_index': np.round(standing_water, 3),
        'outbreak': outbreak
    })

    print(f"\nDataset shape: {df.shape}")
    print(f"Outbreak distribution:\n{df['outbreak'].value_counts()}")
    print(f"\nSample data:\n{df.head()}")

    # Prepare features and target
    X = df.drop('outbreak', axis=1)
    y = df['outbreak']

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # Scale features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Train Random Forest
    rf = RandomForestClassifier(n_estimators=100, random_state=42)
    rf.fit(X_train_scaled, y_train)

    y_pred = rf.predict(X_test_scaled)

    print(f"\n--- Random Forest Results ---")
    print(f"Accuracy: {accuracy_score(y_test, y_pred):.3f}")
    print(f"\nClassification Report:\n{classification_report(y_test, y_pred, target_names=['No Outbreak', 'Outbreak'])}")

    # Feature importance — what drives dengue outbreaks?
    feature_importance = pd.DataFrame({
        'feature': X.columns,
        'importance': rf.feature_importances_
    }).sort_values('importance', ascending=False)

    print("Feature Importance (what drives dengue outbreaks in Trinidad):")
    for _, row in feature_importance.iterrows():
        bar = "█" * int(row['importance'] * 50)
        print(f"  {row['feature']:25s} {row['importance']:.3f} {bar}")

    # Cross-validation
    cv_scores = cross_val_score(rf, scaler.transform(X), y, cv=5, scoring='accuracy')
    print(f"\n5-Fold Cross-Validation Accuracy: {cv_scores.mean():.3f} (+/- {cv_scores.std():.3f})")

    return rf, scaler


# ============================================================================
# EXAMPLE 2: Student Grade Prediction (Barbados School Data)
# ============================================================================

def student_grade_prediction_barbados():
    """
    Predict student grades in Barbados secondary schools.

    Caribbean Context:
        Education is highly valued across the Caribbean. In Barbados,
        students sit CXC (Caribbean Examinations Council) exams.
        Predicting which students might struggle allows early intervention.

    Features:
        - Study hours per week
        - Attendance rate (%)
        - Parent education level
        - Extra-curricular activities
        - Previous term grade
        - Distance to school (km)
        - Internet access at home

    Target:
        - Final exam score (0-100) — regression task
    """
    print("\n" + "=" * 70)
    print("EXAMPLE 2: Student Grade Prediction — Barbados Schools")
    print("=" * 70)

    np.random.seed(123)
    n_students = 600

    # Generate synthetic student data
    study_hours = np.clip(np.random.normal(12, 5, n_students), 0, 40)
    attendance = np.clip(np.random.normal(85, 10, n_students), 40, 100)

    # Parent education: 0=primary, 1=secondary, 2=tertiary
    parent_education = np.random.choice([0, 1, 2], n_students, p=[0.2, 0.5, 0.3])

    # Extra-curricular: number of activities (cricket, track, debate, etc.)
    extra_curricular = np.random.poisson(2, n_students)

    # Previous term grade
    prev_grade = np.clip(np.random.normal(65, 15, n_students), 20, 100)

    # Distance to school in km (Barbados is small — most < 15km)
    distance_km = np.clip(np.random.exponential(4, n_students), 0.5, 15)

    # Internet access (1=yes, 0=no)
    internet_access = np.random.choice([0, 1], n_students, p=[0.15, 0.85])

    # Generate final grade
    grade = (
        15 +
        0.8 * study_hours +
        0.3 * attendance +
        3.0 * parent_education +
        1.5 * extra_curricular +
        0.25 * prev_grade -
        0.5 * distance_km +
        4.0 * internet_access +
        np.random.normal(0, 5, n_students)
    )
    grade = np.clip(grade, 0, 100)

    df = pd.DataFrame({
        'study_hours_weekly': np.round(study_hours, 1),
        'attendance_pct': np.round(attendance, 1),
        'parent_education': parent_education,
        'extra_curricular_count': extra_curricular,
        'previous_grade': np.round(prev_grade, 1),
        'distance_to_school_km': np.round(distance_km, 1),
        'internet_access': internet_access,
        'final_grade': np.round(grade, 1)
    })

    print(f"\nDataset shape: {df.shape}")
    print(f"\nGrade statistics:\n{df['final_grade'].describe()}")
    print(f"\nSample data:\n{df.head()}")

    # Prepare features and target
    X = df.drop('final_grade', axis=1)
    y = df['final_grade']

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # Try multiple models
    models = {
        'Linear Regression': LinearRegression(),
        'Random Forest': RandomForestRegressor(n_estimators=100, random_state=42),
        'Gradient Boosting': GradientBoostingRegressor(n_estimators=100, random_state=42)
    }

    print("\n--- Model Comparison ---")
    print(f"{'Model':<25s} {'MAE':>8s} {'RMSE':>8s} {'R²':>8s}")
    print("-" * 51)

    best_model = None
    best_r2 = -float('inf')

    for name, model in models.items():
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)

        mae = mean_absolute_error(y_test, y_pred)
        rmse = np.sqrt(mean_squared_error(y_test, y_pred))
        r2 = r2_score(y_test, y_pred)

        print(f"{name:<25s} {mae:>8.2f} {rmse:>8.2f} {r2:>8.3f}")

        if r2 > best_r2:
            best_r2 = r2
            best_model = (name, model)

    print(f"\nBest model: {best_model[0]} (R² = {best_r2:.3f})")

    # Feature importance from the best tree-based model
    gb = models['Gradient Boosting']
    feature_importance = pd.DataFrame({
        'feature': X.columns,
        'importance': gb.feature_importances_
    }).sort_values('importance', ascending=False)

    print("\nWhat matters most for Barbados student grades:")
    for _, row in feature_importance.iterrows():
        bar = "█" * int(row['importance'] * 50)
        print(f"  {row['feature']:25s} {row['importance']:.3f} {bar}")

    # Predict for a hypothetical student
    print("\n--- Prediction for a Hypothetical Barbados Student ---")
    new_student = pd.DataFrame({
        'study_hours_weekly': [15],
        'attendance_pct': [92],
        'parent_education': [2],
        'extra_curricular_count': [3],
        'previous_grade': [72],
        'distance_to_school_km': [3.5],
        'internet_access': [1]
    })
    predicted_grade = gb.predict(new_student)[0]
    print(f"  Study hours: 15/week, Attendance: 92%, Parent edu: Tertiary")
    print(f"  Extra-curricular: 3 activities, Previous grade: 72")
    print(f"  Distance: 3.5km, Internet: Yes")
    print(f"  Predicted final grade: {predicted_grade:.1f}")

    return gb


# ============================================================================
# EXAMPLE 3: Fish Catch Prediction (The Bahamas)
# ============================================================================

def fish_catch_prediction_bahamas():
    """
    Predict daily fish catch volume for Bahamian fishermen.

    Caribbean Context:
        Fishing is a vital industry in The Bahamas, supporting livelihoods
        and food security across the archipelago. Predicting catch volumes
        helps fishermen plan trips and markets manage supply.

    Features:
        - Sea surface temperature (°C)
        - Moon phase (0-1, full moon = 1)
        - Wind speed (knots)
        - Season (1-4)
        - Fishing zone (Nassau, Andros, Exuma, Abaco, Eleuthera)
        - Boat size (small, medium, large)
        - Hours at sea

    Target:
        - Fish species caught (Snapper/Grouper/Mahi-Mahi/Lobster) — multi-class
    """
    print("\n" + "=" * 70)
    print("EXAMPLE 3: Fish Catch Classification — The Bahamas")
    print("=" * 70)

    np.random.seed(456)
    n_samples = 800

    # Generate synthetic fishing data
    sea_temp = np.clip(np.random.normal(27, 2, n_samples), 22, 32)
    moon_phase = np.random.uniform(0, 1, n_samples)
    wind_speed = np.clip(np.random.exponential(8, n_samples), 0, 35)
    season = np.random.choice([1, 2, 3, 4], n_samples)  # 1=Winter, 2=Spring, 3=Summer, 4=Fall

    zones = ['Nassau', 'Andros', 'Exuma', 'Abaco', 'Eleuthera']
    zone = np.random.choice(zones, n_samples)

    boat_sizes = ['small', 'medium', 'large']
    boat_size = np.random.choice(boat_sizes, n_samples, p=[0.4, 0.4, 0.2])

    hours_at_sea = np.clip(np.random.normal(8, 3, n_samples), 2, 16)

    # Determine primary catch based on conditions
    species_list = []
    for i in range(n_samples):
        probs = [0.25, 0.25, 0.25, 0.25]  # Base: Snapper, Grouper, Mahi-Mahi, Lobster

        # Snapper: prefer cooler water, reef zones
        if sea_temp[i] < 27:
            probs[0] += 0.15
        if zone[i] in ['Andros', 'Exuma']:
            probs[0] += 0.1

        # Grouper: prefer deeper water, larger boats
        if boat_size[i] == 'large':
            probs[1] += 0.15
        if zone[i] in ['Nassau', 'Abaco']:
            probs[1] += 0.1

        # Mahi-Mahi: pelagic, prefer warm water, open ocean
        if sea_temp[i] > 28:
            probs[2] += 0.2
        if hours_at_sea[i] > 10:
            probs[2] += 0.1

        # Lobster: prefer full moon, shallow reef
        if moon_phase[i] > 0.7:
            probs[3] += 0.15
        if season[i] in [3, 4]:  # Summer/Fall lobster season
            probs[3] += 0.15

        probs = np.array(probs)
        probs /= probs.sum()
        species_list.append(np.random.choice(
            ['Snapper', 'Grouper', 'Mahi-Mahi', 'Lobster'], p=probs
        ))

    # Encode categorical features
    le_zone = LabelEncoder()
    le_boat = LabelEncoder()

    df = pd.DataFrame({
        'sea_temp_c': np.round(sea_temp, 1),
        'moon_phase': np.round(moon_phase, 2),
        'wind_speed_knots': np.round(wind_speed, 1),
        'season': season,
        'zone': zone,
        'boat_size': boat_size,
        'hours_at_sea': np.round(hours_at_sea, 1),
        'primary_catch': species_list
    })

    print(f"\nDataset shape: {df.shape}")
    print(f"\nCatch distribution:\n{df['primary_catch'].value_counts()}")
    print(f"\nSample data:\n{df.head()}")

    # Encode categoricals
    df_encoded = df.copy()
    df_encoded['zone'] = le_zone.fit_transform(df['zone'])
    df_encoded['boat_size'] = le_boat.fit_transform(df['boat_size'])

    X = df_encoded.drop('primary_catch', axis=1)
    y = df_encoded['primary_catch']

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # Train classifier
    rf = RandomForestClassifier(n_estimators=150, random_state=42)
    rf.fit(X_train, y_train)
    y_pred = rf.predict(X_test)

    print(f"\n--- Random Forest Multi-Class Results ---")
    print(f"Accuracy: {accuracy_score(y_test, y_pred):.3f}")
    print(f"\nClassification Report:\n{classification_report(y_test, y_pred)}")

    # Confusion matrix
    print("Confusion Matrix:")
    cm = confusion_matrix(y_test, y_pred, labels=['Snapper', 'Grouper', 'Mahi-Mahi', 'Lobster'])
    species = ['Snapper', 'Grouper', 'Mahi-Mahi', 'Lobster']
    print(f"{'':>12s} {'Snapper':>10s} {'Grouper':>10s} {'Mahi-Mahi':>10s} {'Lobster':>10s}")
    for i, sp in enumerate(species):
        print(f"{sp:>12s} {cm[i][0]:>10d} {cm[i][1]:>10d} {cm[i][2]:>10d} {cm[i][3]:>10d}")

    # Feature importance
    feature_importance = pd.DataFrame({
        'feature': X.columns,
        'importance': rf.feature_importances_
    }).sort_values('importance', ascending=False)

    print("\nWhat determines the catch in Bahamian waters:")
    for _, row in feature_importance.iterrows():
        bar = "█" * int(row['importance'] * 50)
        print(f"  {row['feature']:25s} {row['importance']:.3f} {bar}")

    return rf


# ============================================================================
# EXAMPLE 4: Crop Price Forecasting (Jamaica)
# ============================================================================

def crop_price_forecasting_jamaica():
    """
    Forecast crop prices in Jamaican markets.

    Caribbean Context:
        Agriculture is a key sector of Jamaica's economy. Crops like yam,
        banana, scotch bonnet pepper, and ackee are staples. Price volatility
        affects farmers' income and food security. Predicting prices helps
        farmers decide what to plant and when to sell.

    Features:
        - Month (seasonality)
        - Rainfall previous month (mm)
        - Supply volume (tonnes available)
        - Demand index (tourism arrivals proxy)
        - Fuel price (affects transport cost)
        - Import competition index
        - Crop type (Yam, Banana, Scotch Bonnet, Ackee)

    Target:
        - Price per kg (JMD) — regression task
    """
    print("\n" + "=" * 70)
    print("EXAMPLE 4: Crop Price Forecasting — Jamaica")
    print("=" * 70)

    np.random.seed(789)
    n_samples = 1000

    crops = ['Yam', 'Banana', 'Scotch Bonnet', 'Ackee']
    base_prices = {'Yam': 180, 'Banana': 120, 'Scotch Bonnet': 350, 'Ackee': 500}

    data_rows = []
    for _ in range(n_samples):
        crop = np.random.choice(crops)
        month = np.random.randint(1, 13)

        # Rainfall varies by month (wet season June-Nov)
        if 6 <= month <= 11:
            rainfall = np.random.normal(200, 50)
        else:
            rainfall = np.random.normal(60, 25)
        rainfall = max(0, rainfall)

        # Supply varies — lower supply = higher price
        supply = max(10, np.random.normal(500, 150))

        # Demand index (tourism proxy — peaks Dec-Apr)
        if month in [12, 1, 2, 3, 4]:
            demand = np.random.normal(80, 10)
        else:
            demand = np.random.normal(55, 10)
        demand = np.clip(demand, 20, 100)

        # Fuel price (JMD per litre)
        fuel_price = np.random.normal(180, 20)

        # Import competition
        import_index = np.random.uniform(0, 1)

        # Calculate price
        base = base_prices[crop]
        price = (
            base +
            -0.05 * supply +               # More supply, lower price
            1.5 * demand +                   # More demand, higher price
            0.3 * fuel_price +               # Higher fuel, higher price
            -80 * import_index +             # More imports, lower local price
            0.05 * rainfall +                # Moderate rainfall helps
            np.random.normal(0, 20)          # Random noise
        )
        price = max(50, price)

        data_rows.append({
            'crop': crop,
            'month': month,
            'rainfall_mm': round(rainfall, 1),
            'supply_tonnes': round(supply, 1),
            'demand_index': round(demand, 1),
            'fuel_price_jmd': round(fuel_price, 1),
            'import_competition': round(import_index, 3),
            'price_per_kg_jmd': round(price, 2)
        })

    df = pd.DataFrame(data_rows)

    print(f"\nDataset shape: {df.shape}")
    print(f"\nPrice statistics by crop:")
    print(df.groupby('crop')['price_per_kg_jmd'].describe().round(1))
    print(f"\nSample data:\n{df.head()}")

    # Encode crop type
    df_encoded = pd.get_dummies(df, columns=['crop'], prefix='crop')

    X = df_encoded.drop('price_per_kg_jmd', axis=1)
    y = df_encoded['price_per_kg_jmd']

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # Train Gradient Boosting Regressor
    gb = GradientBoostingRegressor(
        n_estimators=200,
        max_depth=5,
        learning_rate=0.1,
        random_state=42
    )
    gb.fit(X_train, y_train)
    y_pred = gb.predict(X_test)

    mae = mean_absolute_error(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    r2 = r2_score(y_test, y_pred)

    print(f"\n--- Gradient Boosting Results ---")
    print(f"MAE:  {mae:.2f} JMD (average error in price prediction)")
    print(f"RMSE: {rmse:.2f} JMD")
    print(f"R²:   {r2:.3f} (explains {r2*100:.1f}% of price variance)")

    # Feature importance
    feature_importance = pd.DataFrame({
        'feature': X.columns,
        'importance': gb.feature_importances_
    }).sort_values('importance', ascending=False).head(10)

    print("\nTop factors affecting Jamaican crop prices:")
    for _, row in feature_importance.iterrows():
        bar = "█" * int(row['importance'] * 50)
        print(f"  {row['feature']:25s} {row['importance']:.3f} {bar}")

    # Forecast scenario
    print("\n--- Price Forecast Scenario ---")
    print("  Scenario: December (peak tourism), low supply, high fuel prices")
    print("  What will Scotch Bonnet peppers cost?")

    # Create scenario (need all one-hot columns)
    scenario = pd.DataFrame(0, index=[0], columns=X.columns)
    scenario['month'] = 12
    scenario['rainfall_mm'] = 50
    scenario['supply_tonnes'] = 300
    scenario['demand_index'] = 85
    scenario['fuel_price_jmd'] = 210
    scenario['import_competition'] = 0.2
    scenario['crop_Scotch Bonnet'] = 1

    predicted_price = gb.predict(scenario)[0]
    print(f"  Predicted price: {predicted_price:.2f} JMD per kg")

    # Compare with low-season scenario
    scenario_low = scenario.copy()
    scenario_low['month'] = 6
    scenario_low['demand_index'] = 50
    scenario_low['supply_tonnes'] = 700
    scenario_low['fuel_price_jmd'] = 160
    predicted_low = gb.predict(scenario_low)[0]
    print(f"\n  Low season (June) comparison: {predicted_low:.2f} JMD per kg")
    print(f"  Seasonal price difference: {predicted_price - predicted_low:.2f} JMD per kg")

    return gb


# ============================================================================
# MAIN — Run All Examples
# ============================================================================

def main():
    """Run all Caribbean supervised learning examples."""
    print("╔══════════════════════════════════════════════════════════════════════╗")
    print("║     Caribbean AI Curriculum — Supervised Learning Examples          ║")
    print("║     Designed by Adrian Dunkley (Adriandunkley.net) | FREE          ║")
    print("╚══════════════════════════════════════════════════════════════════════╝")
    print()
    print("We have four examples featuring real Caribbean scenarios:")
    print("  1. Dengue Prediction (Trinidad & Tobago)")
    print("  2. Student Grade Prediction (Barbados)")
    print("  3. Fish Catch Classification (The Bahamas)")
    print("  4. Crop Price Forecasting (Jamaica)")
    print()

    # Run each example
    dengue_prediction_trinidad()
    student_grade_prediction_barbados()
    fish_catch_prediction_bahamas()
    crop_price_forecasting_jamaica()

    print("\n" + "=" * 70)
    print("All examples complete! Big up to all Caribbean data scientists!")
    print("Try modifying the examples — change parameters, try new models,")
    print("or add yuh own Caribbean data. The possibilities are endless!")
    print("=" * 70)


if __name__ == "__main__":
    main()
