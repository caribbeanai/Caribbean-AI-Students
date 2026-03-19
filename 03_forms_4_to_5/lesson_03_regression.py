"""
================================================================================
🌳 FORMS 4-5 | LESSON 03: REGRESSION
================================================================================
Caribbean AI Academy — Designed by Adrian Dunkley (https://Adriandunkley.net)
FREE for Caribbean students. No paywall. No gatekeeping.
================================================================================

"Yuh learn fi classify tings into groups. Now it's time fi predict
NUMBERS. How much sugarcane Jamaica going produce? How much rain
Guyana going get? Regression answer THOSE questions."

In dis lesson yuh going learn:
  1. Linear Regression — fitting a straight line through data
  2. Polynomial Regression — when di relationship curved
  3. Apply it to Caribbean agriculture: sugarcane yield prediction
  4. Evaluate with R², MSE, MAE

Prerequisites: Lessons 01-02
Install: pip install numpy pandas scikit-learn matplotlib
================================================================================
"""

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')

print("=" * 70)
print("🌳 LESSON 03: REGRESSION")
print("   'Predicting NUMBERS — how much, how many, how far'")
print("=" * 70)

# ============================================================================
# PART 1: WHAT IS REGRESSION?
# ============================================================================

print("""
╔══════════════════════════════════════════════════════════════════════╗
║  PART 1: WHAT IS REGRESSION?                                      ║
╚══════════════════════════════════════════════════════════════════════╝

In Classification (Lesson 01), we predict CATEGORIES:
  - "Will hurricane hit?" → YES or NO

In Regression, we predict NUMBERS:
  - "How many tonnes of sugarcane wi go harvest?" → 45,200 tonnes
  - "What will sea temperature be next month?" → 29.3°C
  - "How many tourists visiting Jamaica in December?" → 385,000

Think of it like dis:
  - Classification = "Which box it go in?"
  - Regression = "How much / how many / how far?"

Di simplest form: LINEAR REGRESSION
  Draw di BEST straight line through yuh data points.
  Use dat line fi predict new values.
""")

# ============================================================================
# PART 2: CARIBBEAN SUGARCANE DATA
# ============================================================================

print("""
╔══════════════════════════════════════════════════════════════════════╗
║  PART 2: SUGARCANE YIELD DATA — JAMAICA, GUYANA, BARBADOS,        ║
║          TRINIDAD                                                   ║
╚══════════════════════════════════════════════════════════════════════╝

Sugarcane is one of di most important crops in Caribbean history.
Guyana, Jamaica, Barbados, and Trinidad all grow sugarcane.

Di yield (how much yuh harvest) depend heavily on RAINFALL.
Too little rain? Cane dry up. Too much? Roots rot.
There's a sweet spot!

Let we create realistic data fi each country.
""")

np.random.seed(42)

def generate_sugarcane_data(country, n=50, rain_range=(800, 2500),
                             base_yield=35, optimal_rain=1600,
                             noise_std=4):
    """Generate synthetic sugarcane yield data for a Caribbean country."""
    rainfall = np.random.uniform(rain_range[0], rain_range[1], n)
    # Yield follows a curve — peaks at optimal rainfall, drops if too wet or dry
    # Quadratic relationship: yield = a - b*(rainfall - optimal)^2
    normalized_diff = (rainfall - optimal_rain) / 500
    yield_tonnes = base_yield - 8 * normalized_diff**2 + np.random.normal(0, noise_std, n)
    yield_tonnes = np.clip(yield_tonnes, 5, 60)  # realistic bounds
    return rainfall, yield_tonnes

# Generate data for 4 countries
countries_config = {
    'Jamaica':    {'n': 60, 'rain_range': (900, 2200), 'base_yield': 38,
                   'optimal_rain': 1500, 'noise_std': 3.5},
    'Guyana':     {'n': 60, 'rain_range': (1200, 2800), 'base_yield': 42,
                   'optimal_rain': 1800, 'noise_std': 4.0},
    'Barbados':   {'n': 50, 'rain_range': (800, 1800), 'base_yield': 34,
                   'optimal_rain': 1300, 'noise_std': 3.0},
    'Trinidad':   {'n': 50, 'rain_range': (1000, 2400), 'base_yield': 36,
                   'optimal_rain': 1600, 'noise_std': 3.5},
}

all_data = []
for country, cfg in countries_config.items():
    rain, yield_t = generate_sugarcane_data(country, **cfg)
    for r, y in zip(rain, yield_t):
        all_data.append({'country': country, 'rainfall_mm': round(r, 1),
                         'yield_tonnes_per_ha': round(y, 2)})

cane_df = pd.DataFrame(all_data)

print("🌾 Sugarcane Yield Dataset Preview:")
print("-" * 50)
print(cane_df.head(10).to_string(index=False))
print(f"\nTotal samples: {len(cane_df)}")
print(f"\nSummary by Country:")
summary = cane_df.groupby('country').agg(
    avg_rainfall=('rainfall_mm', 'mean'),
    avg_yield=('yield_tonnes_per_ha', 'mean'),
    samples=('yield_tonnes_per_ha', 'count')
).round(1)
print(summary.to_string())

# ============================================================================
# PART 3: LINEAR REGRESSION — JAMAICA
# ============================================================================

print("""
╔══════════════════════════════════════════════════════════════════════╗
║  PART 3: LINEAR REGRESSION — JAMAICA SUGARCANE                     ║
╚══════════════════════════════════════════════════════════════════════╝

Let we start simple. Fit a STRAIGHT LINE to Jamaica's data.

Linear Regression find di line: y = mx + b
  - m = slope (how much yield change per mm of rain)
  - b = intercept (baseline yield)
""")

jamaica = cane_df[cane_df['country'] == 'Jamaica']
X_ja = jamaica[['rainfall_mm']].values
y_ja = jamaica['yield_tonnes_per_ha'].values

X_ja_train, X_ja_test, y_ja_train, y_ja_test = train_test_split(
    X_ja, y_ja, test_size=0.2, random_state=42
)

# Fit linear regression
lr_model = LinearRegression()
lr_model.fit(X_ja_train, y_ja_train)

# Predict
y_ja_pred = lr_model.predict(X_ja_test)

# Evaluate
lr_r2 = r2_score(y_ja_test, y_ja_pred)
lr_mse = mean_squared_error(y_ja_test, y_ja_pred)
lr_mae = mean_absolute_error(y_ja_test, y_ja_pred)

print(f"📈 Jamaica Linear Regression Results:")
print(f"   Equation: yield = {lr_model.coef_[0]:.4f} * rainfall + {lr_model.intercept_:.2f}")
print(f"   Slope:     {lr_model.coef_[0]:.4f} (tonnes per mm of rain)")
print(f"   Intercept: {lr_model.intercept_:.2f}")
print(f"\n   Evaluation Metrics:")
print(f"   R² Score:            {lr_r2:.4f}  (1.0 = perfect, 0 = useless)")
print(f"   Mean Squared Error:  {lr_mse:.2f}")
print(f"   Mean Absolute Error: {lr_mae:.2f} tonnes/ha")

print("""
IMPORTANT NOTE:
   R² might be LOW for linear regression here. Why?
   Because sugarcane yield vs rainfall is NOT a straight line!
   It's a CURVE — yield peaks at optimal rainfall then drops.
   We need POLYNOMIAL regression!
""")

# ============================================================================
# PART 4: POLYNOMIAL REGRESSION — THE CURVE
# ============================================================================

print("""
╔══════════════════════════════════════════════════════════════════════╗
║  PART 4: POLYNOMIAL REGRESSION — WHEN DI LINE NAH ENOUGH          ║
╚══════════════════════════════════════════════════════════════════════╝

Real life nah always follow a straight line.

Sugarcane yield vs rainfall look like an UPSIDE-DOWN U:
  ⌒  ← yield peak at optimal rainfall

Too little rain: crop suffer from drought
Too much rain:   roots waterlog, nutrients wash away
Just right:      maximum yield!

Polynomial Regression fit a CURVE instead of a line:
  Degree 2: y = ax² + bx + c  (quadratic — U shape)
  Degree 3: y = ax³ + bx² + cx + d  (cubic — S shape)

Fi sugarcane, degree 2 (quadratic) make di most sense.
""")

# Polynomial Regression (degree 2)
poly = PolynomialFeatures(degree=2, include_bias=False)
X_ja_train_poly = poly.fit_transform(X_ja_train)
X_ja_test_poly = poly.transform(X_ja_test)

poly_model = LinearRegression()
poly_model.fit(X_ja_train_poly, y_ja_train)

y_ja_pred_poly = poly_model.predict(X_ja_test_poly)

poly_r2 = r2_score(y_ja_test, y_ja_pred_poly)
poly_mse = mean_squared_error(y_ja_test, y_ja_pred_poly)
poly_mae = mean_absolute_error(y_ja_test, y_ja_pred_poly)

print(f"📈 Jamaica Polynomial Regression (Degree 2) Results:")
print(f"   Coefficients: rain² = {poly_model.coef_[1]:.8f}, rain = {poly_model.coef_[0]:.4f}")
print(f"   Intercept:    {poly_model.intercept_:.2f}")
print(f"\n   Evaluation Metrics:")
print(f"   R² Score:            {poly_r2:.4f}")
print(f"   Mean Squared Error:  {poly_mse:.2f}")
print(f"   Mean Absolute Error: {poly_mae:.2f} tonnes/ha")

print(f"\n📊 Comparison:")
print(f"   {'Model':<25s} {'R²':>8s} {'MSE':>10s} {'MAE':>10s}")
print(f"   {'─'*53}")
print(f"   {'Linear Regression':<25s} {lr_r2:>8.4f} {lr_mse:>10.2f} {lr_mae:>10.2f}")
print(f"   {'Polynomial (degree 2)':<25s} {poly_r2:>8.4f} {poly_mse:>10.2f} {poly_mae:>10.2f}")

if poly_r2 > lr_r2:
    print(f"\n   Polynomial regression fit BETTER! Di curve capture di relationship better.")
else:
    print(f"\n   Interesting — linear did well here. Di data might be in a mostly linear range.")

# ============================================================================
# PART 5: ALL FOUR COUNTRIES
# ============================================================================

print("""
╔══════════════════════════════════════════════════════════════════════╗
║  PART 5: REGRESSION FOR ALL FOUR COUNTRIES                         ║
╚══════════════════════════════════════════════════════════════════════╝
""")

fig, axes = plt.subplots(2, 2, figsize=(14, 11))
axes = axes.flatten()
country_colors = {'Jamaica': '#2ecc71', 'Guyana': '#3498db',
                   'Barbados': '#e74c3c', 'Trinidad': '#f39c12'}

results_all = []

for idx, (country, color) in enumerate(country_colors.items()):
    subset = cane_df[cane_df['country'] == country]
    X_c = subset[['rainfall_mm']].values
    y_c = subset['yield_tonnes_per_ha'].values

    # Split
    X_tr, X_te, y_tr, y_te = train_test_split(X_c, y_c, test_size=0.2, random_state=42)

    # Polynomial regression degree 2
    poly_feat = PolynomialFeatures(degree=2, include_bias=False)
    X_tr_p = poly_feat.fit_transform(X_tr)
    X_te_p = poly_feat.transform(X_te)

    model = LinearRegression()
    model.fit(X_tr_p, y_tr)
    y_pred = model.predict(X_te_p)

    r2 = r2_score(y_te, y_pred)
    mae = mean_absolute_error(y_te, y_pred)

    results_all.append({'country': country, 'r2': r2, 'mae': mae})

    # Plot
    ax = axes[idx]
    ax.scatter(X_c, y_c, c=color, alpha=0.6, s=50, edgecolors='black', linewidth=0.3)

    # Plot regression curve
    X_plot = np.linspace(X_c.min(), X_c.max(), 200).reshape(-1, 1)
    X_plot_p = poly_feat.transform(X_plot)
    y_plot = model.predict(X_plot_p)
    ax.plot(X_plot, y_plot, 'k-', linewidth=2, label=f'Poly Fit (R²={r2:.2f})')

    ax.set_xlabel('Rainfall (mm)', fontsize=11)
    ax.set_ylabel('Yield (tonnes/ha)', fontsize=11)
    ax.set_title(f'{country} — Sugarcane Yield vs Rainfall', fontsize=13)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)

plt.suptitle('Caribbean Sugarcane Yield Prediction by Country',
             fontsize=15, fontweight='bold', y=1.02)
plt.tight_layout()
plt.savefig('/home/user/Caribbean-AI-Students/03_forms_4_to_5/sugarcane_regression.png',
            dpi=150, bbox_inches='tight')
print("📊 Plot saved to: sugarcane_regression.png")
plt.close()

print(f"\n📊 Results Summary — All Countries:")
print(f"   {'Country':<20s} {'R² Score':>10s} {'MAE (t/ha)':>12s}")
print(f"   {'─'*42}")
for r in results_all:
    print(f"   {r['country']:<20s} {r['r2']:>10.4f} {r['mae']:>12.2f}")

# ============================================================================
# PART 6: PREDICTING FUTURE YIELDS
# ============================================================================

print("""
╔══════════════════════════════════════════════════════════════════════╗
║  PART 6: MAKING PREDICTIONS — PLANNING FI NEXT SEASON              ║
╚══════════════════════════════════════════════════════════════════════╝

Di REAL value of regression: predicting FUTURE outcomes!
Farmers can use dis fi plan planting, government fi plan exports.
""")

# Use Jamaica model for predictions
print("🌾 Jamaica Sugarcane Yield Predictions:")
print(f"   {'Rainfall (mm)':>15s} {'Predicted Yield (t/ha)':>25s} {'Notes'}")
print(f"   {'─'*65}")

prediction_rains = [900, 1100, 1300, 1500, 1700, 1900, 2100]
notes = ['Dry year — drought risk', 'Below average', 'Getting better',
         'Near optimal!', 'Optimal range', 'Getting too wet', 'Very wet — flooding risk']

poly_full = PolynomialFeatures(degree=2, include_bias=False)
X_ja_full_poly = poly_full.fit_transform(X_ja)
model_full = LinearRegression()
model_full.fit(X_ja_full_poly, y_ja)

for rain, note in zip(prediction_rains, notes):
    pred = model_full.predict(poly_full.transform([[rain]]))[0]
    bar = "█" * int(max(pred, 0) / 1.5)
    print(f"   {rain:>12d} mm {pred:>20.1f} t/ha  {bar}  {note}")

# ============================================================================
# PART 7: BONUS — USAIN BOLT SPRINT TIME PREDICTION
# ============================================================================

print("""
╔══════════════════════════════════════════════════════════════════════╗
║  BONUS: PREDICTING SPRINT TIMES — USAIN BOLT STYLE                ║
╚══════════════════════════════════════════════════════════════════════╝

Let we use regression fi predict 100m sprint times based on training!
""")

np.random.seed(77)
n_athletes = 200

# Features
training_hours = np.random.uniform(10, 40, n_athletes)
leg_strength = np.random.uniform(50, 100, n_athletes)     # strength score
reaction_time = np.random.uniform(0.12, 0.20, n_athletes)
age = np.random.uniform(18, 35, n_athletes)
wind = np.random.uniform(-2.0, 2.0, n_athletes)

# Sprint time (lower = better)
sprint_time = (
    12.0
    - training_hours * 0.03
    - leg_strength * 0.02
    + reaction_time * 5
    + np.abs(age - 26) * 0.02  # peak at age 26
    - wind * 0.05
    + np.random.normal(0, 0.2, n_athletes)
)

sprint_df = pd.DataFrame({
    'training_hours_weekly': np.round(training_hours, 1),
    'leg_strength_score': np.round(leg_strength, 1),
    'reaction_time_s': np.round(reaction_time, 3),
    'age': np.round(age, 1),
    'wind_speed_ms': np.round(wind, 2),
    'sprint_time_100m': np.round(sprint_time, 2)
})

X_sprint = sprint_df[['training_hours_weekly', 'leg_strength_score',
                        'reaction_time_s', 'age', 'wind_speed_ms']]
y_sprint = sprint_df['sprint_time_100m']

Xs_tr, Xs_te, ys_tr, ys_te = train_test_split(X_sprint, y_sprint, test_size=0.2, random_state=42)

sprint_model = LinearRegression()
sprint_model.fit(Xs_tr, ys_tr)
ys_pred = sprint_model.predict(Xs_te)

sprint_r2 = r2_score(ys_te, ys_pred)
sprint_mae = mean_absolute_error(ys_te, ys_pred)

print(f"🏃 Sprint Time Prediction Model:")
print(f"   R² Score: {sprint_r2:.4f}")
print(f"   MAE:      {sprint_mae:.3f} seconds")
print(f"\n   Feature Contributions (coefficients):")
for feat, coef in sorted(zip(X_sprint.columns, sprint_model.coef_),
                          key=lambda x: abs(x[1]), reverse=True):
    direction = "faster" if coef < 0 else "slower"
    print(f"   {feat:30s} {coef:>8.4f}  ({direction})")

# Simulate Bolt-like athlete
bolt_like = pd.DataFrame({
    'training_hours_weekly': [38],
    'leg_strength_score': [98],
    'reaction_time_s': [0.146],
    'age': [26],
    'wind_speed_ms': [0.9]
})
bolt_pred = sprint_model.predict(bolt_like)[0]
print(f"\n   🇯🇲 Bolt-like athlete prediction: {bolt_pred:.2f}s")
print(f"   (Usain Bolt's actual world record: 9.58s)")

# ============================================================================
# PART 8: KEY CONCEPTS SUMMARY
# ============================================================================

print("""
╔══════════════════════════════════════════════════════════════════════╗
║  KEY CONCEPTS SUMMARY                                              ║
╚══════════════════════════════════════════════════════════════════════╝

What yuh learn today:

1. REGRESSION = Predicting continuous numbers (not categories)
   - How much? How many? How far? How fast?

2. LINEAR REGRESSION = y = mx + b
   - Best straight line through di data
   - Good when relationship is roughly linear

3. POLYNOMIAL REGRESSION = y = ax² + bx + c (and higher)
   - Captures CURVED relationships
   - Sugarcane yield vs rainfall = perfect example

4. R² SCORE = How much variance yuh model explain
   - 1.0 = perfect fit, 0 = no better than guessing the mean

5. MSE / MAE = How far off yuh predictions are
   - Lower = better predictions

6. OVERFITTING WARNING:
   - Higher polynomial degree ≠ always better
   - Degree 10 might fit training data perfectly but fail on new data
   - Dis called OVERFITTING (same concept from Lesson 01!)

REAL WORLD: Caribbean agricultural ministries could use regression
to plan crop production, set export targets, and prepare fi drought!
""")

# ============================================================================
# QUIZ TIME!
# ============================================================================

print("""
╔══════════════════════════════════════════════════════════════════════╗
║  📝 QUIZ TIME! — 8 Questions                                      ║
╚══════════════════════════════════════════════════════════════════════╝

Answer these questions. Check yuh answers in quiz_answers.md

Q1: What is di difference between classification and regression?
    a) Classification is better than regression
    b) Classification predicts categories; regression predicts numbers
    c) Regression only works with big data
    d) There is no difference

Q2: In linear regression, what does di SLOPE tell yuh?
    a) Where the line starts
    b) How much the output changes for each unit change in input
    c) The accuracy of the model
    d) The number of data points

Q3: Why did polynomial regression work BETTER than linear for sugarcane?
    a) Polynomial is always better
    b) The relationship between rainfall and yield is curved (not linear)
    c) We had more data for polynomial
    d) Linear regression is outdated

Q4: What does R² = 0.85 mean?
    a) The model is 85% complete
    b) The model explains 85% of the variance in the data
    c) The model has 85 features
    d) 85% of the data is correct

Q5: If yuh increase polynomial degree from 2 to 20, what RISK yuh face?
    a) The model becomes too simple
    b) Overfitting — perfect on training data, terrible on new data
    c) The model runs slower but is always more accurate
    d) Nothing, higher degree is always better

Q6: A Jamaican farmer want fi know optimal rainfall fi sugarcane.
    Based on di polynomial model, approximately where is di peak?
    a) At the lowest rainfall
    b) At the highest rainfall
    c) At a middle range (around 1400-1600mm for Jamaica)
    d) Rainfall doesn't affect yield

Q7: MAE of 2.5 tonnes/ha means:
    a) The model is wrong 2.5% of the time
    b) On average, predictions are off by 2.5 tonnes per hectare
    c) The model needs 2.5 more features
    d) 2.5 tonnes were lost

Q8: Why is it important fi Caribbean countries to predict crop yields?
    a) It's not important
    b) For food security planning, export forecasting, and drought preparation
    c) Only for academic purposes
    d) Computers need something to do
""")

print("""
╔══════════════════════════════════════════════════════════════════════╗
║  🏆 BONUS CHALLENGE                                                ║
╚══════════════════════════════════════════════════════════════════════╝

Challenge 1: Add TEMPERATURE as a second feature alongside rainfall.
  Does a 2-feature model predict better?

Challenge 2: Try polynomial degrees 1 through 6. Plot R² for each.
  At what degree does overfitting start?

Challenge 3: Create data for Grenada nutmeg yield or Dominica banana yield.
  Fit regression models and compare to sugarcane.

Challenge 4: Predict tourism arrivals based on airfare price, season,
  and hotel availability for a Caribbean island of yuh choice.

Next lesson: Classification — multi-class Caribbean music genre classifier!

Walk good! 🌾
""")
