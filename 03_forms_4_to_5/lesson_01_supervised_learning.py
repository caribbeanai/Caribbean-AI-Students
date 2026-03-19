"""
================================================================================
🌳 FORMS 4-5 | LESSON 01: SUPERVISED LEARNING
================================================================================
Caribbean AI Academy — Designed by Adrian Dunkley (https://Adriandunkley.net)
FREE for Caribbean students. No paywall. No gatekeeping.
================================================================================

"Alright, big man and big woman ting now. Yuh ready fi teach di machine
how fi learn from EXAMPLES? Dat's supervised learning — yuh show it di
answer dem, and it figure out di pattern."

In dis lesson yuh going learn:
  1. What supervised learning is (and why it matter fi we)
  2. Decision Trees — how di machine make decisions like a flowchart
  3. Random Forests — many trees = better answers
  4. Apply it to REAL Caribbean problems: Hurricane prediction!

Prerequisites: Python basics, numpy, basic data understanding
Install: pip install numpy pandas scikit-learn matplotlib
================================================================================
"""

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import warnings
warnings.filterwarnings('ignore')

# 🧩 Puzzle Piece 4/12: Two digits - 05. Remember this number.

print("=" * 70)
print("🌳 LESSON 01: SUPERVISED LEARNING")
print("   'Teaching di machine fi learn from examples'")
print("=" * 70)

# ============================================================================
# PART 1: WHAT IS SUPERVISED LEARNING?
# ============================================================================

print("""
╔══════════════════════════════════════════════════════════════════════╗
║  PART 1: WHAT IS SUPERVISED LEARNING?                              ║
╚══════════════════════════════════════════════════════════════════════╝

Listen up. Supervised learning is like when yuh granny teach yuh fi cook.

She show yuh:
  "Dis is ackee and saltfish" → She tell yuh di NAME (di label)
  "Dis is curry goat"         → Again, she gi yuh di answer
  "Dis is rice and peas"      → Another example with di answer

After enough examples, yuh can IDENTIFY di food yuhself.
Even if it look likkle different from what Granny make.

Dat's EXACTLY what supervised learning do:
  - Yuh give di machine EXAMPLES with LABELS (di answers)
  - Di machine find PATTERNS in di data
  - Then it can PREDICT labels for NEW data it never see before

Two main types:
  1. CLASSIFICATION — predict a CATEGORY (hurricane hit: yes/no)
  2. REGRESSION — predict a NUMBER (how much rainfall tomorrow)

Today we focus pon CLASSIFICATION.
""")

# ============================================================================
# PART 2: CARIBBEAN HURRICANE DATA
# ============================================================================

print("""
╔══════════════════════════════════════════════════════════════════════╗
║  PART 2: CREATING CARIBBEAN HURRICANE DATA                         ║
╚══════════════════════════════════════════════════════════════════════╝

Every hurricane season, Caribbean people hold dem breath.
From June to November, we watching di weather channel nonstop.

Let we build a model dat predict WHETHER a hurricane going hit
a Caribbean island based on weather conditions.

Features (di INPUTS):
  - Sea surface temperature (°C) — warmer water = more energy fi storm
  - Wind shear (km/h) — high shear break up storms
  - Atmospheric pressure (hPa) — low pressure = stronger storm
  - Distance from island (km) — closer = more likely fi hit
  - Storm forward speed (km/h) — slower storms can change direction

Label (di OUTPUT):
  - Did it hit? 1 = YES, 0 = NO
""")

# Create realistic synthetic hurricane data
np.random.seed(42)  # fi reproducibility
n_samples = 500

# Generate features
sea_temp = np.random.uniform(25.0, 32.0, n_samples)        # °C
wind_shear = np.random.uniform(5.0, 45.0, n_samples)       # km/h
pressure = np.random.uniform(940.0, 1015.0, n_samples)     # hPa
distance = np.random.uniform(50.0, 800.0, n_samples)       # km from island
storm_speed = np.random.uniform(10.0, 50.0, n_samples)     # km/h forward speed

# Create realistic labels — hurricanes more likely to hit when:
# - sea temp is HIGH (>28°C), wind shear is LOW (<20), pressure LOW (<980)
# - distance is SMALL (<300km), storm speed MODERATE
hit_score = (
    (sea_temp - 25) * 2.5 +          # warmer water → more likely
    (30 - wind_shear) * 0.8 +         # less shear → more likely
    (1010 - pressure) * 0.15 +         # lower pressure → more likely
    (500 - distance) * 0.02 +          # closer → more likely
    np.random.normal(0, 3, n_samples)  # some randomness (nature unpredictable!)
)

hit = (hit_score > 12).astype(int)

# Build DataFrame
hurricane_data = pd.DataFrame({
    'sea_temp_celsius': np.round(sea_temp, 1),
    'wind_shear_kmh': np.round(wind_shear, 1),
    'pressure_hpa': np.round(pressure, 1),
    'distance_km': np.round(distance, 1),
    'storm_speed_kmh': np.round(storm_speed, 1),
    'hurricane_hit': hit
})

print("🌊 Hurricane Dataset Preview:")
print("-" * 60)
print(hurricane_data.head(10).to_string(index=False))
print(f"\nTotal samples: {len(hurricane_data)}")
print(f"Hurricanes dat HIT:     {hit.sum()} ({hit.mean()*100:.1f}%)")
print(f"Hurricanes dat MISSED:  {(1-hit).sum()} ({(1-hit.mean())*100:.1f}%)")

# ============================================================================
# PART 3: PREPARING DI DATA
# ============================================================================

print("""
╔══════════════════════════════════════════════════════════════════════╗
║  PART 3: PREPARING DI DATA — TRAIN/TEST SPLIT                     ║
╚══════════════════════════════════════════════════════════════════════╝

Before we train di model, we need fi split di data:

  TRAINING SET (80%) — Di machine learn from dis
  TESTING SET  (20%) — We test di machine on data it NEVER see before

Why? Same reason teacher nah gi yuh di exact exam questions fi practice.
Yuh need fi prove yuh UNDERSTAND, not just memorize!
""")

# Separate features (X) and label (y)
X = hurricane_data[['sea_temp_celsius', 'wind_shear_kmh', 'pressure_hpa',
                      'distance_km', 'storm_speed_kmh']]
y = hurricane_data['hurricane_hit']

# Split: 80% train, 20% test
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

print(f"Training samples: {len(X_train)}")
print(f"Testing samples:  {len(X_test)}")
print(f"\nTraining set — Hits: {y_train.sum()}, Misses: {(1-y_train).sum()}")
print(f"Testing set  — Hits: {y_test.sum()}, Misses: {(1-y_test).sum()}")

# ============================================================================
# PART 4: DECISION TREES
# ============================================================================

print("""
╔══════════════════════════════════════════════════════════════════════╗
║  PART 4: DECISION TREES — DI FLOWCHART MODEL                      ║
╚══════════════════════════════════════════════════════════════════════╝

A Decision Tree is like how yuh make decisions already:

  "Is di sea water hot hot? (> 29°C)"
      ├── YES → "Is di storm close? (< 200km)"
      │         ├── YES → "HURRICANE LIKELY! Board up di house!"
      │         └── NO  → "Watch it still, but probably safe"
      └── NO  → "Probably nah go hit we. But stay alert!"

Di machine build dis flowchart AUTOMATICALLY from di data.
Each split pick di BEST question fi separate hits from misses.
""")

# Train a Decision Tree
dt_model = DecisionTreeClassifier(
    max_depth=5,           # don't make it too deep (prevent overfitting)
    random_state=42,
    min_samples_leaf=10    # each leaf need at least 10 samples
)
dt_model.fit(X_train, y_train)

# Predict on test set
dt_predictions = dt_model.predict(X_test)
dt_accuracy = accuracy_score(y_test, dt_predictions)

print(f"🌳 Decision Tree Results:")
print(f"   Accuracy: {dt_accuracy*100:.1f}%")
print(f"\n   Classification Report:")
print(classification_report(y_test, dt_predictions,
                            target_names=['Miss', 'Hit']))

# Show feature importance
print("📊 Feature Importance (which features matter most?):")
for feat, imp in sorted(zip(X.columns, dt_model.feature_importances_),
                         key=lambda x: x[1], reverse=True):
    bar = "█" * int(imp * 40)
    print(f"   {feat:25s} {imp:.3f} {bar}")

# ============================================================================
# PART 5: RANDOM FORESTS
# ============================================================================

print("""
╔══════════════════════════════════════════════════════════════════════╗
║  PART 5: RANDOM FORESTS — MANY TREES BETTER THAN ONE              ║
╚══════════════════════════════════════════════════════════════════════╝

One tree good. But yuh know what BETTER? A whole FOREST.

Random Forest = many decision trees voting together.

Think of it like asking ONE person if hurricane coming vs asking
100 people and going with di majority opinion. Di crowd is usually
smarter than any one person!

How it work:
  1. Create 100 different decision trees
  2. Each tree see a RANDOM subset of di data
  3. Each tree see a RANDOM subset of di features
  4. All trees VOTE on di answer
  5. Majority win!

Dis reduce OVERFITTING (when di model memorize instead of learn).
""")

# Train Random Forest
rf_model = RandomForestClassifier(
    n_estimators=100,      # 100 trees in di forest
    max_depth=8,
    random_state=42,
    min_samples_leaf=5
)
rf_model.fit(X_train, y_train)

# Predict
rf_predictions = rf_model.predict(X_test)
rf_accuracy = accuracy_score(y_test, rf_predictions)

print(f"🌲🌲🌲 Random Forest Results:")
print(f"   Accuracy: {rf_accuracy*100:.1f}%")
print(f"\n   Classification Report:")
print(classification_report(y_test, rf_predictions,
                            target_names=['Miss', 'Hit']))

# Feature importance from Random Forest
print("📊 Random Forest Feature Importance:")
for feat, imp in sorted(zip(X.columns, rf_model.feature_importances_),
                         key=lambda x: x[1], reverse=True):
    bar = "█" * int(imp * 40)
    print(f"   {feat:25s} {imp:.3f} {bar}")

# ============================================================================
# PART 6: COMPARING DI MODELS
# ============================================================================

print("""
╔══════════════════════════════════════════════════════════════════════╗
║  PART 6: COMPARING DI MODELS                                      ║
╚══════════════════════════════════════════════════════════════════════╝
""")

print(f"📊 Model Comparison:")
print(f"   {'Model':<25s} {'Accuracy':>10s}")
print(f"   {'-'*35}")
print(f"   {'Decision Tree':<25s} {dt_accuracy*100:>9.1f}%")
print(f"   {'Random Forest':<25s} {rf_accuracy*100:>9.1f}%")

if rf_accuracy > dt_accuracy:
    print(f"\n   🏆 Random Forest win! Di crowd wisdom beat di single tree.")
elif dt_accuracy > rf_accuracy:
    print(f"\n   🏆 Decision Tree win dis time! Sometimes simpler is better.")
else:
    print(f"\n   🤝 It's a tie! Both model perform di same.")

# ============================================================================
# PART 7: MAKING PREDICTIONS ON NEW DATA
# ============================================================================

print("""
╔══════════════════════════════════════════════════════════════════════╗
║  PART 7: MAKING PREDICTIONS — REAL SCENARIOS                      ║
╚══════════════════════════════════════════════════════════════════════╝

Let we test di model with some realistic Caribbean scenarios!
""")

scenarios = pd.DataFrame({
    'sea_temp_celsius': [30.5, 26.0, 29.8, 27.5, 31.2],
    'wind_shear_kmh':   [8.0, 35.0, 12.0, 25.0, 6.0],
    'pressure_hpa':     [955.0, 1005.0, 960.0, 990.0, 945.0],
    'distance_km':      [120.0, 600.0, 200.0, 400.0, 80.0],
    'storm_speed_kmh':  [25.0, 40.0, 15.0, 30.0, 20.0]
})

scenario_names = [
    "Cat 4 heading fi Jamaica — hot water, low pressure, close",
    "Weak system far from Barbados — cool water, high shear",
    "Strong storm approaching Trinidad — warm water, low shear",
    "Moderate storm near St. Lucia — mixed signals",
    "Monster hurricane bearing down on Antigua — DANGER"
]

predictions = rf_model.predict(scenarios)
probabilities = rf_model.predict_proba(scenarios)

print("🌀 Hurricane Prediction Scenarios:")
print("=" * 70)
for i, (name, pred, prob) in enumerate(zip(scenario_names, predictions, probabilities)):
    result = "🔴 LIKELY HIT" if pred == 1 else "🟢 LIKELY MISS"
    confidence = max(prob) * 100
    print(f"\n  Scenario {i+1}: {name}")
    print(f"  Prediction: {result} (Confidence: {confidence:.0f}%)")
    print(f"  Hit probability: {prob[1]*100:.0f}% | Miss probability: {prob[0]*100:.0f}%")

# ============================================================================
# PART 8: CARIBBEAN SPORTS — CRICKET ANALYSIS
# ============================================================================

print("""
╔══════════════════════════════════════════════════════════════════════╗
║  BONUS: CRICKET MATCH OUTCOME PREDICTION                          ║
╚══════════════════════════════════════════════════════════════════════╝

Caribbean people LOVE cricket. West Indies cricket is in we blood!
Let we use supervised learning fi predict T20 match outcomes.
""")

# Synthetic cricket data
np.random.seed(99)
n_matches = 300

# Features for WI batting first
powerplay_runs = np.random.randint(30, 70, n_matches)       # runs in first 6 overs
run_rate = np.random.uniform(6.0, 12.0, n_matches)          # overall run rate
wickets_lost = np.random.randint(2, 10, n_matches)           # total wickets lost
boundaries = np.random.randint(8, 25, n_matches)             # 4s and 6s
extras = np.random.randint(2, 18, n_matches)                 # extras conceded

# Win more likely with high run rate, good powerplay, fewer wickets lost
win_score = (
    run_rate * 3 +
    powerplay_runs * 0.15 +
    boundaries * 0.4 -
    wickets_lost * 1.5 +
    np.random.normal(0, 3, n_matches)
)
wi_won = (win_score > 28).astype(int)

cricket_X = pd.DataFrame({
    'powerplay_runs': powerplay_runs,
    'run_rate': np.round(run_rate, 2),
    'wickets_lost': wickets_lost,
    'boundaries': boundaries,
    'extras': extras
})

# Train and evaluate
Xc_train, Xc_test, yc_train, yc_test = train_test_split(
    cricket_X, wi_won, test_size=0.2, random_state=42
)

cricket_rf = RandomForestClassifier(n_estimators=100, max_depth=6, random_state=42)
cricket_rf.fit(Xc_train, yc_train)
cricket_pred = cricket_rf.predict(Xc_test)
cricket_acc = accuracy_score(yc_test, cricket_pred)

print(f"🏏 West Indies T20 Match Outcome Predictor:")
print(f"   Accuracy: {cricket_acc*100:.1f}%")
print(f"\n   Feature Importance:")
for feat, imp in sorted(zip(cricket_X.columns, cricket_rf.feature_importances_),
                         key=lambda x: x[1], reverse=True):
    bar = "█" * int(imp * 40)
    print(f"   {feat:20s} {imp:.3f} {bar}")

# ============================================================================
# PART 9: USAIN BOLT SPRINT CLASSIFICATION
# ============================================================================

print("""
╔══════════════════════════════════════════════════════════════════════╗
║  BONUS: SPRINT PERFORMANCE CLASSIFICATION                          ║
╚══════════════════════════════════════════════════════════════════════╝

Jamaica produce some of di FASTEST humans on Earth! Usain Bolt,
Shelly-Ann Fraser-Pryce, Elaine Thompson-Herah — LEGENDS.

Let we classify sprint performances into medal categories.
""")

np.random.seed(77)
n_sprints = 400

# Features
reaction_time = np.random.uniform(0.120, 0.200, n_sprints)   # seconds
top_speed = np.random.uniform(9.5, 12.5, n_sprints)          # m/s
acceleration = np.random.uniform(3.5, 5.5, n_sprints)        # m/s²
wind_speed = np.random.uniform(-2.0, 2.0, n_sprints)         # m/s (tailwind positive)
training_hours = np.random.uniform(15, 40, n_sprints)         # weekly hours

# 0=No medal, 1=Bronze, 2=Silver, 3=Gold
performance = (
    top_speed * 2 + acceleration * 3 - reaction_time * 20 +
    wind_speed * 0.5 + training_hours * 0.1 +
    np.random.normal(0, 1.5, n_sprints)
)

medal = np.digitize(performance,
                     bins=[0, np.percentile(performance, 50),
                           np.percentile(performance, 75),
                           np.percentile(performance, 90)])
medal = np.clip(medal, 0, 3)

sprint_X = pd.DataFrame({
    'reaction_time': np.round(reaction_time, 3),
    'top_speed_ms': np.round(top_speed, 2),
    'acceleration': np.round(acceleration, 2),
    'wind_speed_ms': np.round(wind_speed, 2),
    'training_hours_weekly': np.round(training_hours, 1)
})

Xs_train, Xs_test, ys_train, ys_test = train_test_split(
    sprint_X, medal, test_size=0.2, random_state=42
)

sprint_rf = RandomForestClassifier(n_estimators=100, max_depth=8, random_state=42)
sprint_rf.fit(Xs_train, ys_train)
sprint_pred = sprint_rf.predict(Xs_test)
sprint_acc = accuracy_score(ys_test, sprint_pred)

medal_names = ['No Medal', 'Bronze 🥉', 'Silver 🥈', 'Gold 🥇']
print(f"🏃 Sprint Performance Classifier:")
print(f"   Accuracy: {sprint_acc*100:.1f}%")
print(f"\n   Classification Report:")
print(classification_report(ys_test, sprint_pred,
                            target_names=medal_names,
                            zero_division=0))

# ============================================================================
# PART 10: KEY CONCEPTS SUMMARY
# ============================================================================

print("""
╔══════════════════════════════════════════════════════════════════════╗
║  KEY CONCEPTS SUMMARY                                              ║
╚══════════════════════════════════════════════════════════════════════╝

What yuh learn today:

1. SUPERVISED LEARNING = Learning from labeled examples
   - Like Granny teaching yuh what each food is

2. DECISION TREE = A flowchart of yes/no questions
   - Simple, easy fi understand, but can overfit

3. RANDOM FOREST = Many decision trees voting together
   - More robust, harder fi fool, usually more accurate

4. TRAIN/TEST SPLIT = Never test on what yuh train on
   - Like a teacher using different questions for exam

5. FEATURE IMPORTANCE = Which inputs matter most
   - Sea temperature and pressure matter most for hurricanes!

6. CLASSIFICATION REPORT = Precision, Recall, F1-score
   - Different ways fi measure how good yuh model is

7. OVERFITTING = When di model memorize instead of learn
   - Like studying only past papers and failing new questions

REMEMBER: A model is only as good as di data yuh feed it!
""")

# ============================================================================
# QUIZ TIME!
# ============================================================================

print("""
╔══════════════════════════════════════════════════════════════════════╗
║  📝 QUIZ TIME! — 8 Questions                                      ║
╚══════════════════════════════════════════════════════════════════════╝

Answer these questions. Check yuh answers in quiz_answers.md

Q1: What is di MAIN difference between supervised and unsupervised learning?
    a) Supervised is faster
    b) Supervised uses labeled data, unsupervised doesn't
    c) Unsupervised is more accurate
    d) There is no difference

Q2: In we hurricane dataset, what is di LABEL (target variable)?
    a) Sea surface temperature
    b) Wind shear
    c) Whether the hurricane hit or not (1/0)
    d) Atmospheric pressure

Q3: Why do we split data into training and testing sets?
    a) To make the dataset smaller
    b) To test if the model can generalize to unseen data
    c) Because the computer can't handle all the data at once
    d) To make training faster

Q4: What is a Random Forest?
    a) A single large decision tree
    b) An ensemble of many decision trees that vote together
    c) A type of neural network
    d) A clustering algorithm

Q5: In we hurricane model, which feature would yuh EXPECT to be most
    important for predicting a hit?
    a) Storm forward speed
    b) Sea surface temperature and pressure
    c) The day of the week
    d) The name of the hurricane

Q6: What is OVERFITTING?
    a) When the model is too simple
    b) When the model memorizes training data but fails on new data
    c) When you have too much data
    d) When the model is very accurate

Q7: If a Decision Tree has accuracy of 85% and a Random Forest has 90%,
    which should yuh generally prefer and why?
    a) Decision Tree — simpler is always better
    b) Random Forest — higher accuracy and more robust
    c) Neither — always use neural networks
    d) It depends only on speed

Q8: A hurricane has sea temp 31°C, low wind shear (8 km/h), pressure
    950 hPa, and is 100km from yuh island. Based on what yuh learn,
    should yuh prepare fi impact?
    a) No, those conditions are normal
    b) Maybe, need more data
    c) Yes! High sea temp, low shear, low pressure, and close distance
       all indicate high probability of a hit
    d) Only if it's hurricane season
""")

# ============================================================================
# CHALLENGE
# ============================================================================

print("""
╔══════════════════════════════════════════════════════════════════════╗
║  🏆 BONUS CHALLENGE                                                ║
╚══════════════════════════════════════════════════════════════════════╝

Challenge 1: Add more features to the hurricane dataset
  (e.g., time of year, El Niño index, historical patterns)

Challenge 2: Try different max_depth values for the Decision Tree.
  What happen when yuh make it very deep (max_depth=20)?
  What happen when it very shallow (max_depth=2)?

Challenge 3: Modify the cricket analysis to include bowling stats.
  Can yuh predict match outcome better with MORE features?

Challenge 4: Add more Caribbean athletes to the sprint dataset.
  Include 200m and 400m events too!

Next lesson: Unsupervised Learning — finding hidden patterns
in Caribbean economic data WITHOUT labels!

Walk good! 🇯🇲🇹🇹🇧🇧🇬🇾🇧🇸🇦🇬🇱🇨🇩🇲🇬🇩🇰🇳🇻🇨🇭🇹🇩🇴🇨🇺🇧🇿🇸🇷
""")
