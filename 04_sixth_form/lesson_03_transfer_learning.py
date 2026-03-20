"""
=============================================================================
Caribbean AI Academy - Sixth Form (Ages 16-18)
Lesson 03: Transfer Learning
=============================================================================
Topic: Reusing Knowledge Across Caribbean AI Tasks

Yow! Imagine yuh learn to play cricket in Jamaica, then yuh move to
Trinidad and pick up football quick-quick because yuh already have
athleticism, hand-eye coordination, and game sense. Dat is transfer
learning!

In AI, transfer learning means: train a model on one task where yuh
have plenty data, then TRANSFER that knowledge to a new task where
data is scarce. This is CRITICAL for Caribbean AI because:
- We often have small datasets (small population islands)
- Collecting labelled data is expensive
- Models trained on global data can be adapted for local use

We demonstrate with scikit-learn: train on one Caribbean task, then
apply learned features/knowledge to another.

Requirements: pip install scikit-learn numpy
Author: Adrian Dunkley | Caribbean AI Academy
=============================================================================
"""

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report

print("=" * 65)
print("  LESSON 03: Transfer Learning")
print("  Reusing Knowledge Across Caribbean AI Tasks")
print("=" * 65)


# =====================================================================
# SECTION 1: What is Transfer Learning?
# =====================================================================
print("\n--- Part 1: Understanding Transfer Learning ---\n")
print("""
TRANSFER LEARNING is like a Jamaican sprinter becoming a bobsledder:
  - Skills transfer: speed, power, explosiveness
  - New skills needed: ice handling, sled control
  - But starting from ZERO would take much longer!

In AI, there are several strategies:

1. FEATURE EXTRACTION
   - Use a pre-trained model as a feature extractor
   - Like using a senior cricketer's game sense for coaching analysis

2. FINE-TUNING
   - Take a pre-trained model and retrain the last few layers
   - Like a Barbados netball player adjusting to TT court dimensions

3. DOMAIN ADAPTATION
   - Adapt a model from one domain (e.g., global weather) to another
     (e.g., Caribbean microclimates)

Why it matters for the Caribbean:
- ImageNet has millions of images but almost NONE from the Caribbean
- Medical datasets are mostly from USA/Europe, not our population
- Transfer learning bridges that gap!
""")


# =====================================================================
# SECTION 2: Source Task — Caribbean Economic Classification
# =====================================================================
print("--- Part 2: Source Task — Economic Sector Classification ---\n")

np.random.seed(42)


def generate_economic_data(n_samples=500):
    """
    Generate synthetic Caribbean economic indicator data.
    Task: Classify economic activity into sectors.

    Features: GDP contribution, employment rate, export value,
              seasonal variation, energy usage

    Sectors: Tourism, Agriculture, Manufacturing, Services
    Caribbean context: Tourism-heavy economies (Bahamas, Barbados),
    agriculture (Guyana, Belize), mixed (Jamaica, Trinidad)
    """
    X, y = [], []

    for _ in range(n_samples):
        sector = np.random.randint(0, 4)

        if sector == 0:  # Tourism (Bahamas, Barbados, Antigua style)
            gdp = np.random.normal(0.35, 0.08)
            employment = np.random.normal(0.40, 0.10)
            exports = np.random.normal(0.20, 0.05)
            seasonal = np.random.normal(0.8, 0.15)  # Very seasonal
            energy = np.random.normal(0.30, 0.08)
        elif sector == 1:  # Agriculture (Guyana, Belize, Suriname)
            gdp = np.random.normal(0.15, 0.05)
            employment = np.random.normal(0.30, 0.10)
            exports = np.random.normal(0.40, 0.10)
            seasonal = np.random.normal(0.6, 0.12)
            energy = np.random.normal(0.20, 0.06)
        elif sector == 2:  # Manufacturing (Trinidad, Jamaica)
            gdp = np.random.normal(0.20, 0.06)
            employment = np.random.normal(0.25, 0.08)
            exports = np.random.normal(0.35, 0.08)
            seasonal = np.random.normal(0.2, 0.10)  # Less seasonal
            energy = np.random.normal(0.60, 0.12)  # High energy
        else:  # Services/Finance (Cayman, BVI, Barbados)
            gdp = np.random.normal(0.30, 0.07)
            employment = np.random.normal(0.35, 0.09)
            exports = np.random.normal(0.15, 0.05)
            seasonal = np.random.normal(0.3, 0.10)
            energy = np.random.normal(0.25, 0.07)

        X.append([gdp, employment, exports, seasonal, energy])
        y.append(sector)

    return np.array(X), np.array(y)


SECTORS = ['Tourism', 'Agriculture', 'Manufacturing', 'Services']
X_source, y_source = generate_economic_data(n_samples=500)
print(f"Source dataset: {X_source.shape[0]} samples, {X_source.shape[1]} features")
print(f"Sectors: {SECTORS}")

# Train source model
scaler_source = StandardScaler()
X_source_scaled = scaler_source.fit_transform(X_source)
X_s_train, X_s_test, y_s_train, y_s_test = train_test_split(
    X_source_scaled, y_source, test_size=0.2, random_state=42
)

source_model = RandomForestClassifier(n_estimators=100, random_state=42)
source_model.fit(X_s_train, y_s_train)

source_acc = accuracy_score(y_s_test, source_model.predict(X_s_test))
print(f"Source model accuracy: {source_acc:.1%}")
print("(This model has learned general patterns about Caribbean economies)")


# =====================================================================
# SECTION 3: Target Task — Investment Risk Classification (SMALL DATA)
# =====================================================================
print("\n--- Part 3: Target Task — Investment Risk (Small Dataset) ---\n")


def generate_investment_data(n_samples=50):
    """
    Generate SMALL investment risk dataset — like what a Caribbean
    development bank might have. Only 50 samples!

    Features: Same economic indicators (GDP, employment, exports,
              seasonal, energy) — SIMILAR domain!
    Labels: Low Risk (0), Medium Risk (1), High Risk (2)
    """
    X, y = [], []
    for _ in range(n_samples):
        risk = np.random.randint(0, 3)

        if risk == 0:  # Low risk — stable economies
            gdp = np.random.normal(0.30, 0.06)
            employment = np.random.normal(0.35, 0.08)
            exports = np.random.normal(0.30, 0.07)
            seasonal = np.random.normal(0.3, 0.10)
            energy = np.random.normal(0.35, 0.08)
        elif risk == 1:  # Medium risk
            gdp = np.random.normal(0.20, 0.07)
            employment = np.random.normal(0.25, 0.10)
            exports = np.random.normal(0.20, 0.08)
            seasonal = np.random.normal(0.6, 0.15)
            energy = np.random.normal(0.40, 0.10)
        else:  # High risk — vulnerable economies
            gdp = np.random.normal(0.10, 0.05)
            employment = np.random.normal(0.15, 0.08)
            exports = np.random.normal(0.10, 0.06)
            seasonal = np.random.normal(0.8, 0.12)
            energy = np.random.normal(0.55, 0.15)

        X.append([gdp, employment, exports, seasonal, energy])
        y.append(risk)

    return np.array(X), np.array(y)


RISK_LEVELS = ['Low Risk', 'Medium Risk', 'High Risk']
X_target, y_target = generate_investment_data(n_samples=50)
print(f"Target dataset: ONLY {X_target.shape[0]} samples! (Small island reality)")
print(f"Risk levels: {RISK_LEVELS}")

X_target_scaled = scaler_source.transform(X_target)  # Use SOURCE scaler!
X_t_train, X_t_test, y_t_train, y_t_test = train_test_split(
    X_target_scaled, y_target, test_size=0.3, random_state=42
)


# =====================================================================
# SECTION 4: Baseline — Train from Scratch on Small Data
# =====================================================================
print("\n--- Part 4: Baseline — Training from Scratch ---\n")

baseline_model = RandomForestClassifier(n_estimators=50, random_state=42)
baseline_model.fit(X_t_train, y_t_train)
baseline_acc = accuracy_score(y_t_test, baseline_model.predict(X_t_test))
print(f"Baseline accuracy (trained from scratch on 35 samples): {baseline_acc:.1%}")
print("With so little data, the model struggles — just like trying to")
print("learn cricket by watching only 3 matches!\n")


# =====================================================================
# SECTION 5: Transfer Learning — Use Source Knowledge
# =====================================================================
print("--- Part 5: Transfer Learning Approach ---\n")

# Strategy: Use the source model's feature transformations
# The Random Forest learned which features matter for Caribbean economies.
# We extract those learned representations and use them for our new task.

# Method 1: Feature extraction using source model's leaf indices
# Each tree in the forest maps input to a leaf node — this IS a
# learned representation of the economic data!
source_leaves_train = source_model.apply(X_t_train)  # Leaf indices
source_leaves_test = source_model.apply(X_t_test)

print(f"Source model leaf features shape: {source_leaves_train.shape}")
print(f"(100 trees each assign a leaf index = 100 learned features)")

# Combine original features with transferred features
X_t_train_combined = np.hstack([X_t_train, source_leaves_train])
X_t_test_combined = np.hstack([X_t_test, source_leaves_test])

transfer_model = LogisticRegression(max_iter=1000, random_state=42)
transfer_model.fit(X_t_train_combined, y_t_train)
transfer_acc = accuracy_score(y_t_test, transfer_model.predict(X_t_test_combined))

print(f"\nTransfer learning accuracy: {transfer_acc:.1%}")
print(f"Baseline accuracy:          {baseline_acc:.1%}")
improvement = transfer_acc - baseline_acc
if improvement > 0:
    print(f"Improvement: +{improvement:.1%} -- Transfer learning helps!")
else:
    print(f"Similar performance — both domains may be different enough")
    print(f"that more sophisticated transfer is needed.")

# Method 2: Feature importance transfer
print("\n--- Method 2: Feature Importance Transfer ---\n")
importances = source_model.feature_importances_
feature_names = ['GDP', 'Employment', 'Exports', 'Seasonal', 'Energy']
print("Source model learned these feature importances:")
for name, imp in sorted(zip(feature_names, importances),
                         key=lambda x: -x[1]):
    bar = '#' * int(imp * 50)
    print(f"  {name:<12} {imp:.3f} {bar}")

# Use importances as feature weights for the target task
X_t_train_weighted = X_t_train * importances
X_t_test_weighted = X_t_test * importances

weighted_model = LogisticRegression(max_iter=1000, random_state=42)
weighted_model.fit(X_t_train_weighted, y_t_train)
weighted_acc = accuracy_score(y_t_test, weighted_model.predict(X_t_test_weighted))
print(f"\nWeighted transfer accuracy: {weighted_acc:.1%}")


# =====================================================================
# SECTION 6: Summary & Caribbean Applications
# =====================================================================
print("\n--- Part 6: Real-World Caribbean Transfer Learning ---\n")
print(f"""
RESULTS COMPARISON:
  Baseline (from scratch):       {baseline_acc:.1%}
  Transfer (leaf features):      {transfer_acc:.1%}
  Transfer (weighted features):  {weighted_acc:.1%}

REAL-WORLD CARIBBEAN APPLICATIONS:

1. MEDICAL IMAGING (All Caribbean hospitals)
   - Take a model trained on millions of US X-rays
   - Fine-tune on a few hundred Caribbean patient scans
   - Works for sickle cell, tropical diseases unique to our region

2. LANGUAGE MODELS (Jamaica, Trinidad, Haiti, Curacao)
   - Start with English language model, fine-tune on Patois/Creole
   - Much better than training from scratch with limited Creole text

3. CROP DISEASE DETECTION (Guyana, Belize, St Vincent)
   - Pre-trained on global crop datasets (millions of images)
   - Fine-tune on local crops: dasheen, breadfruit, ackee, cassava

4. MARINE SPECIES ID (Belize, Bonaire, Tobago)
   - Global fish recognition model adapted for Caribbean species
   - Like a footballer learning a new position — base skills transfer!

5. DISASTER RESPONSE (Hurricane-prone nations)
   - Models trained on global disaster imagery
   - Adapted for Caribbean building styles and terrain

KEY INSIGHT: Transfer learning is the great equalizer! Caribbean
nations with small datasets can leverage global AI advances.
It's like how Caribbean athletes compete globally despite small
populations — we maximize what we have!
""")


# =====================================================================
# QUIZ
# =====================================================================
print("=" * 65)
print("  QUIZ: Transfer Learning")
print("=" * 65)
print("""
Q1: What is transfer learning?
    a) Moving a computer from one island to another
    b) Using knowledge learned from one task to improve performance
       on a different but related task
    c) Copying a dataset
    d) Training two models at the same time

Q2: Why is transfer learning especially valuable for Caribbean AI?
    a) Caribbean computers are faster
    b) Caribbean datasets are often small due to smaller populations,
       so leveraging pre-trained models saves time and improves results
    c) It only works in tropical climates
    d) Caribbean data is easier to collect

Q3: In our demo, what was the "source task"?
    a) Investment risk classification
    b) Economic sector classification (with 500 samples)
    c) Image recognition
    d) Language translation

Q4: What does "fine-tuning" mean in transfer learning?
    a) Making the model smaller
    b) Taking a pre-trained model and retraining some layers on new data
    c) Tuning a radio to a Caribbean station
    d) Removing all learned weights

Q5: A UWI researcher has 200 labelled images of Caribbean coral
    species. What transfer learning strategy would you recommend?
    a) Train from scratch — 200 is plenty
    b) Use a model pre-trained on ImageNet, freeze early layers,
       fine-tune later layers on the 200 coral images
    c) Don't use AI at all
    d) Only use text data

Q6: What is "domain adaptation"?
    a) Buying a new internet domain
    b) Adapting a model trained in one domain (e.g., US health data)
       to work well in a different domain (e.g., Caribbean health data)
    c) Changing the model's programming language
    d) Adapting to a new classroom

Q7: In our experiment, how did we "transfer" knowledge from the
    source model to the target task?
    a) We copied the labels
    b) We used the source Random Forest's leaf node indices as
       additional features for the target model
    c) We used the same model without changes
    d) We deleted the source model

Q8: Why might transfer learning NOT work well sometimes?
    a) The source and target domains are too different (negative transfer)
    b) The computer is too old
    c) Transfer learning always works perfectly
    d) The data is too clean

Q9: How is transfer learning like a Caribbean athlete switching sports?
    a) It's not similar at all
    b) Base skills (fitness, coordination) transfer, but sport-specific
       skills still need to be learned — similar to how general
       features transfer but task-specific layers need retraining
    c) Athletes don't switch sports
    d) They have to start from zero

Q10: Name two Caribbean-specific tasks where transfer learning
     from global models would be beneficial.
     (Open-ended — discuss with classmates!)

(Answers in quiz_answers.md)
""")

print("=" * 65)
print("  Lesson 03 Complete! Transfer learning = work smarter!")
print("  Next up: Reinforcement Learning")
print("=" * 65)
