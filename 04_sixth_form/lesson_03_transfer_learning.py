"""
=============================================================================
Caribbean AI Academy - Sixth Form (Ages 16-18)
Lesson 03: Transfer Learning
=============================================================================
Topic: Reusing Knowledge Across Caribbean AI Tasks

Yow! Transfer learning is one of the BIGGEST ideas in modern AI.
Instead of training a model from scratch every time (expensive and
slow), we TRANSFER knowledge from one task to another.

Think of it like this: if yuh already know how to play cricket,
learning baseball is MUCH easier — the batting, throwing, and
fielding skills transfer. Same idea with AI models!

Why this matters for the Caribbean:
  - We often have SMALL datasets (small island populations)
  - Training big models from scratch needs expensive GPUs
  - Transfer learning lets us leverage models trained on BIG
    datasets and fine-tune them for OUR Caribbean-specific tasks

In this lesson we use scikit-learn to demonstrate the concept:
train on one Caribbean task, transfer to another.

Requirements: pip install scikit-learn numpy
Author: Adrian Dunkley | Caribbean AI Academy
=============================================================================
"""

import numpy as np
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
from sklearn.preprocessing import StandardScaler

print("=" * 65)
print("  LESSON 03: Transfer Learning")
print("  Reusing AI Knowledge Across Caribbean Tasks")
print("=" * 65)

# =====================================================================
# SECTION 1: What is Transfer Learning?
# =====================================================================
print("\n--- Part 1: The Concept ---\n")
print("""
TRANSFER LEARNING = Training on Task A, then applying that
knowledge to Task B.

Caribbean Examples:
  - Train a model to classify Jamaican land use from satellite data,
    then TRANSFER it to classify Trinidad land use (similar geography)
  - Train on English text analysis, then fine-tune for Caribbean
    Creole/Patois text
  - Learn hurricane patterns from Atlantic data, apply to specific
    Caribbean island predictions

In deep learning (CNNs), this usually means:
  1. Take a pre-trained model (e.g., trained on millions of images)
  2. Freeze the early layers (they detect general features)
  3. Retrain only the last few layers on YOUR specific data

With scikit-learn, we'll demonstrate the concept by:
  1. Training on one Caribbean economic dataset (Source Task)
  2. Using that model's learned features on another dataset (Target Task)
""")


# =====================================================================
# SECTION 2: Source Task — Caribbean Tourism Classification
# =====================================================================
print("--- Part 2: Source Task — Tourism Season Classification ---\n")


def generate_tourism_data(n_samples=500, country="Jamaica"):
    """
    Generate synthetic tourism data for a Caribbean country.
    Features: monthly_visitors, hotel_occupancy, avg_spend_usd,
              cruise_arrivals, temperature, rainfall
    Target: season type (0=Low, 1=Shoulder, 2=Peak)
    """
    np.random.seed(42 if country == "Jamaica" else 123)
    X = []
    y = []

    for _ in range(n_samples):
        season = np.random.choice([0, 1, 2], p=[0.3, 0.35, 0.35])

        if season == 0:  # Low season (May-Nov for most Caribbean)
            visitors = np.random.normal(80000, 15000)
            occupancy = np.random.normal(45, 10)
            spend = np.random.normal(120, 25)
            cruise = np.random.normal(5000, 2000)
            temp = np.random.normal(30, 1.5)
            rain = np.random.normal(200, 50)
        elif season == 1:  # Shoulder season
            visitors = np.random.normal(140000, 20000)
            occupancy = np.random.normal(65, 8)
            spend = np.random.normal(180, 30)
            cruise = np.random.normal(12000, 3000)
            temp = np.random.normal(28, 1.5)
            rain = np.random.normal(100, 30)
        else:  # Peak season (Dec-Apr)
            visitors = np.random.normal(220000, 25000)
            occupancy = np.random.normal(85, 7)
            spend = np.random.normal(250, 35)
            cruise = np.random.normal(25000, 5000)
            temp = np.random.normal(26, 1.5)
            rain = np.random.normal(50, 20)

        X.append([max(0, visitors), np.clip(occupancy, 0, 100),
                  max(0, spend), max(0, cruise), temp, max(0, rain)])
        y.append(season)

    return np.array(X, dtype=np.float32), np.array(y)


# Generate SOURCE data (Jamaica)
X_jamaica, y_jamaica = generate_tourism_data(500, "Jamaica")
feature_names = ['Visitors', 'Occupancy%', 'AvgSpend$', 'Cruise',
                 'Temp_C', 'Rainfall_mm']
season_names = ['Low Season', 'Shoulder', 'Peak Season']

print(f"Source Dataset: Jamaica Tourism ({len(X_jamaica)} samples)")
print(f"Features: {feature_names}")
print(f"Classes: {season_names}")
for i, s in enumerate(season_names):
    print(f"  {s}: {np.sum(y_jamaica == i)} samples")

# Train the source model
scaler_source = StandardScaler()
X_ja_scaled = scaler_source.fit_transform(X_jamaica)
X_ja_train, X_ja_test, y_ja_train, y_ja_test = train_test_split(
    X_ja_scaled, y_jamaica, test_size=0.2, random_state=42
)

source_model = GradientBoostingClassifier(
    n_estimators=100, max_depth=4, random_state=42
)
source_model.fit(X_ja_train, y_ja_train)

source_acc = accuracy_score(y_ja_test, source_model.predict(X_ja_test))
print(f"\nSource Model (Jamaica) Test Accuracy: {source_acc:.1%}")

# Feature importance — what did the model learn?
print("\nFeature Importance (what the model learned from Jamaica):")
for name, imp in sorted(zip(feature_names,
                            source_model.feature_importances_),
                        key=lambda x: x[1], reverse=True):
    bar = "#" * int(imp * 50)
    print(f"  {name:<15} {imp:.3f} {bar}")


# =====================================================================
# SECTION 3: Target Task — Transfer to Barbados
# =====================================================================
print("\n--- Part 3: Transfer to Barbados (Small Dataset!) ---\n")

# Generate TARGET data (Barbados — SMALL dataset, simulating
# limited data availability typical of smaller Caribbean nations)
X_barbados, y_barbados = generate_tourism_data(80, "Barbados")

print(f"Target Dataset: Barbados Tourism (only {len(X_barbados)} samples!)")
print("(Smaller islands often have limited data — transfer learning helps!)")

# Scale using the SOURCE scaler (transfer the preprocessing knowledge)
X_bb_scaled = scaler_source.transform(X_barbados)
X_bb_train, X_bb_test, y_bb_train, y_bb_test = train_test_split(
    X_bb_scaled, y_barbados, test_size=0.25, random_state=42
)

# APPROACH 1: Train from scratch on Barbados (limited data)
scratch_model = GradientBoostingClassifier(
    n_estimators=100, max_depth=4, random_state=42
)
scratch_model.fit(X_bb_train, y_bb_train)
scratch_acc = accuracy_score(y_bb_test, scratch_model.predict(X_bb_test))

# APPROACH 2: Transfer — use Jamaica model directly on Barbados
transfer_direct_acc = accuracy_score(
    y_bb_test, source_model.predict(X_bb_test)
)

# APPROACH 3: Fine-tune — combine Jamaica knowledge + Barbados data
# We simulate fine-tuning by training on combined features
# (source model predictions as additional features)
ja_predictions_train = source_model.predict_proba(X_bb_train)
ja_predictions_test = source_model.predict_proba(X_bb_test)

X_combined_train = np.hstack([X_bb_train, ja_predictions_train])
X_combined_test = np.hstack([X_bb_test, ja_predictions_test])

finetune_model = LogisticRegression(max_iter=1000, random_state=42)
finetune_model.fit(X_combined_train, y_bb_train)
finetune_acc = accuracy_score(
    y_bb_test, finetune_model.predict(X_combined_test)
)

print("\n--- Results Comparison ---\n")
print(f"  1. Train from scratch (Barbados only):  {scratch_acc:.1%}")
print(f"  2. Direct transfer (Jamaica model):     {transfer_direct_acc:.1%}")
print(f"  3. Fine-tuned transfer:                 {finetune_acc:.1%}")
print("""
Key Insight: When yuh have limited data (common in smaller Caribbean
nations), transfer learning often performs better than training from
scratch. The Jamaica model already learned general tourism patterns
that apply across the Caribbean!

This is like how a Jamaican sprinter (Usain Bolt) could transition
to bobsled — the explosive speed and athletic skills TRANSFER!
""")


# =====================================================================
# SECTION 4: Real-World Transfer Learning in Deep Learning
# =====================================================================
print("--- Part 4: Transfer Learning with Deep Learning (Concept) ---\n")
print("""
In practice, transfer learning is HUGE in deep learning:

HOW IT WORKS WITH CNNs (e.g., for Caribbean satellite images):

  1. Start with a PRE-TRAINED model (e.g., ResNet, trained on
     millions of images from ImageNet)

  2. The early layers already know how to detect:
     - Edges, corners, textures (Layer 1-2)
     - Shapes, patterns (Layer 3-4)
     - Complex objects (Layer 5+)

  3. FREEZE early layers (keep their knowledge)

  4. REPLACE the final classification layer with YOUR task:
     - Original: 1000 ImageNet classes
     - New: 5 Caribbean land cover classes

  5. FINE-TUNE on your small Caribbean dataset

  Result: You get excellent performance even with just a few
  hundred Caribbean satellite images!

POPULAR PRE-TRAINED MODELS:
  - ResNet (image classification)
  - BERT / GPT (text / language)
  - YOLOv8 (object detection)
  - Whisper (speech recognition — could adapt for Caribbean accents!)

WHY THIS IS CRITICAL FOR THE CARIBBEAN:
  - Training GPT-4 from scratch costs millions of USD
  - Caribbean research budgets are limited
  - Transfer learning = world-class AI on a Caribbean budget
  - UWI researchers regularly use transfer learning for local tasks
""")


# =====================================================================
# SECTION 5: Hands-On Exercise
# =====================================================================
print("--- Part 5: Try It Yourself ---\n")
print("""
EXERCISE: Transfer Learning for Caribbean Agriculture

Scenario: You trained a crop disease classifier on Jamaican banana
plants (large dataset). Now you want to classify diseases on
St. Lucian banana plants (small dataset, only 50 images).

Steps to try:
  1. Generate synthetic "Jamaica" crop data (large, 500 samples)
  2. Train a RandomForest classifier
  3. Generate synthetic "St. Lucia" crop data (small, 50 samples)
  4. Compare: train from scratch vs. transfer from Jamaica model
  5. Which approach gives better accuracy?

Hint: The crops are similar (both Caribbean banana varieties),
so knowledge SHOULD transfer well — like how cricket skills
transfer between playing in Sabina Park vs Kensington Oval!
""")


# =====================================================================
# QUIZ
# =====================================================================
print("=" * 65)
print("  QUIZ: Transfer Learning")
print("=" * 65)
print("""
Q1: What is transfer learning in one sentence?
    a) Training a model to transfer data between computers
    b) Using knowledge learned from one task to improve
       performance on a different but related task
    c) Copying a model without any changes
    d) Training two models at the same time

Q2: Why is transfer learning particularly valuable for Caribbean
    AI applications?
    a) Caribbean data is always perfect
    b) Small island nations often have limited datasets, and
       transfer learning leverages larger external datasets
    c) It only works in tropical climates
    d) It's cheaper than buying computers

Q3: In CNN transfer learning, why do we freeze early layers?
    a) To make training faster only
    b) Because early layers learn general features (edges, textures)
       that are useful across many tasks
    c) Because early layers are broken
    d) To reduce the model size

Q4: What does "fine-tuning" mean in transfer learning?
    a) Making the model smaller
    b) Retraining only the later layers on your new target dataset
       while keeping early layer knowledge
    c) Fixing bugs in the code
    d) Training from scratch

Q5: In our experiment, why might the Jamaica tourism model work
    well on Barbados data?
    a) Jamaica and Barbados are the same country
    b) Tourism patterns across Caribbean nations share similar
       seasonal trends and economic features
    c) The model memorized Barbados data
    d) Random chance

Q6: What is a "pre-trained model"?
    a) A model that has not been trained yet
    b) A model previously trained on a large dataset that can be
       adapted for new tasks
    c) A model that only works once
    d) A model trained on Caribbean data only

Q7: A UWI researcher has 100 coral reef images from Tobago.
    What transfer learning strategy would you recommend?
    a) Train a massive CNN from scratch
    b) Use a pre-trained ImageNet model, freeze early layers,
       and fine-tune the last layers on the 100 Tobago images
    c) Don't use AI at all
    d) Collect 1 million more images first

Q8: What is the main RISK of transfer learning?
    a) It's too fast
    b) If the source and target tasks are too different, the
       transferred knowledge may hurt rather than help
       (negative transfer)
    c) It uses too little memory
    d) It always overfits

Q9: How is transfer learning like a West Indies cricketer
    switching from Test to T20 format?
    a) It's not related at all
    b) Core skills (batting technique, bowling accuracy) transfer,
       but the player fine-tunes strategy for the new format
    c) They have to relearn everything
    d) The cricket ball is different

Q10: Name TWO Caribbean applications where transfer learning
     from international models would be beneficial.
     a) Training models on Mars data
     b) Adapting an English NLP model for Caribbean Creole text,
        and adapting a global weather model for Caribbean hurricanes
     c) Building a calculator app
     d) Designing websites

(Answers in quiz_answers.md)
""")

print("=" * 65)
print("  Lesson 03 Complete! Knowledge transfers — just like skills!")
print("  Next up: Reinforcement Learning — AI learns by doing")
print("=" * 65)
