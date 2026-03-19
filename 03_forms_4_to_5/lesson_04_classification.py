"""
================================================================================
🌳 FORMS 4-5 | LESSON 04: MULTI-CLASS CLASSIFICATION
================================================================================
Caribbean AI Academy — Designed by Adrian Dunkley (https://Adriandunkley.net)
FREE for Caribbean students. No paywall. No gatekeeping.
================================================================================

"Caribbean music is di heartbeat of di region. Reggae, soca, dancehall,
calypso, reggaeton, kompa, chutney — each one unique, each one OURS.
Today we teach di machine fi tell dem apart!"

In dis lesson yuh going learn:
  1. Multi-class classification (more than 2 categories)
  2. Multiple classifiers: Decision Tree, Random Forest, KNN, SVM
  3. Compare classifier performance
  4. Confusion matrices — where di model get confused

Prerequisites: Lessons 01-03
Install: pip install numpy pandas scikit-learn matplotlib
================================================================================
"""

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')

print("=" * 70)
print("🌳 LESSON 04: MULTI-CLASS CLASSIFICATION")
print("   'Teaching di machine fi know we music'")
print("=" * 70)

# ============================================================================
# PART 1: MULTI-CLASS VS BINARY CLASSIFICATION
# ============================================================================

print("""
╔══════════════════════════════════════════════════════════════════════╗
║  PART 1: MULTI-CLASS CLASSIFICATION                                ║
╚══════════════════════════════════════════════════════════════════════╝

In Lesson 01, we did BINARY classification (2 classes):
  Hurricane hit: YES or NO

Now we doing MULTI-CLASS classification (many classes):
  Music genre: Reggae, Soca, Dancehall, Calypso, Reggaeton, Kompa, Chutney

Dis more challenging! Instead of drawing ONE line between two groups,
di machine haffi draw MANY boundaries between MANY groups.

Think of it like dis:
  Binary    = "Is dis mango ripe or not?" (2 options)
  Multi-class = "Is dis a mango, papaya, guava, soursop, or june plum?" (5 options)
""")

# ============================================================================
# PART 2: CARIBBEAN MUSIC GENRE DATA
# ============================================================================

print("""
╔══════════════════════════════════════════════════════════════════════╗
║  PART 2: CREATING CARIBBEAN MUSIC GENRE DATA                      ║
╚══════════════════════════════════════════════════════════════════════╝

We creating synthetic AUDIO FEATURES fi 7 Caribbean music genres.
In real life, these come from digital signal processing.
We simulate: tempo, beats_per_bar, energy, danceability,
             bass_intensity, vocal_presence, rhythm_complexity

Each genre has its own "signature":
  - Reggae:    medium tempo, strong offbeat, heavy bass
  - Soca:      fast tempo, high energy, very danceable
  - Dancehall: medium-fast, strong bass, rhythmic vocals
  - Calypso:   medium tempo, melodic, moderate energy
  - Reggaeton: medium tempo, distinctive dembow beat, bass-heavy
  - Kompa:     slow-medium, smooth rhythm, romantic feel
  - Chutney:   fast tempo, high energy, complex rhythms (Indo-Caribbean)
""")

np.random.seed(42)

genres = {
    'Reggae': {
        'tempo': (70, 90), 'beats_per_bar': (3.5, 4.5), 'energy': (0.4, 0.65),
        'danceability': (0.55, 0.75), 'bass_intensity': (0.7, 0.95),
        'vocal_presence': (0.5, 0.75), 'rhythm_complexity': (0.3, 0.55)
    },
    'Soca': {
        'tempo': (130, 160), 'beats_per_bar': (3.8, 4.2), 'energy': (0.8, 1.0),
        'danceability': (0.85, 1.0), 'bass_intensity': (0.6, 0.85),
        'vocal_presence': (0.6, 0.85), 'rhythm_complexity': (0.4, 0.65)
    },
    'Dancehall': {
        'tempo': (95, 115), 'beats_per_bar': (3.8, 4.2), 'energy': (0.65, 0.85),
        'danceability': (0.75, 0.95), 'bass_intensity': (0.75, 0.95),
        'vocal_presence': (0.65, 0.90), 'rhythm_complexity': (0.5, 0.75)
    },
    'Calypso': {
        'tempo': (100, 130), 'beats_per_bar': (3.5, 4.5), 'energy': (0.5, 0.7),
        'danceability': (0.6, 0.8), 'bass_intensity': (0.4, 0.6),
        'vocal_presence': (0.7, 0.9), 'rhythm_complexity': (0.35, 0.55)
    },
    'Reggaeton': {
        'tempo': (85, 105), 'beats_per_bar': (3.9, 4.1), 'energy': (0.7, 0.9),
        'danceability': (0.8, 0.95), 'bass_intensity': (0.8, 1.0),
        'vocal_presence': (0.55, 0.8), 'rhythm_complexity': (0.25, 0.45)
    },
    'Kompa': {
        'tempo': (110, 135), 'beats_per_bar': (3.8, 4.2), 'energy': (0.45, 0.65),
        'danceability': (0.7, 0.85), 'bass_intensity': (0.5, 0.7),
        'vocal_presence': (0.6, 0.8), 'rhythm_complexity': (0.4, 0.6)
    },
    'Chutney': {
        'tempo': (130, 160), 'beats_per_bar': (3.5, 4.5), 'energy': (0.75, 0.95),
        'danceability': (0.8, 0.95), 'bass_intensity': (0.45, 0.65),
        'vocal_presence': (0.7, 0.9), 'rhythm_complexity': (0.65, 0.9)
    }
}

samples_per_genre = 100
all_tracks = []

for genre, params in genres.items():
    for _ in range(samples_per_genre):
        track = {'genre': genre}
        for feature, (low, high) in params.items():
            track[feature] = round(np.random.uniform(low, high), 3)
        all_tracks.append(track)

music_df = pd.DataFrame(all_tracks)
# Shuffle
music_df = music_df.sample(frac=1, random_state=42).reset_index(drop=True)

print("🎵 Caribbean Music Dataset Preview:")
print("-" * 90)
print(music_df.head(10).to_string(index=False))
print(f"\nTotal tracks: {len(music_df)}")
print(f"Genres: {music_df['genre'].nunique()}")
print(f"\nSamples per genre:")
print(music_df['genre'].value_counts().to_string())

# Genre profiles
print(f"\n📊 Genre Audio Profiles (averages):")
profile = music_df.groupby('genre').mean(numeric_only=True).round(3)
print(profile.to_string())

# ============================================================================
# PART 3: PREPARING DATA
# ============================================================================

print("""
╔══════════════════════════════════════════════════════════════════════╗
║  PART 3: PREPARING DI DATA                                        ║
╚══════════════════════════════════════════════════════════════════════╝
""")

feature_cols = ['tempo', 'beats_per_bar', 'energy', 'danceability',
                'bass_intensity', 'vocal_presence', 'rhythm_complexity']

X = music_df[feature_cols].values
y = music_df['genre'].values

# Encode labels
le = LabelEncoder()
y_encoded = le.fit_transform(y)

# Split
X_train, X_test, y_train, y_test = train_test_split(
    X, y_encoded, test_size=0.2, random_state=42, stratify=y_encoded
)

# Scale for algorithms that need it (KNN, SVM)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

print(f"Training samples: {len(X_train)}")
print(f"Testing samples:  {len(X_test)}")
print(f"Classes: {list(le.classes_)}")

# ============================================================================
# PART 4: TRAINING MULTIPLE CLASSIFIERS
# ============================================================================

print("""
╔══════════════════════════════════════════════════════════════════════╗
║  PART 4: TRAINING MULTIPLE CLASSIFIERS                             ║
╚══════════════════════════════════════════════════════════════════════╝

We going try 5 different classifiers and see which one best fi dis task.
Each algorithm has its own approach:

1. DECISION TREE — Ask yes/no questions (like 20 questions game)
2. RANDOM FOREST — Many trees vote together
3. K-NEAREST NEIGHBORS (KNN) — Look at di K closest examples
4. SUPPORT VECTOR MACHINE (SVM) — Find boundaries between classes
5. GRADIENT BOOSTING — Trees dat learn from each other's mistakes
""")

classifiers = {
    'Decision Tree': (DecisionTreeClassifier(max_depth=10, random_state=42), False),
    'Random Forest': (RandomForestClassifier(n_estimators=100, max_depth=12, random_state=42), False),
    'K-Nearest Neighbors (K=5)': (KNeighborsClassifier(n_neighbors=5), True),
    'Support Vector Machine': (SVC(kernel='rbf', random_state=42), True),
    'Gradient Boosting': (GradientBoostingClassifier(n_estimators=100, max_depth=5, random_state=42), False),
}

results = {}
print(f"{'Classifier':<30s} {'Accuracy':>10s}")
print(f"{'─'*40}")

for name, (clf, needs_scaling) in classifiers.items():
    if needs_scaling:
        clf.fit(X_train_scaled, y_train)
        y_pred = clf.predict(X_test_scaled)
    else:
        clf.fit(X_train, y_train)
        y_pred = clf.predict(X_test)

    acc = accuracy_score(y_test, y_pred)
    results[name] = {'accuracy': acc, 'predictions': y_pred}
    print(f"{name:<30s} {acc*100:>9.1f}%")

# Find best classifier
best_name = max(results, key=lambda x: results[x]['accuracy'])
best_acc = results[best_name]['accuracy']
print(f"\n🏆 Best Classifier: {best_name} ({best_acc*100:.1f}%)")

# ============================================================================
# PART 5: DETAILED ANALYSIS OF BEST MODEL
# ============================================================================

print("""
╔══════════════════════════════════════════════════════════════════════╗
║  PART 5: DETAILED ANALYSIS — WHERE DI MODEL GET CONFUSED?          ║
╚══════════════════════════════════════════════════════════════════════╝
""")

best_pred = results[best_name]['predictions']

print(f"📊 Classification Report for {best_name}:")
print(classification_report(y_test, best_pred,
                            target_names=le.classes_))

# Confusion Matrix
cm = confusion_matrix(y_test, best_pred)
print("📊 Confusion Matrix:")
print(f"   (Rows = Actual, Columns = Predicted)")
print(f"\n   {'':>12s}", end="")
for g in le.classes_:
    print(f"{g[:6]:>8s}", end="")
print()
for i, genre in enumerate(le.classes_):
    print(f"   {genre[:12]:>12s}", end="")
    for j in range(len(le.classes_)):
        val = cm[i][j]
        if i == j:
            print(f"  [{val:>3d}]", end="")  # correct predictions
        else:
            print(f"   {val:>3d} ", end="")

    print()

# Which genres get confused most?
print(f"\n🔍 Where di model get confused:")
for i in range(len(le.classes_)):
    for j in range(len(le.classes_)):
        if i != j and cm[i][j] >= 3:
            print(f"   {le.classes_[i]} mistaken for {le.classes_[j]}: {cm[i][j]} times")

# ============================================================================
# PART 6: FEATURE IMPORTANCE
# ============================================================================

print("""
╔══════════════════════════════════════════════════════════════════════╗
║  PART 6: WHICH AUDIO FEATURES MATTER MOST?                        ║
╚══════════════════════════════════════════════════════════════════════╝
""")

# Use Random Forest for feature importance
rf = classifiers['Random Forest'][0]
print("📊 Feature Importance (Random Forest):")
for feat, imp in sorted(zip(feature_cols, rf.feature_importances_),
                         key=lambda x: x[1], reverse=True):
    bar = "█" * int(imp * 50)
    print(f"   {feat:22s} {imp:.3f} {bar}")

print("""
💡 INSIGHT: Tempo is likely di most important feature!
   - Soca and Chutney are FAST (130-160 BPM)
   - Reggae is SLOW (70-90 BPM)
   - Dis alone separate several genres!

   But fi distinguish Soca from Chutney (both fast),
   we need rhythm_complexity and bass_intensity.
""")

# ============================================================================
# PART 7: VISUALIZE RESULTS
# ============================================================================

fig, axes = plt.subplots(1, 2, figsize=(16, 7))

# Plot 1: Accuracy comparison
ax1 = axes[0]
names = list(results.keys())
accuracies = [results[n]['accuracy'] * 100 for n in names]
colors = ['#e74c3c', '#2ecc71', '#3498db', '#f39c12', '#9b59b6']
bars = ax1.barh(names, accuracies, color=colors, edgecolor='black', linewidth=0.5)
ax1.set_xlabel('Accuracy (%)', fontsize=12)
ax1.set_title('Classifier Comparison — Caribbean Music Genre', fontsize=13)
ax1.set_xlim(0, 105)
for bar, acc in zip(bars, accuracies):
    ax1.text(bar.get_width() + 0.5, bar.get_y() + bar.get_height()/2,
             f'{acc:.1f}%', va='center', fontsize=10, fontweight='bold')
ax1.grid(axis='x', alpha=0.3)

# Plot 2: Confusion matrix heatmap
ax2 = axes[1]
im = ax2.imshow(cm, cmap='YlOrRd', aspect='auto')
ax2.set_xticks(range(len(le.classes_)))
ax2.set_yticks(range(len(le.classes_)))
ax2.set_xticklabels([g[:5] for g in le.classes_], rotation=45, ha='right', fontsize=9)
ax2.set_yticklabels([g[:5] for g in le.classes_], fontsize=9)
ax2.set_xlabel('Predicted', fontsize=12)
ax2.set_ylabel('Actual', fontsize=12)
ax2.set_title(f'Confusion Matrix — {best_name}', fontsize=13)

for i in range(len(le.classes_)):
    for j in range(len(le.classes_)):
        color = 'white' if cm[i, j] > cm.max() / 2 else 'black'
        ax2.text(j, i, str(cm[i, j]), ha='center', va='center',
                 fontsize=10, fontweight='bold', color=color)

plt.colorbar(im, ax=ax2)
plt.tight_layout()
plt.savefig('/home/user/Caribbean-AI-Students/03_forms_4_to_5/music_classification.png',
            dpi=150, bbox_inches='tight')
print("📊 Plot saved to: music_classification.png")
plt.close()

# ============================================================================
# PART 8: CLASSIFY NEW SONGS
# ============================================================================

print("""
╔══════════════════════════════════════════════════════════════════════╗
║  PART 8: CLASSIFY NEW SONGS!                                      ║
╚══════════════════════════════════════════════════════════════════════╝

Let we test di model on some new "songs" with known characteristics!
""")

new_songs = pd.DataFrame({
    'name': [
        "Bob Marley vibes — slow, heavy bass",
        "Carnival jump-up — fast, high energy",
        "Daddy Yankee style — dembow beat",
        "Tabanka Djaz feel — smooth, romantic",
        "Sundar Popo tribute — fast, complex rhythm",
        "Mighty Sparrow style — mid-tempo, melodic",
        "Vybz Kartel energy — dancehall riddim"
    ],
    'tempo': [78, 148, 95, 120, 145, 115, 108],
    'beats_per_bar': [4.0, 4.0, 4.0, 4.0, 4.0, 4.0, 4.0],
    'energy': [0.5, 0.9, 0.8, 0.55, 0.85, 0.6, 0.75],
    'danceability': [0.65, 0.95, 0.88, 0.78, 0.90, 0.7, 0.85],
    'bass_intensity': [0.85, 0.75, 0.90, 0.6, 0.55, 0.5, 0.88],
    'vocal_presence': [0.65, 0.75, 0.7, 0.7, 0.8, 0.85, 0.8],
    'rhythm_complexity': [0.4, 0.55, 0.35, 0.5, 0.8, 0.45, 0.6]
})

X_new = new_songs[feature_cols].values
predictions = rf.predict(X_new)
pred_genres = le.inverse_transform(predictions)

print(f"🎶 New Song Predictions:")
print(f"   {'Song Description':<45s} {'Predicted Genre'}")
print(f"   {'─'*65}")
for _, row in new_songs.iterrows():
    idx = new_songs.index[new_songs['name'] == row['name']][0]
    print(f"   {row['name']:<45s} {pred_genres[idx]}")

# ============================================================================
# PART 9: CARIBBEAN HEALTHCARE — BONUS
# ============================================================================

print("""
╔══════════════════════════════════════════════════════════════════════╗
║  BONUS: DENGUE SEVERITY CLASSIFICATION                             ║
╚══════════════════════════════════════════════════════════════════════╝

Dengue fever is a SERIOUS problem in di Caribbean.
Let we build a quick classifier fi dengue severity.
""")

np.random.seed(88)
n_patients = 300

# Features
temperature = np.random.uniform(37.0, 41.5, n_patients)
platelet_count = np.random.uniform(20, 250, n_patients)  # thousands/uL
white_blood_cell = np.random.uniform(2.0, 12.0, n_patients)
days_fever = np.random.randint(1, 10, n_patients)
age = np.random.randint(5, 75, n_patients)

# Severity: 0=Mild, 1=Moderate, 2=Severe
severity_score = (
    (temperature - 37) * 3 +
    (150 - platelet_count) * 0.05 +
    (6 - white_blood_cell) * 0.5 +
    days_fever * 0.8 +
    np.random.normal(0, 1.5, n_patients)
)

severity = np.digitize(severity_score,
                        bins=[np.percentile(severity_score, 33),
                              np.percentile(severity_score, 66)])

dengue_X = np.column_stack([temperature, platelet_count, white_blood_cell,
                              days_fever, age])
Xd_tr, Xd_te, yd_tr, yd_te = train_test_split(dengue_X, severity,
                                                  test_size=0.2, random_state=42)

dengue_rf = RandomForestClassifier(n_estimators=100, max_depth=8, random_state=42)
dengue_rf.fit(Xd_tr, yd_tr)
yd_pred = dengue_rf.predict(Xd_te)

print(f"🏥 Dengue Severity Classifier:")
print(f"   Accuracy: {accuracy_score(yd_te, yd_pred)*100:.1f}%")
print(classification_report(yd_te, yd_pred,
                            target_names=['Mild', 'Moderate', 'Severe']))

# ============================================================================
# KEY CONCEPTS SUMMARY
# ============================================================================

print("""
╔══════════════════════════════════════════════════════════════════════╗
║  KEY CONCEPTS SUMMARY                                              ║
╚══════════════════════════════════════════════════════════════════════╝

What yuh learn today:

1. MULTI-CLASS CLASSIFICATION = More than 2 categories
   - 7 Caribbean music genres, 3 dengue severity levels, etc.

2. MULTIPLE CLASSIFIERS = Different algorithms, different strengths
   - Decision Tree: simple, interpretable
   - Random Forest: robust, good all-around
   - KNN: intuitive, no training needed
   - SVM: powerful for complex boundaries
   - Gradient Boosting: learns from mistakes

3. CONFUSION MATRIX = Shows exactly WHERE di model mess up
   - Which genres get mixed up? (Calypso ↔ Kompa maybe?)

4. FEATURE IMPORTANCE = Which features drive di classification
   - Tempo is king fi music genre classification!

5. SCALING MATTERS = Some algorithms (KNN, SVM) need scaled data

6. NO SINGLE BEST ALGORITHM = Always try multiple and compare!

REAL WORLD: Music streaming services in di Caribbean could use dis
to automatically tag songs, create playlists, and recommend music!
""")

# ============================================================================
# QUIZ TIME!
# ============================================================================

print("""
╔══════════════════════════════════════════════════════════════════════╗
║  📝 QUIZ TIME! — 8 Questions                                      ║
╚══════════════════════════════════════════════════════════════════════╝

Answer these questions. Check yuh answers in quiz_answers.md

Q1: What is di difference between binary and multi-class classification?
    a) Binary is for text, multi-class is for numbers
    b) Binary has 2 classes; multi-class has 3 or more
    c) Multi-class is always more accurate
    d) They are the same thing

Q2: K-Nearest Neighbors (KNN) classifies a new point by:
    a) Building a decision tree
    b) Looking at the K closest training examples and voting
    c) Finding the mathematical equation
    d) Random guessing

Q3: Why did we SCALE di data for KNN and SVM but not for Random Forest?
    a) Random Forest doesn't work with scaled data
    b) KNN and SVM use distances, which are affected by feature scales
    c) It was a mistake
    d) Scaling is always optional

Q4: In di confusion matrix, what do di DIAGONAL values represent?
    a) The errors
    b) The correct predictions (actual = predicted)
    c) The total count
    d) The feature importance

Q5: If Calypso and Kompa get confused often, what does dat suggest?
    a) The algorithm is broken
    b) These genres have similar audio features (similar tempo, energy)
    c) We need less data
    d) We should remove one genre

Q6: Which audio feature yuh EXPECT to be most useful fi distinguishing
    Reggae from Soca?
    a) Beats per bar (both around 4)
    b) Tempo (Reggae ~80 BPM, Soca ~145 BPM — very different!)
    c) Vocal presence (similar for both)
    d) The song title

Q7: Gradient Boosting "learns from mistakes." What dis mean?
    a) It deletes wrong predictions
    b) Each new tree focuses on correcting errors from previous trees
    c) It asks the user for corrections
    d) It runs the same model twice

Q8: Fi di dengue classifier, why is PLATELET COUNT an important feature?
    a) It's not important
    b) Low platelet count is a clinical indicator of severe dengue
    c) Higher platelets mean more dengue
    d) Platelets only matter for malaria
""")

print("""
╔══════════════════════════════════════════════════════════════════════╗
║  🏆 BONUS CHALLENGE                                                ║
╚══════════════════════════════════════════════════════════════════════╝

Challenge 1: Add 2 more Caribbean genres (e.g., Bouyon from Dominica,
  Zouk from Martinique/Guadeloupe). Create their audio profiles.

Challenge 2: Implement a simple "Shazam-like" function that takes
  audio features and returns the top 3 most likely genres.

Challenge 3: Try tuning K in KNN (try K=1,3,5,7,11,15).
  Plot accuracy vs K. What is di best K?

Challenge 4: Build a Caribbean food classifier! Features could be:
  spice_level, sweetness, cooking_time, uses_coconut, etc.

Next lesson: NLP — sentiment analysis on Caribbean hotel reviews!

Walk good! 🎵
""")
