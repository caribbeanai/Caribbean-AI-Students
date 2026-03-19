"""
================================================================================
🌳 FORMS 4-5 | LESSON 02: UNSUPERVISED LEARNING
================================================================================
Caribbean AI Academy — Designed by Adrian Dunkley (https://Adriandunkley.net)
FREE for Caribbean students. No paywall. No gatekeeping.
================================================================================

"Nobody telling di machine di answers dis time. It haffi FIGURE OUT
di patterns by itself. Dat's unsupervised learning — like exploring
a new island without a map."

In dis lesson yuh going learn:
  1. What unsupervised learning is
  2. K-Means Clustering — grouping similar things together
  3. Apply it to Caribbean economic data — cluster CARICOM nations
  4. Visualize clusters with matplotlib

Prerequisites: Lesson 01 (Supervised Learning)
Install: pip install numpy pandas scikit-learn matplotlib
================================================================================
"""

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score
import matplotlib
matplotlib.use('Agg')  # fi save without display
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')

print("=" * 70)
print("🌳 LESSON 02: UNSUPERVISED LEARNING")
print("   'When di machine haffi figure it out WITHOUT answers'")
print("=" * 70)

# ============================================================================
# PART 1: WHAT IS UNSUPERVISED LEARNING?
# ============================================================================

print("""
╔══════════════════════════════════════════════════════════════════════╗
║  PART 1: WHAT IS UNSUPERVISED LEARNING?                           ║
╚══════════════════════════════════════════════════════════════════════╝

Remember in Lesson 01, we TOLD di machine di answers (labels)?
"Dis is a hit, dat is a miss."

In UNSUPERVISED learning, we say:
"Here di data. I nah tell yuh nutten. Figure out what go together."

Real life example:
  Imagine yuh walk into a Carnival fete and see 500 people.
  Nobody tell yuh who is who. But yuh start GROUPING people:
    - Dem over deh in feathers and beads? Dey di mas players
    - Dem with di whistles and flags? Dey di soca fans
    - Dem by di food stall? Dey di foodies
    - Dem in di VIP section? Dey di sponsors

  Yuh CLUSTERED people based on patterns yuh observed!
  Nobody TOLD yuh the groups — yuh figured it out!

Dat's K-Means Clustering.
""")

# ============================================================================
# PART 2: CARIBBEAN ECONOMIC DATA
# ============================================================================

print("""
╔══════════════════════════════════════════════════════════════════════╗
║  PART 2: CARIBBEAN ECONOMIC DATA                                   ║
╚══════════════════════════════════════════════════════════════════════╝

Di Caribbean is diverse! Some islands depend heavy on tourism,
others on oil and gas, others on agriculture. Let we see if
K-Means can discover these economic groups AUTOMATICALLY.

We going use 4 features for each country:
  - GDP per capita (USD) — how rich on average
  - Tourism % of GDP — how much tourism matter
  - Agriculture % of GDP — how much farming matter
  - Population (thousands) — how big di country

NOTE: Dis is SYNTHETIC data inspired by real Caribbean economics.
Numbers approximate — fi learning purposes.
""")

# Caribbean Economic Data (synthetic but realistic approximations)
caribbean_data = pd.DataFrame({
    'country': [
        'Jamaica', 'Trinidad & Tobago', 'Barbados', 'Guyana',
        'The Bahamas', 'Haiti', 'Dominican Republic', 'Cuba',
        'Antigua & Barbuda', 'Belize', 'Dominica', 'Grenada',
        'St. Kitts & Nevis', 'St. Lucia', 'St. Vincent',
        'Suriname', 'Puerto Rico', 'Cayman Islands',
        'Curacao', 'Aruba', 'Bermuda', 'Turks & Caicos',
        'Montserrat', 'BVI'
    ],
    'gdp_per_capita_usd': [
        5800, 17000, 16500, 6900,
        32000, 1800, 8500, 9500,
        17500, 4800, 7500, 10200,
        19200, 11600, 7400,
        6200, 32000, 73000,
        17800, 28000, 85000, 27000,
        12000, 34000
    ],
    'tourism_pct_gdp': [
        28, 8, 38, 5,
        45, 10, 18, 12,
        58, 35, 25, 22,
        30, 42, 20,
        3, 7, 25,
        15, 50, 8, 55,
        10, 35
    ],
    'agriculture_pct_gdp': [
        7, 1, 2, 18,
        1, 22, 6, 4,
        2, 12, 15, 8,
        1, 3, 8,
        10, 1, 0,
        1, 0, 0, 1,
        2, 1
    ],
    'population_thousands': [
        2960, 1400, 287, 790,
        400, 11400, 10800, 11300,
        97, 400, 72, 112,
        53, 180, 110,
        590, 3200, 65,
        155, 107, 64, 40,
        5, 30
    ]
})

print("🌍 Caribbean Economic Data:")
print("-" * 80)
print(caribbean_data.to_string(index=False))

# ============================================================================
# PART 3: DATA PREPROCESSING — SCALING
# ============================================================================

print("""
╔══════════════════════════════════════════════════════════════════════╗
║  PART 3: DATA PREPROCESSING — WHY WE SCALE                        ║
╚══════════════════════════════════════════════════════════════════════╝

IMPORTANT: K-Means use DISTANCE to measure similarity.

Problem: GDP per capita range from 1,800 to 85,000
         Tourism % range from 3 to 58

If we nah scale, GDP going DOMINATE because di numbers bigger!
Di algorithm would think Haiti and Bermuda different ONLY because of GDP,
ignoring everything else.

Solution: STANDARDIZE — make every feature have mean=0, std=1.
Now every feature equally important.
""")

# Select numeric features
features = ['gdp_per_capita_usd', 'tourism_pct_gdp',
            'agriculture_pct_gdp', 'population_thousands']
X = caribbean_data[features].values

# Scale the data
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

print("Before Scaling (first 5 countries):")
print(f"  {'Country':<22s} {'GDP':>8s} {'Tourism%':>10s} {'Agri%':>8s} {'Pop(K)':>8s}")
for i in range(5):
    print(f"  {caribbean_data['country'].iloc[i]:<22s} "
          f"{X[i][0]:>8.0f} {X[i][1]:>10.0f} {X[i][2]:>8.0f} {X[i][3]:>8.0f}")

print("\nAfter Scaling (first 5 countries):")
print(f"  {'Country':<22s} {'GDP':>8s} {'Tourism%':>10s} {'Agri%':>8s} {'Pop(K)':>8s}")
for i in range(5):
    print(f"  {caribbean_data['country'].iloc[i]:<22s} "
          f"{X_scaled[i][0]:>8.2f} {X_scaled[i][1]:>10.2f} {X_scaled[i][2]:>8.2f} {X_scaled[i][3]:>8.2f}")

# ============================================================================
# PART 4: K-MEANS CLUSTERING
# ============================================================================

print("""
╔══════════════════════════════════════════════════════════════════════╗
║  PART 4: K-MEANS CLUSTERING — HOW IT WORK                         ║
╚══════════════════════════════════════════════════════════════════════╝

K-Means Algorithm (step by step):

1. PICK K (number of clusters yuh want)
2. RANDOMLY place K center points (centroids)
3. ASSIGN each country to di NEAREST centroid
4. MOVE each centroid to di MIDDLE of its assigned countries
5. REPEAT steps 3-4 until nutten change

It like sorting yuh music playlist:
  - Yuh pick 4 genre categories (K=4)
  - Each song go to di nearest genre
  - Di genre definition update based on what songs in deh
  - Keep adjusting until everything settle

But HOW yuh pick K? We use di ELBOW METHOD!
""")

# Elbow Method — find optimal K
inertias = []
K_range = range(2, 9)
for k in K_range:
    kmeans_temp = KMeans(n_clusters=k, random_state=42, n_init=10)
    kmeans_temp.fit(X_scaled)
    inertias.append(kmeans_temp.inertia_)

print("📊 Elbow Method — Finding Optimal K:")
print(f"   {'K':>3s}  {'Inertia':>10s}  {'Visual'}")
for k, inertia in zip(K_range, inertias):
    bar = "█" * int(inertia / max(inertias) * 40)
    print(f"   {k:>3d}  {inertia:>10.1f}  {bar}")

# Also compute silhouette scores
sil_scores = []
for k in K_range:
    kmeans_temp = KMeans(n_clusters=k, random_state=42, n_init=10)
    labels_temp = kmeans_temp.fit_predict(X_scaled)
    sil = silhouette_score(X_scaled, labels_temp)
    sil_scores.append(sil)

print(f"\n📊 Silhouette Scores (higher = better-defined clusters):")
for k, sil in zip(K_range, sil_scores):
    bar = "█" * int(sil * 40)
    print(f"   K={k}: {sil:.3f}  {bar}")

best_k = list(K_range)[np.argmax(sil_scores)]
print(f"\n   Best K by silhouette score: {best_k}")

# ============================================================================
# PART 5: APPLYING K-MEANS WITH K=4
# ============================================================================

print("""
╔══════════════════════════════════════════════════════════════════════╗
║  PART 5: CLUSTERING CARIBBEAN NATIONS (K=4)                        ║
╚══════════════════════════════════════════════════════════════════════╝

Let we use K=4 clusters fi group di Caribbean nations.
Why 4? Because we expect roughly:
  - Tourism-heavy small islands
  - Large diverse economies
  - Oil/resource-rich nations
  - Agriculture-dependent nations

But remember — WE not telling di machine dis! It haffi figure it out!
""")

# Apply K-Means with K=4
kmeans = KMeans(n_clusters=4, random_state=42, n_init=10)
clusters = kmeans.fit_predict(X_scaled)

caribbean_data['cluster'] = clusters

# Display clusters
cluster_names = {}
for c in range(4):
    members = caribbean_data[caribbean_data['cluster'] == c]
    avg_gdp = members['gdp_per_capita_usd'].mean()
    avg_tourism = members['tourism_pct_gdp'].mean()
    avg_agri = members['agriculture_pct_gdp'].mean()
    avg_pop = members['population_thousands'].mean()

    # Auto-label based on characteristics
    if avg_tourism > 35:
        label = "Tourism Powerhouses"
    elif avg_gdp > 30000:
        label = "High-Income Economies"
    elif avg_agri > 10:
        label = "Agriculture & Developing"
    else:
        label = "Diversified Mid-Income"

    cluster_names[c] = label

print("🏝️  Caribbean Economic Clusters:\n")
for c in range(4):
    members = caribbean_data[caribbean_data['cluster'] == c]
    print(f"  CLUSTER {c}: {cluster_names[c]}")
    print(f"  {'─' * 50}")
    countries = ", ".join(members['country'].tolist())
    print(f"  Countries: {countries}")
    print(f"  Avg GDP/capita:   ${members['gdp_per_capita_usd'].mean():,.0f}")
    print(f"  Avg Tourism %:    {members['tourism_pct_gdp'].mean():.0f}%")
    print(f"  Avg Agriculture %:{members['agriculture_pct_gdp'].mean():.0f}%")
    print(f"  Avg Population:   {members['population_thousands'].mean():,.0f}K")
    print()

# ============================================================================
# PART 6: VISUALIZATION
# ============================================================================

print("""
╔══════════════════════════════════════════════════════════════════════╗
║  PART 6: VISUALIZING DI CLUSTERS                                   ║
╚══════════════════════════════════════════════════════════════════════╝

A picture worth a thousand words! Let we plot di clusters.
We plot GDP per capita vs Tourism % and color by cluster.
""")

colors = ['#e74c3c', '#3498db', '#2ecc71', '#f39c12']
cluster_labels_display = [cluster_names.get(i, f"Cluster {i}") for i in range(4)]

fig, axes = plt.subplots(1, 2, figsize=(16, 7))

# Plot 1: GDP vs Tourism
ax1 = axes[0]
for c in range(4):
    mask = clusters == c
    ax1.scatter(caribbean_data.loc[mask, 'gdp_per_capita_usd'],
                caribbean_data.loc[mask, 'tourism_pct_gdp'],
                c=colors[c], label=cluster_labels_display[c],
                s=100, edgecolors='black', linewidth=0.5, alpha=0.8)
    # Add country labels
    for _, row in caribbean_data[mask].iterrows():
        ax1.annotate(row['country'], (row['gdp_per_capita_usd'], row['tourism_pct_gdp']),
                     fontsize=7, ha='center', va='bottom', rotation=15)

ax1.set_xlabel('GDP per Capita (USD)', fontsize=12)
ax1.set_ylabel('Tourism % of GDP', fontsize=12)
ax1.set_title('Caribbean Nations: GDP vs Tourism\n(Colored by K-Means Cluster)', fontsize=13)
ax1.legend(fontsize=9, loc='upper left')
ax1.grid(True, alpha=0.3)

# Plot 2: Agriculture vs Population
ax2 = axes[1]
for c in range(4):
    mask = clusters == c
    ax2.scatter(caribbean_data.loc[mask, 'agriculture_pct_gdp'],
                caribbean_data.loc[mask, 'population_thousands'],
                c=colors[c], label=cluster_labels_display[c],
                s=100, edgecolors='black', linewidth=0.5, alpha=0.8)
    for _, row in caribbean_data[mask].iterrows():
        ax2.annotate(row['country'], (row['agriculture_pct_gdp'], row['population_thousands']),
                     fontsize=7, ha='center', va='bottom', rotation=15)

ax2.set_xlabel('Agriculture % of GDP', fontsize=12)
ax2.set_ylabel('Population (thousands)', fontsize=12)
ax2.set_title('Caribbean Nations: Agriculture vs Population\n(Colored by K-Means Cluster)', fontsize=13)
ax2.legend(fontsize=9, loc='upper right')
ax2.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('/home/user/Caribbean-AI-Students/03_forms_4_to_5/caribbean_clusters.png',
            dpi=150, bbox_inches='tight')
print("📊 Plot saved to: caribbean_clusters.png")
plt.close()

# ============================================================================
# PART 7: MARINE & ENERGY — BONUS CLUSTERING
# ============================================================================

print("""
╔══════════════════════════════════════════════════════════════════════╗
║  BONUS: CARIBBEAN ISLAND ENERGY PROFILE CLUSTERING                 ║
╚══════════════════════════════════════════════════════════════════════╝

Di Caribbean have MASSIVE potential fi renewable energy.
Let we cluster islands by energy profile!
""")

np.random.seed(55)
energy_data = pd.DataFrame({
    'island': [
        'Jamaica', 'Trinidad', 'Barbados', 'Guyana', 'Bahamas',
        'Dominica', 'St. Lucia', 'Grenada', 'Aruba', 'Curacao',
        'Antigua', 'St. Kitts', 'BVI', 'Cayman', 'Bermuda'
    ],
    'solar_hours_daily': [7.2, 6.8, 7.5, 5.5, 7.0, 5.8, 6.5, 6.8, 8.0, 7.8, 7.3, 7.1, 7.2, 7.0, 6.5],
    'avg_wind_speed_ms': [6.5, 5.2, 7.8, 4.5, 7.0, 5.0, 6.2, 5.8, 8.5, 7.5, 7.0, 6.8, 7.2, 6.0, 8.0],
    'fossil_fuel_pct': [85, 95, 70, 80, 99, 65, 82, 78, 60, 88, 90, 85, 95, 98, 92],
    'renewable_target_pct': [50, 10, 65, 30, 30, 100, 35, 30, 50, 25, 15, 20, 10, 20, 25]
})

energy_features = ['solar_hours_daily', 'avg_wind_speed_ms',
                    'fossil_fuel_pct', 'renewable_target_pct']
X_energy = scaler.fit_transform(energy_data[energy_features])

# Cluster into 3 groups
km_energy = KMeans(n_clusters=3, random_state=42, n_init=10)
energy_data['cluster'] = km_energy.fit_predict(X_energy)

print("⚡ Caribbean Energy Profile Clusters:\n")
for c in range(3):
    members = energy_data[energy_data['cluster'] == c]
    print(f"  Cluster {c}:")
    print(f"  Islands: {', '.join(members['island'].tolist())}")
    print(f"  Avg Solar Hours: {members['solar_hours_daily'].mean():.1f}h")
    print(f"  Avg Wind Speed:  {members['avg_wind_speed_ms'].mean():.1f} m/s")
    print(f"  Avg Fossil Fuel: {members['fossil_fuel_pct'].mean():.0f}%")
    print(f"  Avg Renewable Target: {members['renewable_target_pct'].mean():.0f}%")
    print()

# ============================================================================
# PART 8: KEY CONCEPTS SUMMARY
# ============================================================================

print("""
╔══════════════════════════════════════════════════════════════════════╗
║  KEY CONCEPTS SUMMARY                                              ║
╚══════════════════════════════════════════════════════════════════════╝

What yuh learn today:

1. UNSUPERVISED LEARNING = No labels, find patterns on yuh own
   - Like sorting people at Carnival without knowing who is who

2. K-MEANS CLUSTERING = Group data into K clusters
   - Place centroids → Assign points → Move centroids → Repeat

3. SCALING/STANDARDIZATION = Make all features equal
   - Without it, big numbers dominate di distance calculation

4. ELBOW METHOD = How fi choose K
   - Plot inertia vs K, look for di "elbow" bend

5. SILHOUETTE SCORE = How well-defined are di clusters
   - Higher = better separation between groups

6. CENTROIDS = Di center of each cluster
   - Represent di "average" member of di group

REAL WORLD USE: Caribbean governments could use clustering
to identify which islands have similar economic profiles
and should collaborate on shared policies!
""")

# ============================================================================
# QUIZ TIME!
# ============================================================================

print("""
╔══════════════════════════════════════════════════════════════════════╗
║  📝 QUIZ TIME! — 8 Questions                                      ║
╚══════════════════════════════════════════════════════════════════════╝

Answer these questions. Check yuh answers in quiz_answers.md

Q1: What is di KEY difference between supervised and unsupervised learning?
    a) Unsupervised is always better
    b) Unsupervised learning has no labeled data — it finds patterns on its own
    c) Supervised learning is older technology
    d) There is no real difference

Q2: In K-Means, what does K represent?
    a) The number of features
    b) The number of data points
    c) The number of clusters to create
    d) The accuracy score

Q3: Why do we STANDARDIZE (scale) the data before clustering?
    a) To make the algorithm faster
    b) To ensure all features contribute equally to distance calculations
    c) Because K-Means requires it by law
    d) To remove missing values

Q4: What is a CENTROID in K-Means?
    a) The first data point in the cluster
    b) The center point of a cluster (average position)
    c) The largest value in the cluster
    d) The outlier in the cluster

Q5: Di Elbow Method help yuh choose:
    a) Which features to use
    b) How to scale the data
    c) The optimal number of clusters (K)
    d) The best algorithm to use

Q6: If silhouette score is 0.8, dat mean:
    a) The clusters are poorly defined
    b) The clusters are well-separated and clearly defined
    c) You need more data
    d) K is too large

Q7: Which Caribbean nation yuh expect to cluster with tourism-heavy
    islands? (Think about which ones DEPEND most on tourism)
    a) Trinidad & Tobago (oil-rich)
    b) The Bahamas (tourism-dependent)
    c) Guyana (agriculture/oil)
    d) Haiti (developing economy)

Q8: After clustering, yuh notice Haiti alone in a cluster. Why might dat be?
    a) The algorithm is broken
    b) Haiti has a very different economic profile (low GDP, high agriculture,
       large population) compared to other Caribbean nations
    c) Haiti is not a Caribbean country
    d) K is too large
""")

print("""
╔══════════════════════════════════════════════════════════════════════╗
║  🏆 BONUS CHALLENGE                                                ║
╚══════════════════════════════════════════════════════════════════════╝

Challenge 1: Try K=3 and K=5 — how do di clusters change?

Challenge 2: Add more features to the economic data
  (e.g., education spending %, healthcare spending %, internet access %)

Challenge 3: Research REAL GDP data for 5 Caribbean countries and
  see if yuh clusters match what yuh created with synthetic data.

Challenge 4: Cluster Caribbean islands by marine/fishing profile:
  fish catch per capita, coral reef area, marine protected area %

Next lesson: Regression — predicting NUMBERS instead of categories!

Walk good! 🌴
""")
