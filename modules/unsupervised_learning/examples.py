"""
Unsupervised Learning Examples - Caribbean AI Curriculum
Designed by Adrian Dunkley (Adriandunkley.net) | FREE

All examples use synthetic Caribbean data and are fully runnable.
Each example demonstrates a different unsupervised learning technique
relevant to the Caribbean region.

Requirements:
    pip install numpy pandas scikit-learn matplotlib
"""

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans, DBSCAN
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score
import warnings
warnings.filterwarnings('ignore')

# ============================================================================
# EXAMPLE 1: K-Means on Caribbean Economic Indicators
# ============================================================================

def kmeans_caribbean_economies():
    """
    Cluster Caribbean nations by economic indicators using K-Means.

    Caribbean Context:
        The Caribbean region includes nations with diverse economic profiles.
        Some are tourism-heavy, others rely on oil and gas (Trinidad),
        agriculture, or financial services. Clustering reveals natural
        economic groupings that can inform regional policy.

    Features per country:
        - GDP per capita (USD)
        - Tourism as % of GDP
        - Agriculture as % of GDP
        - Unemployment rate (%)
        - Population (thousands)
        - Human Development Index (HDI)
        - Debt to GDP ratio (%)
    """
    print("=" * 70)
    print("EXAMPLE 1: K-Means Clustering of Caribbean Economies")
    print("=" * 70)

    # Synthetic data inspired by real Caribbean economic indicators
    countries = [
        'Jamaica', 'Trinidad & Tobago', 'Barbados', 'Bahamas',
        'Guyana', 'Suriname', 'Belize', 'Haiti',
        'Dominican Republic', 'Cuba', 'Antigua & Barbuda',
        'St. Lucia', 'Grenada', 'Dominica', 'St. Kitts & Nevis',
        'St. Vincent', 'Cayman Islands', 'Bermuda',
        'Turks & Caicos', 'Curacao'
    ]

    np.random.seed(42)
    data = {
        'country': countries,
        'gdp_per_capita_usd': [
            5500, 17000, 16000, 32000, 6900, 6200, 4800, 1300,
            8500, 9000, 17000, 11000, 10000, 8000, 19000,
            7500, 73000, 85000, 30000, 18000
        ],
        'tourism_pct_gdp': [
            10, 8, 35, 45, 3, 2, 15, 5,
            18, 10, 55, 42, 25, 20, 30,
            22, 50, 12, 60, 15
        ],
        'agriculture_pct_gdp': [
            7, 1, 2, 1, 15, 10, 12, 22,
            6, 4, 2, 3, 7, 15, 1,
            8, 0.5, 0.3, 1, 1
        ],
        'unemployment_pct': [
            8, 4, 10, 12, 12, 8, 9, 15,
            6, 3, 11, 15, 12, 14, 5,
            18, 3, 5, 8, 13
        ],
        'population_thousands': [
            2960, 1400, 287, 390, 790, 590, 400, 11400,
            10800, 11300, 97, 183, 112, 72, 53,
            110, 65, 64, 45, 160
        ],
        'hdi': [
            0.73, 0.80, 0.81, 0.81, 0.68, 0.72, 0.72, 0.51,
            0.76, 0.78, 0.78, 0.76, 0.77, 0.74, 0.78,
            0.73, 0.89, 0.91, 0.82, 0.81
        ],
        'debt_to_gdp_pct': [
            95, 50, 120, 55, 55, 70, 60, 30,
            55, 25, 85, 65, 60, 75, 60,
            80, 10, 15, 20, 45
        ]
    }

    df = pd.DataFrame(data)
    print(f"\nCaribbean Economic Dataset ({len(df)} countries):")
    print(df.to_string(index=False))

    # Prepare features (exclude country name)
    features = df.drop('country', axis=1)

    # Scale features — CRUCIAL for K-Means!
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(features)

    # Find optimal K using elbow method
    print("\n--- Finding Optimal Number of Clusters (Elbow Method) ---")
    inertias = []
    silhouettes = []
    K_range = range(2, 8)

    for k in K_range:
        km = KMeans(n_clusters=k, random_state=42, n_init=10)
        km.fit(X_scaled)
        inertias.append(km.inertia_)
        silhouettes.append(silhouette_score(X_scaled, km.labels_))
        print(f"  K={k}: Inertia={km.inertia_:.1f}, Silhouette={silhouettes[-1]:.3f}")

    best_k = list(K_range)[np.argmax(silhouettes)]
    print(f"\nBest K by Silhouette Score: {best_k}")

    # Final clustering
    km = KMeans(n_clusters=best_k, random_state=42, n_init=10)
    df['cluster'] = km.fit_predict(X_scaled)

    print(f"\n--- Caribbean Economy Clusters (K={best_k}) ---")
    for cluster_id in range(best_k):
        cluster_countries = df[df['cluster'] == cluster_id]['country'].tolist()
        cluster_data = df[df['cluster'] == cluster_id]
        avg_gdp = cluster_data['gdp_per_capita_usd'].mean()
        avg_tourism = cluster_data['tourism_pct_gdp'].mean()
        avg_agri = cluster_data['agriculture_pct_gdp'].mean()
        print(f"\n  Cluster {cluster_id + 1}:")
        print(f"    Countries: {', '.join(cluster_countries)}")
        print(f"    Avg GDP/capita: ${avg_gdp:,.0f}")
        print(f"    Avg Tourism %: {avg_tourism:.1f}%")
        print(f"    Avg Agriculture %: {avg_agri:.1f}%")

    # PCA for visualization (reduce to 2D)
    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(X_scaled)
    print(f"\n  PCA variance explained: PC1={pca.explained_variance_ratio_[0]:.2%}, "
          f"PC2={pca.explained_variance_ratio_[1]:.2%}")

    print("\n  2D PCA Projection (for visualization):")
    for i, country in enumerate(df['country']):
        print(f"    {country:25s} -> ({X_pca[i, 0]:+.2f}, {X_pca[i, 1]:+.2f})  "
              f"Cluster {df['cluster'].iloc[i] + 1}")

    return df


# ============================================================================
# EXAMPLE 2: PCA on Caribbean Demographic Data
# ============================================================================

def pca_caribbean_demographics():
    """
    Apply PCA to reduce and understand Caribbean demographic data.

    Caribbean Context:
        Caribbean nations share cultural ties but have diverse demographics.
        With many indicators available (health, education, economy, etc.),
        PCA helps identify the key dimensions that differentiate Caribbean
        societies.

    Features (15 dimensions per country):
        Health, education, economic, social, and environmental indicators.
    """
    print("\n" + "=" * 70)
    print("EXAMPLE 2: PCA on Caribbean Demographic Data")
    print("=" * 70)

    np.random.seed(123)

    countries = [
        'Jamaica', 'Trinidad', 'Barbados', 'Bahamas', 'Guyana',
        'Haiti', 'Dominican Rep.', 'Cuba', 'Antigua', 'St. Lucia',
        'Grenada', 'Dominica', 'St. Kitts', 'St. Vincent', 'Belize',
        'Suriname', 'Cayman Islands', 'Bermuda', 'Curacao', 'Aruba'
    ]

    n = len(countries)

    # Generate 15 demographic indicators
    data = {
        'life_expectancy': np.random.normal(74, 5, n),
        'infant_mortality_per_1000': np.random.normal(15, 8, n),
        'literacy_rate': np.clip(np.random.normal(90, 8, n), 50, 100),
        'secondary_enrollment_pct': np.clip(np.random.normal(80, 12, n), 40, 100),
        'gdp_per_capita': np.random.normal(12000, 8000, n),
        'gini_coefficient': np.clip(np.random.normal(0.40, 0.08, n), 0.25, 0.60),
        'urbanization_pct': np.clip(np.random.normal(55, 15, n), 20, 100),
        'internet_penetration': np.clip(np.random.normal(60, 20, n), 10, 98),
        'physicians_per_1000': np.clip(np.random.normal(1.5, 0.8, n), 0.2, 4),
        'renewable_energy_pct': np.clip(np.random.normal(15, 10, n), 1, 60),
        'co2_per_capita_tonnes': np.clip(np.random.normal(5, 3, n), 0.5, 15),
        'youth_unemployment_pct': np.clip(np.random.normal(25, 10, n), 5, 50),
        'remittances_pct_gdp': np.clip(np.random.normal(10, 8, n), 0, 35),
        'tourism_arrivals_per_capita': np.clip(np.random.normal(2, 2, n), 0.1, 10),
        'forest_coverage_pct': np.clip(np.random.normal(40, 15, n), 5, 80)
    }

    df = pd.DataFrame(data, index=countries)
    print(f"\nDemographic dataset: {df.shape[0]} countries x {df.shape[1]} features")
    print(f"\nFeatures: {list(df.columns)}")
    print(f"\nSample data (first 5 countries):\n{df.head().round(1)}")

    # Standardize
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(df)

    # Apply PCA
    pca_full = PCA()
    pca_full.fit(X_scaled)

    # Show variance explained
    print("\n--- Variance Explained by Each Principal Component ---")
    cumulative = 0
    for i, (var, cum_var) in enumerate(zip(
        pca_full.explained_variance_ratio_,
        np.cumsum(pca_full.explained_variance_ratio_)
    )):
        cumulative = cum_var
        bar = "=" * int(var * 100)
        print(f"  PC{i+1:2d}: {var:.3f} ({var*100:.1f}%)  Cumulative: {cumulative:.3f} ({cumulative*100:.1f}%)  {bar}")
        if cumulative > 0.95:
            print(f"  ... (remaining components explain < 5% total)")
            break

    # How many components for 80% variance?
    n_components_80 = np.argmax(np.cumsum(pca_full.explained_variance_ratio_) >= 0.80) + 1
    n_components_90 = np.argmax(np.cumsum(pca_full.explained_variance_ratio_) >= 0.90) + 1
    print(f"\n  Components for 80% variance: {n_components_80}")
    print(f"  Components for 90% variance: {n_components_90}")
    print(f"  Reduced from {df.shape[1]} features to {n_components_80} (80%) or {n_components_90} (90%)")

    # Interpret first 2 PCs
    pca_2d = PCA(n_components=2)
    X_2d = pca_2d.fit_transform(X_scaled)

    print("\n--- Principal Component Interpretation ---")
    for pc_idx in range(2):
        print(f"\n  PC{pc_idx+1} (explains {pca_2d.explained_variance_ratio_[pc_idx]:.1%} of variance):")
        loadings = pd.Series(pca_2d.components_[pc_idx], index=df.columns)
        top_positive = loadings.nlargest(3)
        top_negative = loadings.nsmallest(3)
        print(f"    Top positive loadings (higher value -> higher PC{pc_idx+1}):")
        for feat, val in top_positive.items():
            print(f"      {feat:35s} {val:+.3f}")
        print(f"    Top negative loadings (higher value -> lower PC{pc_idx+1}):")
        for feat, val in top_negative.items():
            print(f"      {feat:35s} {val:+.3f}")

    # Show countries in 2D PCA space
    print("\n--- Countries in PCA Space ---")
    print(f"{'Country':25s} {'PC1':>8s} {'PC2':>8s}")
    print("-" * 43)
    for i, country in enumerate(countries):
        print(f"{country:25s} {X_2d[i, 0]:+8.2f} {X_2d[i, 1]:+8.2f}")

    return pca_2d, X_2d


# ============================================================================
# EXAMPLE 3: DBSCAN on Caribbean Earthquake Data
# ============================================================================

def dbscan_caribbean_earthquakes():
    """
    Cluster Caribbean earthquake epicenters using DBSCAN.

    Caribbean Context:
        The Caribbean sits on the Caribbean Plate, surrounded by subduction
        zones and transform faults. Earthquakes cluster along plate boundaries.
        DBSCAN is ideal because these clusters follow irregular, elongated
        fault lines rather than circular shapes.

    Features:
        - Latitude
        - Longitude
        - Depth (km)
        - Magnitude
    """
    print("\n" + "=" * 70)
    print("EXAMPLE 3: DBSCAN on Caribbean Earthquake Data")
    print("=" * 70)

    np.random.seed(456)

    earthquakes = []

    # Zone 1: Puerto Rico Trench (north of Puerto Rico) — deep earthquakes
    n1 = 80
    lat1 = np.random.normal(19.5, 0.3, n1)
    lon1 = np.random.normal(-66.5, 1.0, n1)
    depth1 = np.random.normal(100, 30, n1)
    mag1 = np.random.exponential(1.5, n1) + 2.5
    for i in range(n1):
        earthquakes.append({'lat': lat1[i], 'lon': lon1[i], 'depth_km': depth1[i],
                           'magnitude': mag1[i], 'true_zone': 'Puerto Rico Trench'})

    # Zone 2: Lesser Antilles subduction (eastern Caribbean arc)
    n2 = 100
    t = np.linspace(0, 1, n2)
    lat2 = 10.5 + t * 7.5 + np.random.normal(0, 0.2, n2)
    lon2 = -61.5 + t * 0.5 + np.random.normal(0, 0.3, n2)
    depth2 = np.random.normal(80, 25, n2)
    mag2 = np.random.exponential(1.2, n2) + 2.0
    for i in range(n2):
        earthquakes.append({'lat': lat2[i], 'lon': lon2[i], 'depth_km': depth2[i],
                           'magnitude': mag2[i], 'true_zone': 'Lesser Antilles Arc'})

    # Zone 3: Jamaica / Cayman Trough (transform fault)
    n3 = 60
    lat3 = np.random.normal(18.5, 0.3, n3)
    lon3 = np.random.uniform(-79, -76, n3)
    depth3 = np.random.normal(15, 8, n3)
    mag3 = np.random.exponential(1.0, n3) + 2.0
    for i in range(n3):
        earthquakes.append({'lat': lat3[i], 'lon': lon3[i], 'depth_km': depth3[i],
                           'magnitude': mag3[i], 'true_zone': 'Cayman Trough'})

    # Noise: random earthquakes scattered around the Caribbean
    n_noise = 30
    lat_noise = np.random.uniform(10, 22, n_noise)
    lon_noise = np.random.uniform(-85, -59, n_noise)
    depth_noise = np.random.uniform(5, 200, n_noise)
    mag_noise = np.random.exponential(0.8, n_noise) + 1.5
    for i in range(n_noise):
        earthquakes.append({'lat': lat_noise[i], 'lon': lon_noise[i],
                           'depth_km': depth_noise[i], 'magnitude': mag_noise[i],
                           'true_zone': 'Scattered'})

    df = pd.DataFrame(earthquakes)
    df['depth_km'] = np.clip(df['depth_km'], 1, 300)
    df['magnitude'] = np.clip(df['magnitude'], 1.5, 8.0)

    print(f"\nEarthquake dataset: {len(df)} events")
    print(f"\nTrue tectonic zones:\n{df['true_zone'].value_counts()}")
    print(f"\nSample data:\n{df.head(10).round(2)}")

    # Prepare features for DBSCAN
    features = df[['lat', 'lon', 'depth_km']].copy()
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(features)

    # Apply DBSCAN with parameter search
    print("\n--- DBSCAN Parameter Search ---")
    for eps in [0.4, 0.5, 0.6, 0.8]:
        for min_samples in [5, 8, 10]:
            db = DBSCAN(eps=eps, min_samples=min_samples)
            labels = db.fit_predict(X_scaled)
            n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
            n_noise_pts = list(labels).count(-1)
            if n_clusters > 1:
                sil = silhouette_score(X_scaled[labels != -1], labels[labels != -1])
            else:
                sil = 0
            print(f"  eps={eps}, min_samples={min_samples}: "
                  f"{n_clusters} clusters, {n_noise_pts} noise points, silhouette={sil:.3f}")

    # Use good parameters
    db = DBSCAN(eps=0.6, min_samples=8)
    df['cluster'] = db.fit_predict(X_scaled)

    n_clusters = len(set(df['cluster'])) - (1 if -1 in df['cluster'].values else 0)
    n_noise_final = (df['cluster'] == -1).sum()

    print(f"\n--- DBSCAN Results (eps=0.6, min_samples=8) ---")
    print(f"  Clusters found: {n_clusters}")
    print(f"  Noise points: {n_noise_final}")

    for cluster_id in sorted(df['cluster'].unique()):
        cluster_data = df[df['cluster'] == cluster_id]
        label = "NOISE" if cluster_id == -1 else f"Cluster {cluster_id}"
        print(f"\n  {label} ({len(cluster_data)} earthquakes):")
        print(f"    Lat range:  {cluster_data['lat'].min():.1f} to {cluster_data['lat'].max():.1f}")
        print(f"    Lon range:  {cluster_data['lon'].min():.1f} to {cluster_data['lon'].max():.1f}")
        print(f"    Avg depth:  {cluster_data['depth_km'].mean():.1f} km")
        print(f"    Avg magnitude: {cluster_data['magnitude'].mean():.1f}")
        if cluster_id != -1:
            true_zones = cluster_data['true_zone'].value_counts()
            dominant = true_zones.index[0]
            print(f"    Dominant true zone: {dominant} ({true_zones.iloc[0]}/{len(cluster_data)})")

    return df


# ============================================================================
# EXAMPLE 4: Market Basket Analysis on Caribbean Grocery Patterns
# ============================================================================

def market_basket_caribbean():
    """
    Discover item associations in Caribbean grocery shopping.

    Caribbean Context:
        Caribbean cuisine has distinctive ingredient combinations.
        Ackee and saltfish in Jamaica, doubles in Trinidad, roti and
        curry everywhere. Market basket analysis reveals these cultural
        food patterns from transaction data.

    This example uses a simple implementation (no mlxtend required).
    """
    print("\n" + "=" * 70)
    print("EXAMPLE 4: Market Basket Analysis — Caribbean Grocery Patterns")
    print("=" * 70)

    np.random.seed(789)

    # Define Caribbean grocery items and common combinations
    all_items = [
        'Rice', 'Red Peas', 'Coconut Milk', 'Chicken', 'Scotch Bonnet',
        'Ackee (canned)', 'Saltfish', 'Breadfruit', 'Yam', 'Plantain',
        'Curry Powder', 'Roti Skin', 'Rum', 'Lime', 'Sugar',
        'Jerk Seasoning', 'Thyme', 'Garlic', 'Onion', 'Tomato',
        'Flour', 'Butter', 'Condensed Milk', 'Sorrel', 'Ginger Beer',
        'Bammie', 'Callaloo', 'Dasheen', 'Green Banana', 'Pepper Sauce'
    ]

    # Define common Caribbean shopping patterns (baskets)
    basket_templates = [
        # Jamaican Sunday dinner
        (['Rice', 'Red Peas', 'Coconut Milk', 'Chicken', 'Thyme'], 0.15),
        # Ackee and saltfish breakfast
        (['Ackee (canned)', 'Saltfish', 'Breadfruit', 'Scotch Bonnet'], 0.12),
        # Trinidad roti
        (['Curry Powder', 'Roti Skin', 'Chicken', 'Scotch Bonnet', 'Garlic'], 0.10),
        # Jerk chicken prep
        (['Jerk Seasoning', 'Chicken', 'Scotch Bonnet', 'Thyme', 'Lime'], 0.10),
        # Rum cocktails
        (['Rum', 'Lime', 'Sugar', 'Ginger Beer'], 0.08),
        # Ground provisions
        (['Yam', 'Plantain', 'Green Banana', 'Dasheen', 'Bammie'], 0.08),
        # Baking
        (['Flour', 'Butter', 'Sugar', 'Condensed Milk'], 0.07),
        # Christmas sorrel
        (['Sorrel', 'Rum', 'Sugar', 'Ginger Beer'], 0.05),
        # Callaloo
        (['Callaloo', 'Coconut Milk', 'Garlic', 'Scotch Bonnet', 'Onion'], 0.05),
    ]

    # Generate transactions
    n_transactions = 1000
    transactions = []

    for _ in range(n_transactions):
        basket = set()
        for items, prob in basket_templates:
            if np.random.random() < prob:
                for item in items:
                    if np.random.random() < 0.85:
                        basket.add(item)
        # Add some random items
        n_random = np.random.poisson(2)
        for _ in range(n_random):
            basket.add(np.random.choice(all_items))
        if len(basket) >= 2:
            transactions.append(list(basket))

    print(f"\nGenerated {len(transactions)} transactions")
    print(f"Average basket size: {np.mean([len(t) for t in transactions]):.1f} items")
    print(f"\nSample transactions:")
    for i in range(5):
        print(f"  Transaction {i+1}: {transactions[i]}")

    # Calculate item frequencies
    item_counts = {}
    for trans in transactions:
        for item in trans:
            item_counts[item] = item_counts.get(item, 0) + 1

    print(f"\n--- Item Frequency (Top 15) ---")
    sorted_items = sorted(item_counts.items(), key=lambda x: x[1], reverse=True)[:15]
    for item, count in sorted_items:
        support = count / len(transactions)
        bar = "=" * int(support * 100)
        print(f"  {item:20s} {count:4d} ({support:.2%}) {bar}")

    # Simple association rule mining (pairs)
    print(f"\n--- Association Rules (Item Pairs) ---")
    print(f"{'Rule':50s} {'Support':>8s} {'Confidence':>10s} {'Lift':>6s}")
    print("-" * 78)

    pair_counts = {}
    for trans in transactions:
        items = sorted(set(trans))
        for i in range(len(items)):
            for j in range(i + 1, len(items)):
                pair = (items[i], items[j])
                pair_counts[pair] = pair_counts.get(pair, 0) + 1

    n_trans = len(transactions)
    rules = []

    for (item_a, item_b), count in pair_counts.items():
        support = count / n_trans
        if support < 0.03:
            continue
        conf_ab = count / item_counts.get(item_a, 1)
        conf_ba = count / item_counts.get(item_b, 1)
        p_b = item_counts.get(item_b, 1) / n_trans
        p_a = item_counts.get(item_a, 1) / n_trans
        lift_ab = conf_ab / p_b if p_b > 0 else 0
        lift_ba = conf_ba / p_a if p_a > 0 else 0

        if conf_ab > 0.3 and lift_ab > 1.5:
            rules.append((item_a, item_b, support, conf_ab, lift_ab))
        if conf_ba > 0.3 and lift_ba > 1.5:
            rules.append((item_b, item_a, support, conf_ba, lift_ba))

    rules.sort(key=lambda x: x[4], reverse=True)

    for item_a, item_b, supp, conf, lift in rules[:20]:
        rule_str = f"{item_a} -> {item_b}"
        print(f"  {rule_str:48s} {supp:>7.2%} {conf:>9.2%} {lift:>6.2f}")

    print("\n--- Caribbean Shopping Insights ---")
    print("  Strong associations reveal Caribbean food culture:")
    if rules:
        top_rule = rules[0]
        print(f"  Strongest rule: {top_rule[0]} -> {top_rule[1]}")
        print(f"    Lift of {top_rule[4]:.2f} means buying {top_rule[0]}")
        print(f"    makes you {top_rule[4]:.1f}x more likely to buy {top_rule[1]}")
    print("\n  These patterns reflect traditional Caribbean dishes:")
    print("    - Ackee + Saltfish (Jamaica's national dish)")
    print("    - Rice + Red Peas + Coconut Milk (Sunday staple)")
    print("    - Curry Powder + Roti Skin (Trinidad influence)")
    print("    - Rum + Lime + Sugar (Caribbean cocktail culture)")

    return rules


# ============================================================================
# MAIN — Run All Examples
# ============================================================================

def main():
    """Run all Caribbean unsupervised learning examples."""
    print("+" * 70)
    print("+     Caribbean AI Curriculum — Unsupervised Learning Examples       +")
    print("+     Designed by Adrian Dunkley (Adriandunkley.net) | FREE          +")
    print("+" * 70)
    print()
    print("Four examples featuring real Caribbean scenarios:")
    print("  1. K-Means Clustering of Caribbean Economies")
    print("  2. PCA on Caribbean Demographic Data")
    print("  3. DBSCAN on Caribbean Earthquake Data")
    print("  4. Market Basket Analysis on Caribbean Grocery Patterns")
    print()

    kmeans_caribbean_economies()
    pca_caribbean_demographics()
    dbscan_caribbean_earthquakes()
    market_basket_caribbean()

    print("\n" + "=" * 70)
    print("All unsupervised learning examples complete!")
    print("The Caribbean has so much hidden structure waiting to be discovered.")
    print("Try these techniques on yuh own data — economic, environmental,")
    print("cultural — and see what patterns emerge!")
    print("=" * 70)


if __name__ == "__main__":
    main()
