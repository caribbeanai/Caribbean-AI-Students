# Unsupervised Learning - Caribbean AI Curriculum

**Designed by Adrian Dunkley ([Adriandunkley.net](https://Adriandunkley.net)) | FREE Curriculum**

---

## Welcome!

So yuh learn bout supervised learning already? Good. Now we movin into unsupervised learning — where we doh have labels, and di machine haffi find patterns on its own. Think of it like exploring a new Caribbean island without a map. Yuh haffi discover di landscape by yuhself!

---

## Table of Contents

1. [What is Unsupervised Learning?](#what-is-unsupervised-learning)
2. [Clustering](#clustering)
3. [Dimensionality Reduction](#dimensionality-reduction)
4. [Anomaly Detection](#anomaly-detection)
5. [Association Rules](#association-rules)
6. [Topic Modeling](#topic-modeling)
7. [Caribbean Applications](#caribbean-applications)
8. [Practical Guidance](#practical-guidance)
9. [Quiz](#quiz)

---

## What is Unsupervised Learning?

In supervised learning, somebody (the "supervisor") gives yuh the answers — labeled data. In **unsupervised learning**, yuh on yuh own. The machine gets raw data without labels and must discover hidden structure.

### The Analogy

Imagine yuh arrive at a Caribbean Carnival for the first time. Nobody tells yuh which masqueraders belong to which band. But yuh start noticing patterns:
- This group wearing blue and gold costumes — they must be together
- That group has feathered headdresses — another band
- Over there, they playing steelpan — different vibes entirely

Yuh just did **clustering** — grouping similar things without being told the groups!

### Supervised vs. Unsupervised

| Aspect | Supervised | Unsupervised |
|--------|-----------|--------------|
| **Labels** | Has labels (y) | No labels |
| **Goal** | Predict known outcomes | Discover hidden patterns |
| **Example** | "This is a snapper" | "These fish seem similar" |
| **Evaluation** | Clear metrics (accuracy, R²) | Harder to evaluate |
| **Caribbean analogy** | Teacher grading CXC papers | Exploring unknown reef |

---

## Clustering

### What is Clustering?

Clustering groups similar data points together. The algorithm decides what "similar" means based on the features yuh provide.

### K-Means Clustering

**The most popular clustering algorithm.** Simple but powerful.

**How it works — the Dancehall analogy:**

Imagine three DJs at a Caribbean street party, each playing from a different speaker. People naturally gravitate toward the music they like:

1. **Initialize:** Place K speakers (centroids) at random locations
2. **Assign:** Each person goes to their nearest speaker (assign points to nearest centroid)
3. **Update:** Each DJ moves their speaker to the center of their crowd (recalculate centroids)
4. **Repeat:** People reshuffle, DJs move again, until nobody changes position

```
Step 1: Random centroids     Step 2: Assign points     Step 3: Update centroids
    *                           *  ···                     ·
  ·   ·                       ·   ·  ···                 · * ·
·   *   ·                   ···  *  ···                ···  ·  ···
  ·   ·                       ·   ·  ···                 · * ·
    *                           *  ···                     ·
                                                      (Repeat until stable)
```

**Choosing K — The Elbow Method:**

Plot the "within-cluster sum of squares" (WCSS) for different K values. The "elbow" where the curve bends is a good K:

```
WCSS
  |\
  | \
  |  \
  |   \___
  |       \____
  |            \________
  |________________________ K
       ^ Elbow (optimal K)
```

### Caribbean Example: Clustering Islands by Similarity

Group Caribbean islands based on:
- GDP per capita
- Population
- Land area
- Tourism percentage of GDP
- Agriculture percentage of GDP
- Distance from mainland

You might find clusters like:
- **Cluster 1:** Large, tourism-heavy (Bahamas, Barbados, Antigua)
- **Cluster 2:** Large, diverse economies (Jamaica, Trinidad, Dominican Republic)
- **Cluster 3:** Small, developing (Dominica, St. Vincent, Grenada)

### DBSCAN (Density-Based Spatial Clustering)

**K-Means assumes round clusters. But what if yuh data has weird shapes?**

DBSCAN finds clusters based on **density** — regions where points are packed closely together.

**The Caribbean party analogy:** Instead of fixed speakers, DBSCAN says "wherever people are packed tight, that's a party." Lone people standing far from any group are labeled as outliers (noise).

**Key parameters:**
- **eps (epsilon):** How close points must be to be "neighbors" (the party radius)
- **min_samples:** Minimum people to form a "party" (cluster)

**Caribbean use case:** Clustering earthquake epicenters around the Caribbean plate boundary. Earthquakes cluster along fault lines — DBSCAN finds these naturally shaped clusters.

### Hierarchical Clustering

**Builds a tree (dendrogram) of clusters.** Like tracing Caribbean family trees — start with individuals, group siblings, then cousins, then extended family.

**Two approaches:**
- **Agglomerative (bottom-up):** Start with each point as its own cluster, merge closest pairs
- **Divisive (top-down):** Start with one big cluster, recursively split

```
Dendrogram (Caribbean music genres):

    ┌────────────────┐
    │                │
  ┌─┴─┐          ┌──┴──┐
  │   │          │     │
Reggae Dancehall  Soca  Calypso
  (Jamaica)      (Trinidad)

Shows that Reggae/Dancehall are more similar to each other
than to Soca/Calypso, but all are Caribbean music.
```

---

## Dimensionality Reduction

### The Problem

Caribbean demographic data might have 50 features per island. How yuh visualize 50 dimensions? Yuh can't! We need to reduce dimensions while keeping the important information.

### PCA (Principal Component Analysis)

**PCA finds the directions of maximum variance in yuh data.**

**The Carnival parade analogy:** Imagine photographing a Carnival parade. If yuh stand at the side, yuh see costumes spreading left-to-right (maximum variance). If yuh stand at the end, everyone is stacked on top of each other (minimum variance). PCA finds the best "viewing angle" to see the most spread.

**How it works:**
1. Center the data (subtract the mean)
2. Find the direction of maximum variance → **PC1** (first principal component)
3. Find the next direction, perpendicular to PC1 → **PC2**
4. Continue for as many components as needed

**Mathematical intuition:**
```
Original: 50 features per Caribbean island

PCA reduces to:
  PC1 = 0.4×GDP + 0.3×Tourism + 0.2×Population + ... (economic strength)
  PC2 = 0.5×Agriculture + 0.3×Rainfall + ...          (agricultural profile)

Now you can plot islands on a 2D map using PC1 and PC2!
```

### t-SNE and UMAP

For **visualization** of high-dimensional data:
- **t-SNE:** Preserves local structure (nearby points stay nearby)
- **UMAP:** Faster, preserves both local and global structure

Caribbean use: Visualizing clusters of Caribbean music songs based on audio features.

---

## Anomaly Detection

### What is Anomaly Detection?

Finding data points that don't fit the pattern — the outliers, the unusual, the suspicious.

### Caribbean Financial Transaction Monitoring

The Caribbean banking sector must comply with international anti-money laundering (AML) regulations. Anomaly detection helps identify suspicious transactions:

**Normal patterns:**
- Regular salary deposits every two weeks
- Grocery shopping at local supermarkets
- Utility bill payments

**Anomalies (potentially suspicious):**
- Sudden large cash deposits
- Transactions at unusual hours
- Multiple transfers to unknown overseas accounts
- Spending patterns that suddenly change

### Methods

1. **Statistical methods:** Points far from the mean (z-score > 3)
2. **Isolation Forest:** Randomly partitions data — anomalies are isolated quickly
3. **Local Outlier Factor (LOF):** Compares local density — anomalies are in sparse regions
4. **Autoencoders:** Neural networks trained to reconstruct normal data — anomalies have high reconstruction error

### Other Caribbean Anomaly Detection Uses

- **Hurricane track anomalies:** Detecting unusual storm paths
- **Coral reef monitoring:** Spotting unusual changes in reef health indicators
- **Power grid monitoring:** Detecting electricity theft or equipment failure
- **Tourism patterns:** Identifying unusual drops/spikes in visitor arrivals

---

## Association Rules

### Market Basket Analysis

**What items do Caribbean shoppers buy together?**

When yuh go to Hi-Lo, PriceSmart, or the local market, certain items go together:

- **Ackee + Saltfish + Breadfruit** (classic Jamaican breakfast)
- **Curry powder + Chicken + Roti skin** (Trinidad curry)
- **Rice + Peas + Coconut milk** (Sunday dinner staples)
- **Rum + Lime + Sugar** (you know what that is!)

### Key Concepts

- **Support:** How often items appear together → `Support(A,B) = P(A ∩ B)`
- **Confidence:** If you buy A, what's the probability you buy B? → `Confidence(A→B) = P(B|A)`
- **Lift:** Does buying A actually increase the chance of buying B? → `Lift(A→B) = Confidence / P(B)`

```
Rule: {Ackee} → {Saltfish}
  Support:    0.15 (15% of all baskets contain both)
  Confidence: 0.85 (85% of ackee buyers also buy saltfish)
  Lift:       3.2  (ackee buyers are 3.2x more likely to buy saltfish)
```

**Application:** Caribbean grocery stores use this for product placement, promotions, and recommendation systems.

---

## Topic Modeling

### Discovering Topics in Caribbean News

Given thousands of Caribbean news articles, what are the main topics?

**LDA (Latent Dirichlet Allocation)** discovers topics automatically:

```
Topic 1: hurricane, storm, damage, category, wind, evacuation
  → Natural Disasters

Topic 2: tourism, hotel, visitor, cruise, beach, resort
  → Tourism Industry

Topic 3: cricket, windies, test, batting, bowling, CPL
  → Cricket/Sports

Topic 4: election, parliament, minister, policy, vote, opposition
  → Politics

Topic 5: carnival, soca, calypso, mas, steelpan, fete
  → Culture/Carnival
```

Each document is a **mixture of topics:**
- An article about a hurricane hitting a tourist area might be 60% Disasters + 30% Tourism + 10% Politics

---

## Caribbean Applications

### Economics & Development
- **Clustering Caribbean economies:** Group islands by economic indicators to identify peer groups for policy comparison
- **Trade pattern analysis:** Discover hidden trade clusters among Caribbean nations
- **Economic indicator reduction:** Use PCA to create composite economic indices

### Environment
- **Earthquake clustering:** DBSCAN on Caribbean seismic data to identify active fault zones
- **Climate zone identification:** Cluster Caribbean microclimates for agricultural planning
- **Marine ecosystem analysis:** Group coral reef sites by health indicators

### Culture & Society
- **Music genre discovery:** Cluster Caribbean music by audio features
- **Dialect analysis:** Group Caribbean English dialects by linguistic features
- **Migration pattern analysis:** Discover Caribbean migration clusters

### Sports
- **Cricket player profiling:** Cluster West Indies cricketers by playing style
- **Track & field talent identification:** Group athletes by performance profiles
- **Team strategy analysis:** Cluster CPL teams by playing strategies

### Healthcare
- **Disease pattern discovery:** Cluster Caribbean regions by health outcomes
- **Patient segmentation:** Group patients for targeted health interventions
- **Pharmaceutical spending patterns:** Discover anomalous prescription patterns

---

## Practical Guidance

### When to Use What

| Method | Use When | Caribbean Example |
|--------|---------|-------------------|
| K-Means | You know roughly how many groups | Segment tourists into 5 types |
| DBSCAN | Clusters have irregular shapes | Earthquake epicenter mapping |
| Hierarchical | Want a tree of relationships | Caribbean language family tree |
| PCA | Too many features, need reduction | Simplify 50 economic indicators |
| t-SNE/UMAP | Need to visualize high-dim data | Plot Caribbean music similarity |
| Isolation Forest | Detect outliers | Fraud detection in banking |
| LDA | Discover topics in text | Caribbean news analysis |
| Apriori | Find item associations | Market basket analysis |

### Challenges

1. **No ground truth:** Without labels, how do you know if clusters are "right"?
   - Use metrics like Silhouette Score, Davies-Bouldin Index
   - Domain knowledge is crucial — do the clusters make sense to a Caribbean expert?

2. **Feature scaling:** Always scale features before clustering!
   - GDP in billions vs. population in millions → different scales dominate

3. **Curse of dimensionality:** Too many features make distances meaningless
   - Use PCA first, then cluster

4. **Choosing K:** Multiple methods exist — elbow, silhouette, gap statistic
   - There is no single "right answer" — use domain knowledge

---

## Quiz

Test yuh understanding of unsupervised learning!

### Question 1
What is the fundamental difference between supervised and unsupervised learning?

- A) Unsupervised learning is always more accurate
- B) Supervised learning uses labeled data; unsupervised discovers patterns without labels
- C) Unsupervised learning requires more data
- D) Supervised learning is only for classification

<details>
<summary>Answer</summary>
**B)** Supervised learning requires labeled training data (input-output pairs), while unsupervised learning works with unlabeled data to discover hidden structure and patterns.
</details>

### Question 2
In K-Means clustering, what does "K" represent?

- A) The number of features
- B) The number of data points
- C) The number of clusters to form
- D) The number of iterations

<details>
<summary>Answer</summary>
**C)** K is the number of clusters you want the algorithm to find. You must specify K before running the algorithm.
</details>

### Question 3
You want to cluster Caribbean earthquake data where earthquakes follow fault lines (non-circular patterns). Which algorithm is best?

- A) K-Means
- B) DBSCAN
- C) PCA
- D) Linear Regression

<details>
<summary>Answer</summary>
**B)** DBSCAN is density-based and can find clusters of arbitrary shape, making it ideal for earthquake data that follows irregular fault lines. K-Means assumes spherical clusters.
</details>

### Question 4
What does PCA do?

- A) Classifies data into categories
- B) Predicts future values
- C) Reduces the number of features while preserving maximum variance
- D) Finds outliers in data

<details>
<summary>Answer</summary>
**C)** PCA (Principal Component Analysis) reduces dimensionality by finding new axes (principal components) that capture the most variance in the data.
</details>

### Question 5
In market basket analysis, what does a "lift" value of 3.0 for the rule {Ackee} → {Saltfish} mean?

- A) Saltfish is 3 times more expensive than ackee
- B) Customers who buy ackee are 3 times more likely to buy saltfish than a random customer
- C) 3% of customers buy both
- D) The rule has 3 exceptions

<details>
<summary>Answer</summary>
**B)** A lift of 3.0 means that customers who buy ackee are 3 times more likely to also buy saltfish compared to the baseline probability of buying saltfish.
</details>

### Question 6
Why is feature scaling important before running K-Means on Caribbean economic data?

- A) It makes the algorithm run faster
- B) Features with larger numerical ranges (like GDP in billions) would dominate distance calculations over smaller-range features (like unemployment rate in percentages)
- C) It is not important
- D) It reduces the number of features

<details>
<summary>Answer</summary>
**B)** Without scaling, features with larger numerical ranges dominate the distance calculations. GDP in billions would completely overshadow a 0-10% unemployment rate, leading to meaningless clusters.
</details>

### Question 7
What type of unsupervised learning would help a Caribbean bank detect suspicious transactions?

- A) Clustering
- B) Dimensionality reduction
- C) Anomaly detection
- D) Topic modeling

<details>
<summary>Answer</summary>
**C)** Anomaly detection identifies data points that deviate significantly from normal patterns, making it ideal for detecting suspicious financial transactions.
</details>

### Question 8
Topic modeling (LDA) on Caribbean newspaper articles would most likely discover:

- A) The correct spelling of words
- B) Clusters of related words that represent recurring themes (politics, tourism, sports, etc.)
- C) The author of each article
- D) The publication date

<details>
<summary>Answer</summary>
**B)** LDA discovers latent topics — groups of words that frequently appear together — revealing recurring themes like politics, tourism, sports, culture, and weather.
</details>

### Question 9
The Silhouette Score measures:

- A) How many clusters exist
- B) How well-separated and cohesive clusters are
- C) The number of outliers
- D) The speed of the algorithm

<details>
<summary>Answer</summary>
**B)** The Silhouette Score ranges from -1 to 1 and measures how similar a point is to its own cluster compared to other clusters. Higher values mean better-defined clusters.
</details>

### Question 10
You run PCA on Caribbean demographic data with 30 features and find that the first 3 principal components explain 85% of the variance. What should you do?

- A) Use all 30 features anyway
- B) Use only the first 3 principal components for further analysis, since they capture most of the information
- C) Discard the dataset and collect new data
- D) Add more features

<details>
<summary>Answer</summary>
**B)** If 3 components explain 85% of the variance, they capture most of the meaningful information. Using 3 components instead of 30 simplifies analysis, visualization, and subsequent modeling while retaining the essential structure.
</details>

---

## Next Steps

- Explore the `examples.py` file for hands-on Caribbean unsupervised learning code
- Try clustering your own Caribbean data
- Combine with supervised learning — use cluster labels as features!
- Move on to **Reinforcement Learning** to learn about training agents

---

*Designed by Adrian Dunkley ([Adriandunkley.net](https://Adriandunkley.net)) | FREE Caribbean AI Curriculum*
*From Port of Spain to Bridgetown, Nassau to Kingston — we learning together!*
