# Supervised Learning - Caribbean AI Curriculum

**Designed by Adrian Dunkley ([Adriandunkley.net](https://Adriandunkley.net)) | FREE Curriculum**

---

## Welcome, Caribbean Learner!

Yuh ready fi learn bout supervised learning? This is one of di most powerful tools in di AI toolkit, and we going break it down Caribbean style. Whether yuh from Kingston to Port of Spain, Nassau to Bridgetown, dis module got yuh covered.

---

## Table of Contents

1. [What is Supervised Learning?](#what-is-supervised-learning)
2. [Classification](#classification)
3. [Regression](#regression)
4. [Key Algorithms](#key-algorithms)
5. [The Math Intuition](#the-math-intuition)
6. [Model Evaluation](#model-evaluation)
7. [Practical Guidance](#practical-guidance)
8. [Caribbean Applications](#caribbean-applications)
9. [Cricket and Sports Analytics](#cricket-and-sports-analytics)
10. [Quiz](#quiz)

---

## What is Supervised Learning?

Think of supervised learning like how yuh grandmother teach yuh to cook. She show yuh the ingredients (input features) and the finished dish (the label/output), and over time yuh learn the recipe (the model). That is supervised learning — learning from labeled examples.

**Formally:** Given a dataset of input-output pairs `{(x₁, y₁), (x₂, y₂), ..., (xₙ, yₙ)}`, the goal is to learn a function `f(x) → y` that maps inputs to outputs, so we can predict `y` for new, unseen `x` values.

### The Two Main Tasks

| Task | Output Type | Caribbean Example |
|------|------------|-------------------|
| **Classification** | Category/Class | Is this fish a snapper, grouper, or parrotfish? |
| **Regression** | Continuous Number | How much rainfall will St. Ann parish get next month? |

---

## Classification

### The Fish Market Analogy

Imagine yuh at the Coronation Market in Kingston, or the Fish Fry in Nassau. A fisherman bring in a catch and yuh need to sort the fish by species. Yuh look at features like:

- **Size** (length in cm)
- **Weight** (in kg)
- **Color** (silver, red, blue-green)
- **Fin shape** (rounded, pointed, forked)
- **Scale pattern** (smooth, rough, spiny)

After seeing hundreds of fish with labels from experienced fishermen, yuh brain learns patterns:
- Red color + medium size + large eyes → **Red Snapper**
- Large + dark spots + thick lips → **Grouper**
- Bright blue-green + beak-like mouth → **Parrotfish**

That is **classification** — learning to assign categories based on features.

### Types of Classification

1. **Binary Classification** — Two classes
   - Will this hurricane hit Jamaica? (Yes/No)
   - Is this transaction fraudulent? (Yes/No)
   - Will this student pass CXC? (Pass/Fail)

2. **Multi-class Classification** — More than two classes
   - Which Caribbean island is this music from? (Jamaica/Trinidad/Cuba/Barbados)
   - What type of Caribbean crop is this? (Sugarcane/Banana/Coffee/Cocoa)
   - What genre of Caribbean music? (Reggae/Soca/Calypso/Dancehall/Reggaeton)

3. **Multi-label Classification** — Multiple labels per instance
   - Tag a Caribbean news article (could be: Politics, Tourism, Sports, Weather)

### Decision Boundaries

When a classifier learns, it creates **decision boundaries** — invisible lines (or curves) that separate different classes in feature space.

```
Imagine a 2D plot:
  X-axis: Fish Length
  Y-axis: Fish Weight

       Weight
         |     * * *  (Grouper - large & heavy)
         |   * * *
         |
         |  . . .     (Snapper - medium)
         | . . . .
         |
         | + + +      (Parrotfish - small & light)
         |_____________ Length

The classifier draws boundaries between these clusters.
```

---

## Regression

### The Rainfall Prediction Analogy

Every Caribbean farmer knows that rain is life. Imagine yuh trying to predict how many millimeters of rain Portland parish going get next month. Yuh look at:

- **Current temperature** (°C)
- **Humidity** (%)
- **Month of the year** (hurricane season? dry season?)
- **Sea surface temperature** (°C)
- **El Niño/La Niña index**
- **Wind patterns**

Unlike classification (where yuh predict a category), here yuh predict a **continuous number** — maybe 150mm, maybe 250mm, maybe 80mm. That is **regression**.

### The Line of Best Fit

The simplest regression is **linear regression** — fitting a straight line through yuh data:

```
Rainfall (mm)
    |        *
    |      *   *
    |    *  /      *
    |   * /   *
    |  */  *
    | /  *
    |/ *
    |______________ Temperature (°C)

    The line (/) shows the learned relationship.
    y = mx + b
    Rainfall = slope × Temperature + intercept
```

### Caribbean Regression Examples

- **Predicting tourist arrivals** based on season, airline prices, global economy
- **Forecasting sugarcane yield** based on rainfall, temperature, fertilizer
- **Estimating property prices** in Montego Bay based on location, size, amenities
- **Predicting cricket run rates** based on pitch condition, weather, batting order

---

## Key Algorithms

### 1. Linear Regression
**The simplest model.** Fits a straight line (or hyperplane) to data.

- **Equation:** `y = w₁x₁ + w₂x₂ + ... + wₙxₙ + b`
- **Caribbean use:** Predicting banana export revenue from production volume and world prices
- **Strength:** Simple, interpretable, fast
- **Weakness:** Cannot capture non-linear relationships

### 2. Logistic Regression
**Despite the name, it is for classification!** Outputs probabilities between 0 and 1.

- **Equation:** `P(y=1) = 1 / (1 + e^(-z))` where `z = wx + b`
- **Caribbean use:** Predicting whether a student will pass CAPE exams
- **Strength:** Gives probability estimates, interpretable
- **Weakness:** Linear decision boundaries only

### 3. Decision Trees
**Like a flowchart of questions.** Easy to understand, like playing 20 Questions.

```
Is the sea temperature > 26.5°C?
├── Yes: Is wind shear < 10 knots?
│   ├── Yes: HURRICANE LIKELY 🌀
│   └── No: Tropical Storm possible
└── No: Is it hurricane season?
    ├── Yes: Monitor closely
    └── No: Low risk
```

- **Caribbean use:** Classifying hurricane risk levels
- **Strength:** Interpretable, handles mixed data types
- **Weakness:** Prone to overfitting

### 4. Random Forests
**A committee of decision trees.** Each tree votes, majority wins.

- Think of it like a Caribbean parish council — many voices, better decisions
- **Caribbean use:** Predicting dengue outbreaks from environmental data
- **Strength:** Robust, handles overfitting well
- **Weakness:** Less interpretable than a single tree

### 5. Support Vector Machines (SVM)
**Finds the best boundary with maximum margin between classes.**

- Imagine drawing the widest possible road between two Caribbean towns (classes)
- **Caribbean use:** Classifying Caribbean music genres from audio features
- **Strength:** Effective in high dimensions, works well with clear margins
- **Weakness:** Slow on large datasets, sensitive to feature scaling

### 6. K-Nearest Neighbors (KNN)
**Yuh known by the company yuh keep!** Classifies based on closest neighbors.

- Just like in Caribbean communities — yuh neighborhood defines yuh
- **Caribbean use:** Recommending Caribbean dishes based on ingredient similarity
- **Strength:** Simple, no training phase
- **Weakness:** Slow prediction, sensitive to irrelevant features

### 7. Gradient Boosting (XGBoost, LightGBM)
**Build many weak models sequentially, each correcting the last.**

- Like a cricket team improving innings by innings
- **Caribbean use:** Predicting Caribbean economic indicators
- **Strength:** State-of-the-art for tabular data
- **Weakness:** Can overfit, many hyperparameters

---

## The Math Intuition

### Loss Functions — How We Measure Mistakes

**For Regression — Mean Squared Error (MSE):**

```
MSE = (1/n) Σ (yᵢ - ŷᵢ)²

Where:
  yᵢ = actual value (real rainfall was 200mm)
  ŷᵢ = predicted value (we predicted 180mm)
  n  = number of data points

We square the errors so big mistakes get penalized more.
Think: if yuh predict 10mm of rain and 200mm fall, that's a BIG problem
for the farmer. Squaring makes the model take large errors seriously.
```

**For Classification — Cross-Entropy Loss:**

```
Loss = -Σ [yᵢ log(ŷᵢ) + (1-yᵢ) log(1-ŷᵢ)]

Where:
  yᵢ = actual class (1 = snapper, 0 = not snapper)
  ŷᵢ = predicted probability of being a snapper

This punishes confident wrong predictions harshly.
If yuh model say "99% chance it a snapper" and it turns out to be
a grouper, the loss is HUGE. Be humble with yuh predictions!
```

### Gradient Descent — How the Model Learns

Imagine yuh lost in the Blue Mountains at night. Yuh want to reach the valley (minimum loss). Yuh can't see, but yuh can feel the slope under yuh feet:

1. **Feel the slope** (compute the gradient)
2. **Step downhill** (update weights in the opposite direction of the gradient)
3. **Repeat** until yuh reach flat ground (convergence)

```
Weight update rule:
  w_new = w_old - learning_rate × gradient

Learning rate is yuh step size:
  - Too large: Yuh overshoot the valley, bouncing around
  - Too small: Yuh take forever to get there
  - Just right: Yuh reach the bottom efficiently
```

### Bias-Variance Tradeoff

**A crucial concept!** Like tuning a Caribbean radio station:

- **High Bias (underfitting):** The radio can only pick up one station. Too simple — misses the signal.
  - Example: Using linear regression for clearly non-linear data

- **High Variance (overfitting):** The radio picks up everything including static. Too complex — memorizes noise.
  - Example: A decision tree with 1000 levels that memorizes training data

- **Sweet Spot:** Clear signal, minimal static. The model generalizes well.

```
Model Complexity →
Error
  |  \                    /
  |   \   Bias          /  Variance
  |    \  decreases   /   increases
  |     \           /
  |      \   ___  /
  |       \_/   \/  ← Sweet spot!
  |     Total Error
  |________________________
```

---

## Model Evaluation

### Train-Test Split

Never test a student with the same exam they practiced on! Same with models:

```python
from sklearn.model_selection import train_test_split

# 80% for training, 20% for testing
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
```

### Classification Metrics

| Metric | What It Measures | Caribbean Analogy |
|--------|-----------------|-------------------|
| **Accuracy** | % correct overall | Overall CXC pass rate |
| **Precision** | Of predicted positives, how many correct? | Of all fish yuh call "snapper," how many really are? |
| **Recall** | Of actual positives, how many found? | Of all actual snappers, how many did yuh correctly identify? |
| **F1 Score** | Balance of precision and recall | The all-rounder cricketer — good at both batting and bowling |

### The Confusion Matrix

```
                    Predicted
                 Snapper | Not Snapper
Actual  Snapper |  TP    |    FN      |
    Not Snapper |  FP    |    TN      |

TP = True Positive  (Said snapper, was snapper — correct!)
FP = False Positive (Said snapper, was grouper — wrong!)
FN = False Negative (Said not snapper, was snapper — missed it!)
TN = True Negative  (Said not snapper, wasn't — correct!)
```

### Regression Metrics

| Metric | Formula | Interpretation |
|--------|---------|---------------|
| **MAE** | Mean of \|y - ŷ\| | Average error in same units (e.g., mm of rain) |
| **MSE** | Mean of (y - ŷ)² | Penalizes large errors more |
| **RMSE** | √MSE | Same units as target, penalizes large errors |
| **R²** | 1 - (SS_res/SS_tot) | % of variance explained (1.0 = perfect) |

### Cross-Validation

Don't rely on a single train-test split! Use **K-Fold Cross-Validation**:

Think of it like a Caribbean round-robin cricket tournament — every team plays every other team. Each "fold" takes a turn being the test set:

```
Fold 1: [TEST] [Train] [Train] [Train] [Train]
Fold 2: [Train] [TEST] [Train] [Train] [Train]
Fold 3: [Train] [Train] [TEST] [Train] [Train]
Fold 4: [Train] [Train] [Train] [TEST] [Train]
Fold 5: [Train] [Train] [Train] [Train] [TEST]

Final score = average across all folds
```

---

## Practical Guidance

### Step-by-Step Workflow

1. **Define the Problem**
   - What are we predicting? Classification or regression?
   - Example: "Predict whether a tourist will return to Jamaica next year"

2. **Collect and Explore Data**
   - Gather features: trip duration, spending, satisfaction score, activities
   - Visualize distributions, check for missing values

3. **Prepare the Data**
   - Handle missing values (common in Caribbean datasets!)
   - Encode categorical variables (parish names, island names)
   - Scale numerical features
   - Split into train/test

4. **Choose and Train Models**
   - Start simple (logistic regression, decision tree)
   - Try more complex models (random forest, gradient boosting)
   - Use cross-validation

5. **Evaluate and Compare**
   - Check metrics on the test set
   - Look at the confusion matrix
   - Check for overfitting (big gap between train and test scores?)

6. **Tune Hyperparameters**
   - Grid search or random search
   - Example: max_depth for decision trees, C for SVM

7. **Deploy and Monitor**
   - Put the model into production
   - Monitor performance over time (data drift is real in the Caribbean — tourism patterns change!)

### Common Pitfalls

- **Data Leakage:** Don't let test data information leak into training. Like giving a student the CXC answers before the exam.
- **Class Imbalance:** If 95% of transactions are legit and 5% fraudulent, a model that always says "legit" gets 95% accuracy but catches zero fraud!
- **Feature Engineering Matters:** Raw data is like raw sugarcane — yuh need to process it. Create meaningful features.
- **Small Data:** Caribbean datasets are often small. Use cross-validation, simpler models, and data augmentation.

---

## Caribbean Applications

### Agriculture
- **Crop yield prediction:** Forecast banana, sugarcane, and cocoa yields based on weather, soil, and farming practices
- **Pest detection:** Classify whether a crop image shows disease or pest damage
- **Optimal planting time:** Predict best planting windows from historical climate data

### Tourism
- **Visitor forecasting:** Predict tourist arrivals for Caribbean destinations
- **Hotel pricing:** Dynamic pricing based on season, events, and demand
- **Satisfaction prediction:** Predict tourist satisfaction from review features

### Healthcare
- **Dengue prediction:** Predict dengue outbreaks from rainfall, temperature, and mosquito data
- **Patient readmission:** Predict which patients will be readmitted to Caribbean hospitals
- **Drug effectiveness:** Predict treatment outcomes for Caribbean populations

### Finance
- **Credit scoring:** Predict loan default probability for Caribbean bank customers
- **Remittance forecasting:** Predict diaspora remittance flows
- **Currency prediction:** Forecast exchange rates for Caribbean currencies

### Environment
- **Hurricane trajectory:** Predict storm paths affecting Caribbean islands
- **Coral reef health:** Classify reef health from underwater imagery
- **Sea level prediction:** Forecast sea level rise impacts on coastal communities

---

## Cricket and Sports Analytics

Cricket is more than a game in the Caribbean — it is culture! Supervised learning has plenty applications:

- **Batting average prediction:** Predict a West Indies batsman's performance based on pitch type, opposition, venue
- **Match outcome prediction:** Will the Windies win? Features: toss result, team composition, venue, recent form
- **Player valuation:** Predict CPL auction prices based on player statistics
- **Bowling analysis:** Classify bowling types (pace, spin, swing) from sensor data
- **Injury prediction:** Predict player injury risk from workload and fitness data

### Track and Field
- **Sprint time prediction:** Predict 100m times based on training data, weather, altitude
- **Talent identification:** Classify young athletes likely to succeed at the international level
- **Performance forecasting:** Predict Caribbean athletes' Olympic performance

---

## Quiz

Test yuh knowledge! Answer these 10 questions:

### Question 1
What is the main difference between classification and regression?

- A) Classification is harder than regression
- B) Classification predicts categories, regression predicts continuous values
- C) Regression only works with numbers
- D) Classification requires more data

<details>
<summary>Answer</summary>
**B)** Classification predicts categories (e.g., fish species), while regression predicts continuous numerical values (e.g., rainfall in mm).
</details>

### Question 2
A Caribbean bank wants to predict whether a loan application will default (yes/no). What type of supervised learning task is this?

- A) Regression
- B) Multi-class classification
- C) Binary classification
- D) Clustering

<details>
<summary>Answer</summary>
**C)** Binary classification — there are exactly two outcomes: default or no default.
</details>

### Question 3
What does overfitting mean in the context of a Caribbean weather prediction model?

- A) The model works perfectly on all data
- B) The model memorizes training data but performs poorly on new weather data
- C) The model is too simple to capture weather patterns
- D) The model needs more features

<details>
<summary>Answer</summary>
**B)** Overfitting means the model has memorized the training data (including noise) but fails to generalize to new, unseen weather data.
</details>

### Question 4
Which metric would be MOST important for a model that detects dengue outbreaks in Trinidad?

- A) Accuracy
- B) Precision
- C) Recall
- D) R² score

<details>
<summary>Answer</summary>
**C)** Recall — when detecting disease outbreaks, it is critical to catch as many true cases as possible (minimize false negatives). Missing a real outbreak is far worse than a false alarm.
</details>

### Question 5
What is the purpose of a train-test split?

- A) To make the dataset larger
- B) To evaluate how well the model generalizes to unseen data
- C) To speed up training
- D) To remove outliers

<details>
<summary>Answer</summary>
**B)** The train-test split lets us evaluate the model on data it has never seen, giving an honest estimate of real-world performance.
</details>

### Question 6
In linear regression for predicting sugarcane yield, what does the "slope" represent?

- A) The total yield
- B) The change in yield for a one-unit change in the input feature
- C) The error in prediction
- D) The number of features

<details>
<summary>Answer</summary>
**B)** The slope represents how much the predicted yield changes for each unit increase in the input feature (e.g., for each additional mm of rainfall).
</details>

### Question 7
Which algorithm works by finding the K closest data points and taking a vote?

- A) Decision Tree
- B) K-Nearest Neighbors
- C) Logistic Regression
- D) Random Forest

<details>
<summary>Answer</summary>
**B)** K-Nearest Neighbors (KNN) classifies a data point based on the majority class among its K nearest neighbors.
</details>

### Question 8
A Random Forest is best described as:

- A) A single very deep decision tree
- B) An ensemble of many decision trees that vote on the outcome
- C) A neural network shaped like a tree
- D) A clustering algorithm for forest data

<details>
<summary>Answer</summary>
**B)** A Random Forest is an ensemble method that builds many decision trees and combines their predictions through voting (classification) or averaging (regression).
</details>

### Question 9
Why is class imbalance a problem? Consider a Caribbean fraud detection system where 99% of transactions are legitimate.

- A) The model trains too slowly
- B) The model can achieve high accuracy by always predicting the majority class while catching no fraud
- C) The model needs more features
- D) Class imbalance is not actually a problem

<details>
<summary>Answer</summary>
**B)** With 99% legitimate transactions, a model that always predicts "legitimate" gets 99% accuracy but catches zero fraudulent transactions — completely useless for fraud detection.
</details>

### Question 10
What is cross-validation and why is it useful for Caribbean datasets that are often small?

- A) A way to increase dataset size
- B) A technique that uses multiple train-test splits to get a more reliable performance estimate
- C) A method to remove bad data
- D) A way to speed up training

<details>
<summary>Answer</summary>
**B)** Cross-validation rotates through multiple train-test splits, ensuring every data point gets used for both training and testing. This is especially valuable for small Caribbean datasets where a single split might not be representative.
</details>

---

## Next Steps

- Check out the `examples.py` file in this module for hands-on Caribbean supervised learning code
- Try modifying the examples with your own Caribbean data
- Move on to the **Unsupervised Learning** module to learn about finding hidden patterns

---

> 🧩 Puzzle Piece 11/12: Almost there! Remember: no spaces, no caps needed, just one continuous string

---

*Designed by Adrian Dunkley ([Adriandunkley.net](https://Adriandunkley.net)) | FREE Caribbean AI Curriculum*
*Big up to all Caribbean learners — di future is yours!*
