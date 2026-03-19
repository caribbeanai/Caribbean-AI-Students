"""
=============================================================================
LESSON 4: Build Yuh First ML Model! 🤖🍊🌴
=============================================================================
Caribbean AI Academy — Forms 1-3 (Ages 11-14)
Designed by Adrian Dunkley (https://Adriandunkley.net)
FREE for Caribbean students who want to be AI Engineers, Scientists,
and Entrepreneurs.

YOOO! Dis is di big one! We actually BUILDING a machine learning model!

We go do TWO tings:
  1. PREDICT Caribbean fruit prices based on season, island, and fruit type
  2. CLASSIFY Caribbean fruits into categories

Dis uses scikit-learn — di most popular ML library fi beginners.
Every single line is explained. Nuh rush — take yuh time and understand
each step.

Let's build! 🔨
=============================================================================
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.metrics import accuracy_score, mean_absolute_error
from sklearn.preprocessing import LabelEncoder

# Set random seed fi reproducible results (so everybody get same answer)
np.random.seed(42)

# =============================================================================
# PART 1: PREDICT CARIBBEAN FRUIT PRICES
# =============================================================================
# We go predict di price of fruits based on:
#   - Which fruit it is
#   - Which island/country yuh buying it
#   - What season it is (peak season = cheaper because plenty supply)

print("=" * 65)
print("PART 1: PREDICTING CARIBBEAN FRUIT PRICES 🍊💰")
print("=" * 65)

# --- Step 1: Create di data ---
# Dis data is made up but based on real Caribbean pricing patterns.
# In real life, yuh would collect actual price data from markets!

# Fruits we go use
fruits = ["Mango", "Ackee", "Soursop", "Guinep", "Papaya",
          "Coconut", "Breadfruit", "Passion Fruit", "Guava",
          "June Plum", "Star Apple", "Jackfruit"]

# Islands/countries
islands = ["Jamaica", "Trinidad", "Barbados", "Guyana",
           "Bahamas", "St. Lucia", "Grenada", "Belize",
           "Dominica", "Antigua", "Haiti", "Dominican Republic"]

# Seasons (1=Jan-Mar, 2=Apr-Jun, 3=Jul-Sep, 4=Oct-Dec)
seasons = [1, 2, 3, 4]

# Generate 500 random data points
n_samples = 500
data_rows = []

for _ in range(n_samples):
    fruit = np.random.choice(fruits)
    island = np.random.choice(islands)
    season = np.random.choice(seasons)

    # Base price depends on fruit (in USD per pound)
    base_prices = {
        "Mango": 1.50, "Ackee": 3.00, "Soursop": 2.50,
        "Guinep": 1.00, "Papaya": 1.25, "Coconut": 0.75,
        "Breadfruit": 2.00, "Passion Fruit": 3.50, "Guava": 1.75,
        "June Plum": 1.25, "Star Apple": 2.00, "Jackfruit": 2.75,
    }
    price = base_prices[fruit]

    # Season adjustment: fruits cheaper when in season
    # Mango season = summer (season 3), Ackee = winter/spring (season 1)
    peak_seasons = {
        "Mango": 3, "Ackee": 1, "Soursop": 3, "Guinep": 3,
        "Papaya": 2, "Coconut": 3, "Breadfruit": 3, "Passion Fruit": 2,
        "Guava": 4, "June Plum": 3, "Star Apple": 1, "Jackfruit": 2,
    }
    if season == peak_seasons[fruit]:
        price *= 0.7   # 30% cheaper in peak season (plenty supply!)
    elif abs(season - peak_seasons[fruit]) == 1 or abs(season - peak_seasons[fruit]) == 3:
        price *= 0.85  # Slightly cheaper near peak season
    else:
        price *= 1.2   # More expensive when out of season

    # Island adjustment: some islands more expensive (tourism tax, import costs)
    island_multiplier = {
        "Jamaica": 1.0, "Trinidad": 0.95, "Barbados": 1.3,
        "Guyana": 0.80, "Bahamas": 1.4, "St. Lucia": 1.2,
        "Grenada": 1.1, "Belize": 0.85, "Dominica": 1.05,
        "Antigua": 1.25, "Haiti": 0.70, "Dominican Republic": 0.75,
    }
    price *= island_multiplier[island]

    # Add some random noise (real prices vary!)
    price *= np.random.uniform(0.85, 1.15)
    price = round(price, 2)

    data_rows.append({
        "Fruit": fruit,
        "Island": island,
        "Season": season,
        "Price_USD": price,
    })

# Create DataFrame
df = pd.DataFrame(data_rows)
print("\n--- Sample of our fruit price data ---")
print(df.head(10))
print(f"\nTotal data points: {len(df)}")
print(f"\n--- Quick Stats ---")
print(df.describe())

# --- Step 2: Prepare di data fi ML ---
# Machine learning models can't read text — dem only understand numbers!
# So we need fi convert "Mango", "Jamaica", etc. into numbers.
# We use LabelEncoder fi dis.

print("\n--- Preparing Data fi Machine Learning ---")

# Create encoders fi text columns
fruit_encoder = LabelEncoder()
island_encoder = LabelEncoder()

df["Fruit_Code"] = fruit_encoder.fit_transform(df["Fruit"])
df["Island_Code"] = island_encoder.fit_transform(df["Island"])

print("Fruit encoding:")
for fruit, code in zip(fruit_encoder.classes_, range(len(fruit_encoder.classes_))):
    print(f"  {fruit} → {code}")

print(f"\nIsland encoding:")
for island, code in zip(island_encoder.classes_, range(len(island_encoder.classes_))):
    print(f"  {island} → {code}")

# Features (X) = what we USE fi predicting
# Target (y)   = what we WANT fi predict
X = df[["Fruit_Code", "Island_Code", "Season"]]
y = df["Price_USD"]

print(f"\nFeatures shape: {X.shape}")
print(f"Target shape: {y.shape}")

# --- Step 3: Split into Training and Testing data ---
# We use 80% fi training (learning) and 20% fi testing (checking).
# Dis is like studying wid 80% of di textbook, then taking an exam on di other 20%.

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

print(f"\nTraining samples: {len(X_train)}")
print(f"Testing samples: {len(X_test)}")

# --- Step 4: Train di model! ---
# We use a Decision Tree — it's like a flowchart of yes/no questions.
# "Is it a Mango?" → "Is it in Barbados?" → "Is it season 3?" → Price!

print("\n--- Training di Model... ---")
model = DecisionTreeRegressor(max_depth=8, random_state=42)
model.fit(X_train, y_train)
print("Model trained! ✅")

# --- Step 5: Test di model ---
# Let's see how good it is at predicting prices it never seen before!

predictions = model.predict(X_test)
mae = mean_absolute_error(y_test, predictions)

print(f"\n--- Model Performance ---")
print(f"Mean Absolute Error: ${mae:.2f}")
print(f"(On average, di model's prediction is off by ${mae:.2f})")

# Show some actual vs predicted prices
print(f"\n--- Sample Predictions vs Actual ---")
results = pd.DataFrame({
    "Fruit": fruit_encoder.inverse_transform(X_test["Fruit_Code"]),
    "Island": island_encoder.inverse_transform(X_test["Island_Code"]),
    "Season": X_test["Season"].values,
    "Actual_Price": y_test.values,
    "Predicted_Price": predictions.round(2),
})
results["Difference"] = abs(results["Actual_Price"] - results["Predicted_Price"]).round(2)
print(results.head(15).to_string(index=False))

# --- Step 6: Make NEW predictions ---
# Now let's predict prices fi fruits we want!

print("\n--- Let's Predict Some Prices! ---")

# What does a Mango cost in Jamaica in summer (season 3)?
mango_code = fruit_encoder.transform(["Mango"])[0]
jamaica_code = island_encoder.transform(["Jamaica"])[0]
prediction = model.predict([[mango_code, jamaica_code, 3]])[0]
print(f"🥭 Mango in Jamaica (summer): ${prediction:.2f}/lb")

# What about Ackee in Jamaica in winter (season 1)?
ackee_code = fruit_encoder.transform(["Ackee"])[0]
prediction = model.predict([[ackee_code, jamaica_code, 1]])[0]
print(f"🍈 Ackee in Jamaica (winter/spring): ${prediction:.2f}/lb")

# Soursop in Barbados in winter (season 4)?
soursop_code = fruit_encoder.transform(["Soursop"])[0]
barbados_code = island_encoder.transform(["Barbados"])[0]
prediction = model.predict([[soursop_code, barbados_code, 4]])[0]
print(f"🍊 Soursop in Barbados (Oct-Dec): ${prediction:.2f}/lb")

# Coconut in Guyana in summer?
coconut_code = fruit_encoder.transform(["Coconut"])[0]
guyana_code = island_encoder.transform(["Guyana"])[0]
prediction = model.predict([[coconut_code, guyana_code, 3]])[0]
print(f"🥥 Coconut in Guyana (summer): ${prediction:.2f}/lb")


# =============================================================================
# PART 2: CLASSIFY CARIBBEAN FRUITS
# =============================================================================
# Now we go do CLASSIFICATION — putting fruits into categories!

print("\n\n" + "=" * 65)
print("PART 2: CLASSIFYING CARIBBEAN FRUITS 🍎🔬")
print("=" * 65)

# We go classify fruits based on their properties into categories:
# "Tropical Sweet", "Tropical Tart", "Starchy", "Citrus-like"

# Create a dataset of fruit properties
classification_data = {
    "Fruit": [
        "Mango", "Mango", "Papaya", "Papaya", "Star Apple", "Star Apple",
        "Guava", "Guava", "Passion Fruit", "Passion Fruit",
        "Soursop", "Soursop", "Tamarind", "Tamarind", "June Plum", "June Plum",
        "Breadfruit", "Breadfruit", "Jackfruit", "Jackfruit",
        "Coconut", "Coconut", "Ackee", "Ackee",
        "Guinep", "Guinep", "Sugar Apple", "Sugar Apple",
        "Carambola", "Carambola", "Sapodilla", "Sapodilla",
    ],
    # Sweetness level (1-10)
    "Sweetness": [
        9, 8, 7, 8, 8, 9,       # Sweet fruits
        6, 7, 4, 5,              # Medium
        7, 6, 3, 2, 4, 3,        # Tart-ish
        3, 2, 7, 6,              # Starchy/mixed
        5, 6, 2, 3,              # Mild/starchy
        7, 6, 9, 8,              # Sweet
        5, 4, 8, 9,              # Mixed/Sweet
    ],
    # Acidity level (1-10)
    "Acidity": [
        3, 2, 2, 3, 1, 2,        # Low acid
        5, 6, 8, 7,              # Higher acid
        4, 5, 8, 9, 7, 8,        # Tart/acidic
        1, 1, 2, 3,              # Low acid (starchy)
        1, 2, 1, 1,              # Very low acid
        5, 6, 2, 1,              # Mixed
        4, 5, 1, 2,              # Mixed
    ],
    # Water content (1-10)
    "Water_Content": [
        7, 8, 9, 8, 6, 7,
        8, 7, 7, 8,
        8, 9, 3, 4, 7, 6,
        4, 3, 5, 6,
        8, 7, 5, 6,
        6, 7, 7, 8,
        8, 7, 6, 5,
    ],
    # Size (1-10, small to large)
    "Size": [
        6, 7, 7, 8, 5, 6,
        4, 5, 3, 4,
        7, 8, 2, 3, 3, 4,
        8, 9, 9, 10,
        5, 6, 5, 4,
        2, 3, 5, 4,
        4, 3, 5, 6,
    ],
    # Category (what we want fi predict!)
    "Category": [
        "Tropical Sweet", "Tropical Sweet", "Tropical Sweet", "Tropical Sweet",
        "Tropical Sweet", "Tropical Sweet",
        "Tropical Tart", "Tropical Tart", "Tropical Tart", "Tropical Tart",
        "Tropical Tart", "Tropical Tart", "Tropical Tart", "Tropical Tart",
        "Tropical Tart", "Tropical Tart",
        "Starchy", "Starchy", "Starchy", "Starchy",
        "Starchy", "Starchy", "Starchy", "Starchy",
        "Tropical Sweet", "Tropical Sweet", "Tropical Sweet", "Tropical Sweet",
        "Tropical Tart", "Tropical Tart", "Tropical Sweet", "Tropical Sweet",
    ],
}

class_df = pd.DataFrame(classification_data)

print("\n--- Fruit Classification Dataset ---")
print(class_df.head(10))
print(f"\nTotal samples: {len(class_df)}")
print(f"\nCategory distribution:")
print(class_df["Category"].value_counts())

# --- Prepare data ---
X_class = class_df[["Sweetness", "Acidity", "Water_Content", "Size"]]
y_class = class_df["Category"]

# Split data
X_train_c, X_test_c, y_train_c, y_test_c = train_test_split(
    X_class, y_class, test_size=0.25, random_state=42
)

print(f"\nTraining samples: {len(X_train_c)}")
print(f"Testing samples: {len(X_test_c)}")

# --- Train a classifier ---
print("\n--- Training di Classifier... ---")
classifier = DecisionTreeClassifier(max_depth=5, random_state=42)
classifier.fit(X_train_c, y_train_c)
print("Classifier trained! ✅")

# --- Test it ---
class_predictions = classifier.predict(X_test_c)
accuracy = accuracy_score(y_test_c, class_predictions)
print(f"\n--- Classifier Performance ---")
print(f"Accuracy: {accuracy:.1%}")
print(f"(Di model correctly classified {accuracy:.1%} of di test fruits!)")

# Show results
print("\n--- Classification Results ---")
class_results = pd.DataFrame({
    "Fruit": class_df.loc[X_test_c.index, "Fruit"].values,
    "Sweetness": X_test_c["Sweetness"].values,
    "Acidity": X_test_c["Acidity"].values,
    "Actual_Category": y_test_c.values,
    "Predicted_Category": class_predictions,
})
class_results["Correct?"] = class_results["Actual_Category"] == class_results["Predicted_Category"]
print(class_results.to_string(index=False))

# --- Predict new fruits! ---
print("\n--- Classify a Mystery Fruit! ---")
print("Mystery fruit properties: Sweetness=8, Acidity=2, Water=7, Size=6")
mystery = classifier.predict([[8, 2, 7, 6]])[0]
print(f"Classification: {mystery}")
print("(High sweetness, low acidity — dat sound like a ripe mango or star apple!)")

print("\nMystery fruit #2: Sweetness=3, Acidity=8, Water=4, Size=3")
mystery2 = classifier.predict([[3, 8, 4, 3]])[0]
print(f"Classification: {mystery2}")
print("(Low sweetness, high acidity — dat's definitely something tart like tamarind!)")

# --- Feature importance ---
# Which properties matter most fi classifying fruits?
print("\n--- What Matters Most fi Classification? ---")
importances = classifier.feature_importances_
feature_names = ["Sweetness", "Acidity", "Water_Content", "Size"]
for name, imp in sorted(zip(feature_names, importances), key=lambda x: x[1], reverse=True):
    bar = "█" * int(imp * 40)
    print(f"  {name:15s}: {imp:.3f} {bar}")


# =============================================================================
# PART 3: WHAT YUH JUST LEARNED
# =============================================================================

print("\n\n" + "=" * 65)
print("PART 3: WHAT YUH JUST LEARNED! 🎓")
print("=" * 65)

print("""
Congratulations! Yuh just built TWO machine learning models! Here's what yuh did:

1. REGRESSION (Predicting fruit prices):
   - Created a dataset of Caribbean fruit prices
   - Encoded text data into numbers
   - Split data into training and testing sets
   - Trained a Decision Tree Regressor
   - Made predictions on new data!

2. CLASSIFICATION (Categorizing fruits):
   - Created a dataset of fruit properties
   - Trained a Decision Tree Classifier
   - Measured accuracy
   - Predicted categories fi mystery fruits
   - Found which features matter most!

KEY CONCEPTS:
  - Features = di input data (what yuh give di model)
  - Target = what yuh want fi predict
  - Training = di model learning from data
  - Testing = checking how good di model is
  - Regression = predicting a NUMBER (like price)
  - Classification = predicting a CATEGORY (like fruit type)

Caribbean sectors dat use dese same techniques:
  🏨 Tourism: Predicting hotel prices by season
  🌾 Agriculture: Classifying crop health from sensor data
  🐟 Fishing: Predicting fish catch amounts
  🏥 Healthcare: Classifying patient risk levels
  ⚡ Energy: Predicting electricity demand
  🏏 Sports: Classifying match outcomes
""")

print("=" * 65)
print("🎉 YUH BUILD YUH FIRST ML MODELS! Big tings! 🎉")
print("Try di quiz, then move to Lesson 5 (Data Visualization)!")
print("=" * 65)


# =============================================================================
# QUIZ TIME! 🧠
# =============================================================================
# Check quiz_answers.md when yuh done!
#
# Q1: What is the difference between REGRESSION and CLASSIFICATION?
#     a) Regression predicts numbers, classification predicts categories
#     b) They are the same thing
#     c) Regression is for text, classification is for numbers
#     d) Classification predicts numbers, regression predicts categories
#
# Q2: Why do we split data into training and testing sets?
#     a) To save memory
#     b) To make it run faster
#     c) To check if di model can handle NEW data it never seen
#     d) Because Python requires it
#
# Q3: What does a LabelEncoder do?
#     a) Adds labels to a chart
#     b) Converts text/categories into numbers fi ML
#     c) Deletes labels from data
#     d) Creates new columns
#
# Q4: If a model has 95% accuracy, what does dat mean?
#     a) It runs 95% faster
#     b) It uses 95% of di data
#     c) It correctly predicts 95 out of 100 examples
#     d) It only works 5% of the time
#
# Q5: What is a Decision Tree?
#     a) A tree yuh find in di Caribbean
#     b) A model dat makes decisions using yes/no questions in a tree structure
#     c) A way fi draw trees in Python
#     d) A list of decisions
#
# Q6: In our fruit price model, name di THREE features (inputs) we used.
#
# Q7: A farmer in Grenada wants fi predict nutmeg prices next month.
#     Would dem use regression or classification? Why?
#
# Q8: If yuh Mean Absolute Error (MAE) is $0.50, is dat good or bad
#     fi predicting fruit prices dat range from $0.50 to $5.00?
#     Explain yuh reasoning.
#
# Q9: Name ONE Caribbean sector where CLASSIFICATION would be useful,
#     and explain what categories yuh would predict.
#
# Q10: Why did we set random_state=42 in our code?
#      a) It's a magic number  b) It makes results random every time
#      c) It ensures we get di same results each time we run it
#      d) Python requires it
# =============================================================================
