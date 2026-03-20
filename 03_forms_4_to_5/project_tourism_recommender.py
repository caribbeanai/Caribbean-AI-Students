"""
Caribbean AI Academy — Forms 4-5: Capstone Project
====================================================
PROJECT: Caribbean Tourism Recommender System
Designed by Adrian Dunkley (https://Adriandunkley.net) — FREE!

YOUR MISSION: Build a system that recommends Caribbean islands
based on tourist preferences. Complete the TODO sections!
"""

import numpy as np

print("""
╔══════════════════════════════════════════════════════════╗
║  CAPSTONE PROJECT: CARIBBEAN TOURISM RECOMMENDER         ║
║  Match tourists with their perfect Caribbean island! 🏝️  ║
║                                                          ║
║  Complete the TODO sections to finish this project.      ║
╚══════════════════════════════════════════════════════════╝
""")

# ============================================================
# STEP 1: Island Feature Database
# ============================================================
# Each island rated 1-10 on: beach, history, food, nightlife, nature, adventure, culture, budget_friendly

island_features = {
    # Island:             [beach, history, food, nightlife, nature, adventure, culture, budget]
    "Jamaica":            [9, 7, 9, 8, 7, 7, 10, 6],
    "Trinidad & Tobago":  [7, 6, 10, 9, 7, 6, 10, 7],
    "Barbados":           [9, 8, 8, 7, 6, 6, 8, 5],
    "Bahamas":            [10, 5, 7, 8, 7, 9, 6, 4],
    "St. Lucia":          [9, 5, 7, 6, 9, 8, 7, 5],
    "Dominica":           [6, 4, 6, 3, 10, 10, 6, 7],
    "Grenada":            [8, 5, 8, 5, 8, 7, 7, 6],
    "Antigua & Barbuda":  [10, 7, 7, 7, 6, 7, 6, 4],
    "St. Kitts & Nevis":  [8, 8, 6, 5, 7, 6, 7, 5],
    "St. Vincent":        [7, 4, 6, 4, 9, 8, 6, 7],
    "Dominican Republic": [9, 7, 8, 9, 7, 8, 8, 8],
    "Cuba":               [8, 10, 8, 8, 7, 6, 10, 9],
    "Puerto Rico":        [8, 9, 9, 9, 7, 7, 9, 6],
    "Haiti":              [6, 10, 7, 5, 6, 5, 10, 8],
    "Cayman Islands":     [9, 4, 7, 6, 6, 9, 5, 3],
    "Curacao":            [8, 7, 8, 7, 5, 7, 8, 6],
    "Aruba":              [9, 5, 7, 8, 4, 6, 6, 5],
    "Belize":             [7, 8, 7, 5, 9, 10, 8, 7],
    "Guyana":             [4, 6, 7, 4, 10, 9, 8, 8],
    "Suriname":           [4, 6, 7, 4, 9, 8, 8, 7],
    "Bermuda":            [8, 7, 7, 6, 6, 6, 7, 3],
    "Turks & Caicos":     [10, 3, 6, 5, 6, 7, 4, 3],
    "Montserrat":         [5, 6, 5, 2, 8, 7, 5, 6],
}

feature_names = ["Beach", "History", "Food", "Nightlife", "Nature", "Adventure", "Culture", "Budget-Friendly"]

print(f"Island database: {len(island_features)} Caribbean destinations")
print(f"Features: {', '.join(feature_names)}\n")

# ============================================================
# STEP 2: Tourist Profiles (Test Data)
# ============================================================

tourist_profiles = [
    {"name": "Beach Lover Maria", "prefs": [10, 2, 5, 3, 4, 3, 3, 5],
     "desc": "Just wants the best beaches!"},
    {"name": "History Buff Carlos", "prefs": [3, 10, 7, 2, 4, 3, 9, 7],
     "desc": "Loves history and culture, traveling on a budget."},
    {"name": "Party Crew", "prefs": [7, 2, 8, 10, 3, 4, 5, 6],
     "desc": "Young group wanting nightlife and good food."},
    {"name": "Nature Explorer Kenji", "prefs": [4, 3, 5, 2, 10, 10, 4, 7],
     "desc": "Adventurer who loves rainforests and hiking."},
    {"name": "Food Tourist Priya", "prefs": [5, 5, 10, 5, 5, 4, 8, 6],
     "desc": "Lives to eat! Wants the best Caribbean cuisine."},
    {"name": "Family Robinson", "prefs": [8, 5, 7, 3, 7, 5, 6, 8],
     "desc": "Family with kids, need budget-friendly and safe beaches."},
]


# ============================================================
# STEP 3: Build the Recommender
# ============================================================

print("=" * 55)
print("STEP 3: BUILD YOUR RECOMMENDER")
print("=" * 55)

# METHOD 1: Cosine Similarity (provided as example)
def cosine_similarity(vec1, vec2):
    """Calculate how similar two vectors are (0 = different, 1 = identical)."""
    dot_product = np.dot(vec1, vec2)
    norm1 = np.linalg.norm(vec1)
    norm2 = np.linalg.norm(vec2)
    if norm1 == 0 or norm2 == 0:
        return 0
    return dot_product / (norm1 * norm2)

# TODO 1: Implement Euclidean Distance similarity
# =================================================
# Euclidean distance measures how "far apart" two vectors are.
# Smaller distance = more similar.
# Convert to similarity: similarity = 1 / (1 + distance)
#
def euclidean_similarity(vec1, vec2):
    # TODO: Calculate Euclidean distance between vec1 and vec2
    # distance = np.sqrt(np.sum((vec1 - vec2) ** 2))
    # return 1 / (1 + distance)
    pass  # Replace with your code

# TODO 2: Implement Weighted Preference matching
# ================================================
# Some tourists care more about certain features.
# Weight the comparison by how important each feature is to the tourist.
#
def weighted_similarity(tourist_prefs, island_scores):
    # TODO: Features the tourist rates highly should count more
    # weights = np.array(tourist_prefs) / np.sum(tourist_prefs)  # Normalize
    # weighted_island = np.array(island_scores) * weights
    # weighted_pref = np.array(tourist_prefs) * weights
    # return cosine_similarity(weighted_island, weighted_pref)
    pass  # Replace with your code


# ============================================================
# STEP 4: Generate Recommendations
# ============================================================

print("\n" + "=" * 55)
print("STEP 4: RECOMMENDATIONS")
print("=" * 55)

for tourist in tourist_profiles:
    print(f"\n🧳 {tourist['name']}: {tourist['desc']}")
    prefs = np.array(tourist["prefs"])

    # Calculate similarity with all islands
    scores = []
    for island, features in island_features.items():
        sim = cosine_similarity(prefs, np.array(features))
        scores.append((island, sim))

    # Sort by similarity (highest first)
    scores.sort(key=lambda x: -x[1])

    # Show top 3 recommendations
    print(f"  Top 3 recommendations:")
    for i, (island, sim) in enumerate(scores[:3]):
        bar = "█" * int(sim * 20)
        print(f"    {i+1}. {island:22s} (match: {sim:.1%}) {bar}")

    # TODO 3: Also show the WORST match
    # ====================================
    # worst = scores[-1]
    # print(f"  Worst match: {worst[0]} ({worst[1]:.1%})")


# ============================================================
# STEP 5: Improve the System (BONUS CHALLENGES)
# ============================================================

print("\n" + "=" * 55)
print("STEP 5: BONUS CHALLENGES")
print("=" * 55)
print("""
TODO 4: Add a "budget" filter
   - If tourist budget is "low", only recommend islands with budget >= 7
   - If "medium", budget >= 5
   - If "high", no filter

TODO 5: Add seasonal recommendations
   - Hurricane season (Jun-Nov): recommend southern islands (Trinidad,
     Curacao, Aruba, Bonaire — they're below the hurricane belt!)
   - Carnival season (Feb-Mar): recommend Trinidad!
   - Winter escape (Dec-Feb): any island, but note peak prices

TODO 6: Add a "surprise me" feature
   - Randomly recommend an island that the tourist might NOT expect
   - But filter out islands with < 40% match

TODO 7: Build a simple command-line interface
   - Ask the tourist to rate each feature 1-10
   - Show recommendations with explanations
   - Let them filter by budget and season

Think about it: This kind of system could be used by Caribbean
tourism boards to better match visitors with destinations.
Every island has something special to offer!
""")
