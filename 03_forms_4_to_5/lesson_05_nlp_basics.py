"""
Caribbean AI Academy — Forms 4-5: NLP Basics
==============================================
Lesson 5: Natural Language Processing with Caribbean Hotel Reviews
Designed by Adrian Dunkley (https://Adriandunkley.net) — FREE!

Learn fi analyze text data — from hotel reviews across di Caribbean!

# 🧩 Puzzle Piece 5/12: Think of an anime about online gaming -
#    the title has two words and 'art' is in the name
"""

import numpy as np
from collections import Counter
import re

# ============================================================
# PART 1: What is NLP?
# ============================================================
print("""
╔══════════════════════════════════════════════════════╗
║  LESSON 5: NLP BASICS — Caribbean Hotel Reviews     ║
║  Natural Language Processing fi di Caribbean!         ║
╚══════════════════════════════════════════════════════╝

NLP (Natural Language Processing) is how we teach computers
to understand human language. Think about it — yuh phone
autocorrect, Google Translate, Siri/Alexa — all of dat is NLP!

Today we gonna analyze hotel reviews from across di Caribbean.
""")

# ============================================================
# PART 2: Caribbean Hotel Review Dataset
# ============================================================

reviews = [
    # Jamaica
    {"text": "Amazing resort in Montego Bay! The jerk chicken was incredible and staff so friendly.",
     "island": "Jamaica", "rating": 5, "sentiment": "positive"},
    {"text": "Beautiful beach at Negril but the room was dirty and overpriced for what you get.",
     "island": "Jamaica", "rating": 2, "sentiment": "negative"},
    {"text": "Dunns River Falls was the highlight. Good hotel, nothing special but clean.",
     "island": "Jamaica", "rating": 3, "sentiment": "neutral"},

    # Trinidad & Tobago
    {"text": "Carnival season was absolutely electric! Hotel in Port of Spain was perfect location.",
     "island": "Trinidad", "rating": 5, "sentiment": "positive"},
    {"text": "Maracas Bay bake and shark is a must! Hotel was basic but the beach made up for it.",
     "island": "Trinidad", "rating": 4, "sentiment": "positive"},
    {"text": "Terrible service, rude staff, and the AC broken for three nights straight.",
     "island": "Trinidad", "rating": 1, "sentiment": "negative"},

    # Barbados
    {"text": "Bridgetown is charming and the rum punch is the best in the world! Lovely hotel.",
     "island": "Barbados", "rating": 5, "sentiment": "positive"},
    {"text": "Flying fish and cou cou was delicious! But the hotel pool was closed the whole trip.",
     "island": "Barbados", "rating": 3, "sentiment": "neutral"},
    {"text": "Worst vacation ever. Hotel had mold and bugs. Would never recommend to anyone.",
     "island": "Barbados", "rating": 1, "sentiment": "negative"},

    # Bahamas
    {"text": "Swimming with pigs in Exuma was unforgettable! Resort was pure luxury and paradise.",
     "island": "Bahamas", "rating": 5, "sentiment": "positive"},
    {"text": "Nassau was too touristy and everything way too expensive. Felt like a tourist trap.",
     "island": "Bahamas", "rating": 2, "sentiment": "negative"},

    # St. Lucia
    {"text": "The Pitons are breathtaking! Our hotel had an amazing view. Highly recommend!",
     "island": "St. Lucia", "rating": 5, "sentiment": "positive"},
    {"text": "Beautiful island but hotel food was bland and service was painfully slow.",
     "island": "St. Lucia", "rating": 2, "sentiment": "negative"},

    # Dominican Republic
    {"text": "Punta Cana resort was all-inclusive perfection. Beach, food, drinks — all amazing!",
     "island": "Dominican Republic", "rating": 5, "sentiment": "positive"},
    {"text": "Got food poisoning at the resort buffet. Ruined the entire trip for our family.",
     "island": "Dominican Republic", "rating": 1, "sentiment": "negative"},

    # Grenada
    {"text": "Spice Island lives up to its name! Nutmeg ice cream and beautiful Grand Anse Beach.",
     "island": "Grenada", "rating": 5, "sentiment": "positive"},

    # Antigua
    {"text": "365 beaches and we loved every one we visited! English Harbour is stunning.",
     "island": "Antigua", "rating": 5, "sentiment": "positive"},

    # Curacao
    {"text": "Willemstad is so colorful and beautiful! Snorkeling was world-class.",
     "island": "Curacao", "rating": 5, "sentiment": "positive"},
    {"text": "Hotel was okay but nothing special. The island itself was the real attraction.",
     "island": "Curacao", "rating": 3, "sentiment": "neutral"},
]

print(f"Dataset: {len(reviews)} Caribbean hotel reviews")
print(f"Islands covered: {', '.join(set(r['island'] for r in reviews))}\n")


# ============================================================
# PART 3: Tokenization
# ============================================================
# Tokenization = breaking text into individual words/pieces

print("=" * 50)
print("PART 3: TOKENIZATION")
print("=" * 50)

def simple_tokenize(text):
    """Break text into lowercase words, removing punctuation."""
    text = text.lower()
    text = re.sub(r'[^\w\s]', '', text)  # Remove punctuation
    tokens = text.split()
    return tokens

# Demo tokenization
sample = reviews[0]["text"]
tokens = simple_tokenize(sample)
print(f"\nOriginal: \"{sample}\"")
print(f"Tokens:   {tokens}")
print(f"Number of tokens: {len(tokens)}")

# Tokenize all reviews
all_tokens = []
for review in reviews:
    all_tokens.extend(simple_tokenize(review["text"]))

print(f"\nTotal tokens across all reviews: {len(all_tokens)}")
print(f"Unique words (vocabulary): {len(set(all_tokens))}")


# ============================================================
# PART 4: Word Frequency Analysis
# ============================================================

print("\n" + "=" * 50)
print("PART 4: WORD FREQUENCY")
print("=" * 50)

# Count word frequencies
word_counts = Counter(all_tokens)

# Remove common "stop words" (the, was, and, etc.)
stop_words = {"the", "was", "and", "is", "a", "in", "for", "but", "our",
              "we", "to", "of", "at", "it", "had", "too", "so", "an",
              "its", "that", "this", "with", "be", "or", "on", "up"}

interesting_words = {word: count for word, count in word_counts.items()
                     if word not in stop_words and len(word) > 2}
interesting_counter = Counter(interesting_words)

print("\nTop 15 most frequent interesting words:")
for word, count in interesting_counter.most_common(15):
    bar = "█" * count
    print(f"  {word:15s} {count:2d} {bar}")


# ============================================================
# PART 5: Sentiment Analysis
# ============================================================

print("\n" + "=" * 50)
print("PART 5: SENTIMENT ANALYSIS")
print("=" * 50)
print("""
Sentiment analysis = figuring out if text is positive, negative, or neutral.
Like when yuh read a review and know instantly if di person enjoyed it or not!
""")

# Build sentiment lexicons (word lists)
positive_words = {
    "amazing", "incredible", "beautiful", "lovely", "perfect", "best",
    "stunning", "unforgettable", "luxury", "paradise", "breathtaking",
    "delicious", "charming", "excellent", "fantastic", "wonderful",
    "loved", "recommend", "electric", "colorful", "highlight"
}

negative_words = {
    "dirty", "overpriced", "terrible", "rude", "broken", "worst",
    "mold", "bugs", "bland", "slow", "expensive", "trap", "ruined",
    "poisoning", "never", "closed", "painful", "disappointing"
}

def analyze_sentiment(text):
    """Simple lexicon-based sentiment analysis."""
    tokens = set(simple_tokenize(text))
    pos_score = len(tokens & positive_words)
    neg_score = len(tokens & negative_words)

    if pos_score > neg_score:
        return "positive", pos_score - neg_score
    elif neg_score > pos_score:
        return "negative", neg_score - pos_score
    else:
        return "neutral", 0

# Analyze all reviews
print("Sentiment Analysis Results:")
print(f"{'Island':20s} {'Predicted':10s} {'Actual':10s} {'Match':5s}")
print("-" * 50)

correct = 0
for review in reviews:
    predicted, confidence = analyze_sentiment(review["text"])
    actual = review["sentiment"]
    match = "YES" if predicted == actual else "NO"
    if predicted == actual:
        correct += 1
    print(f"  {review['island']:20s} {predicted:10s} {actual:10s} {match}")

accuracy = correct / len(reviews) * 100
print(f"\nAccuracy: {correct}/{len(reviews)} = {accuracy:.0f}%")
print("Not bad for a simple word-matching approach!")
print("Real NLP models (BERT, etc.) achieve 90%+ accuracy.")


# ============================================================
# PART 6: Text Classification with Scikit-Learn
# ============================================================

print("\n" + "=" * 50)
print("PART 6: ML-BASED TEXT CLASSIFICATION")
print("=" * 50)

try:
    from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
    from sklearn.naive_bayes import MultinomialNB
    from sklearn.model_selection import cross_val_score

    # Prepare data
    texts = [r["text"] for r in reviews]
    labels = [1 if r["sentiment"] == "positive" else 0 for r in reviews]

    # TF-IDF: Term Frequency - Inverse Document Frequency
    # Words that appear a lot in ONE review but not in ALL reviews are most important
    vectorizer = TfidfVectorizer(max_features=100, stop_words='english')
    X = vectorizer.fit_transform(texts)

    print(f"\nTF-IDF Matrix: {X.shape[0]} reviews x {X.shape[1]} features")
    print(f"Top features: {vectorizer.get_feature_names_out()[:10].tolist()}")

    # Train Naive Bayes classifier
    clf = MultinomialNB()
    scores = cross_val_score(clf, X, labels, cv=3)
    print(f"\nNaive Bayes Cross-Validation Accuracy: {scores.mean():.1%}")

    # Show what the model thinks are positive/negative words
    clf.fit(X, labels)
    feature_names = vectorizer.get_feature_names_out()
    pos_indices = np.argsort(clf.feature_log_prob_[1])[-5:]
    neg_indices = np.argsort(clf.feature_log_prob_[0])[-5:]

    print(f"\nMost 'positive' words: {[feature_names[i] for i in pos_indices]}")
    print(f"Most 'negative' words: {[feature_names[i] for i in neg_indices]}")

except ImportError:
    print("scikit-learn not installed. Run: pip install scikit-learn")


# ============================================================
# PART 7: Island Comparison
# ============================================================

print("\n" + "=" * 50)
print("PART 7: WHICH ISLAND HAS THE BEST REVIEWS?")
print("=" * 50)

island_ratings = {}
for review in reviews:
    island = review["island"]
    if island not in island_ratings:
        island_ratings[island] = []
    island_ratings[island].append(review["rating"])

print(f"\n{'Island':25s} {'Avg Rating':10s} {'Reviews':8s}")
print("-" * 45)
sorted_islands = sorted(island_ratings.items(), key=lambda x: np.mean(x[1]), reverse=True)
for island, ratings in sorted_islands:
    avg = np.mean(ratings)
    stars = "★" * int(round(avg)) + "☆" * (5 - int(round(avg)))
    print(f"  {island:23s} {avg:.1f} {stars}  ({len(ratings)} reviews)")


# ============================================================
# QUIZ
# ============================================================

print("\n" + "=" * 50)
print("INTERNALIZATION QUIZ")
print("=" * 50)
print("""
Q1: What is tokenization?
    a) Encrypting text  b) Breaking text into words/pieces
    c) Translating text  d) Compressing text

Q2: What is sentiment analysis?
    a) Counting words  b) Translating languages
    c) Determining if text is positive/negative  d) Spell checking

Q3: What does TF-IDF stand for?
    a) Text Frequency-Input Data Format
    b) Term Frequency-Inverse Document Frequency
    c) Token Filter-Index Data Frame
    d) Text Format-Integer Document File

Q4: In our analysis, which type of words did we remove?
    a) Long words  b) Caribbean words
    c) Stop words (common words like 'the', 'and')  d) Nouns

Q5: Why might a simple word-matching sentiment analyzer make mistakes?
    a) Words can have different meanings in context
    b) Sarcasm is hard to detect
    c) Negation changes meaning ("not good")
    d) All of the above

Q6: What Caribbean island had the most positive reviews in our dataset?
    (Look at the results above!)

Q7: Name two real-world applications of NLP in the Caribbean.
    (Think about tourism, government, healthcare...)

Q8: Why is Caribbean NLP particularly challenging?
    a) Too many islands  b) Multiple languages and Creole dialects
    c) Not enough computers  d) Reviews are too short

Answers in quiz_answers.md
""")
