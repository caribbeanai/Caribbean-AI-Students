"""
Caribbean AI Academy — LLM Examples
====================================
Large Language Model examples with Caribbean context.
Designed by Adrian Dunkley (https://Adriandunkley.net) — FREE fi everybody!

These examples show how to use HuggingFace transformers for Caribbean NLP tasks.
"""

import warnings
warnings.filterwarnings('ignore')

# ============================================================
# EXAMPLE 1: Caribbean Creole Language Detection
# ============================================================
# Which Caribbean language/dialect is this text written in?

def creole_language_detection():
    """
    Detect whether text is English, Jamaican Patois, Trinidadian Creole,
    Haitian Kreyol, or Papiamento using simple feature-based classification.

    In production, yuh would use a fine-tuned transformer model,
    but dis shows di concept with rule-based features.
    """
    print("=" * 60)
    print("EXAMPLE 1: Caribbean Creole Language Detection")
    print("=" * 60)

    # Caribbean text samples
    texts = {
        "Jamaican Patois": [
            "Mi nah go a di market today, di rain too heavy",
            "Wah gwaan bredren, everyting criss?",
            "She fi come back before night fall ya know",
            "Di pickney dem a play inna di yard",
            "Bwoy, dat food did sweet nuh rass",
        ],
        "Trinidadian Creole": [
            "Ah going down de road to lime with meh pardner",
            "Yuh dotish or what? De fete start at eight",
            "She real tabanca since he leave she",
            "Doh make dat, yuh go get in trouble",
            "We reach already, come nah man",
        ],
        "Haitian Kreyol": [
            "Mwen renmen peyi mwen anpil",
            "Ki jan ou ye jodi a?",
            "Nou pral nan mache a demen",
            "Li te ale lakay li bonè",
            "Timoun yo ap jwe nan lakou a",
        ],
        "Papiamento": [
            "Bon dia, con ta bai cu bo?",
            "Mi ta bai cas awe nochi",
            "E muchanan ta hunga na patio",
            "Nos ta biba na un isla bunita",
            "Danki pa tur cos cu bo a hasi",
        ],
        "Standard English": [
            "The weather forecast predicts heavy rainfall tomorrow",
            "She went to the market to buy fresh vegetables",
            "The children are playing in the yard this afternoon",
            "I would like to visit the museum next weekend",
            "The meeting has been rescheduled to Friday morning",
        ],
    }

    # Simple feature-based detection (keyword markers)
    # In real life, yuh would train a proper classifier!
    markers = {
        "Jamaican Patois": ["mi", "nah", "fi", "dem", "inna", "pickney", "wah", "gwaan", "criss", "bwoy", "nuh"],
        "Trinidadian Creole": ["ah", "de", "meh", "doh", "pardner", "fete", "tabanca", "lime", "dotish", "nah"],
        "Haitian Kreyol": ["mwen", "ou", "nou", "yo", "pral", "anpil", "jodi", "lakay", "nan", "te"],
        "Papiamento": ["mi ta", "bo", "nos", "danki", "bunita", "awe", "hasi", "bai", "muchanan"],
        "Standard English": ["the", "would", "has been", "this afternoon", "forecast"],
    }

    def detect_language(text):
        text_lower = text.lower()
        scores = {}
        for lang, words in markers.items():
            score = sum(1 for w in words if w in text_lower)
            scores[lang] = score
        return max(scores, key=scores.get)

    # Test detection
    correct = 0
    total = 0
    for true_lang, samples in texts.items():
        for sample in samples:
            predicted = detect_language(sample)
            status = "OK" if predicted == true_lang else "MISS"
            if predicted == true_lang:
                correct += 1
            total += 1
            print(f"  [{status}] \"{sample[:50]}...\" -> {predicted}")

    print(f"\nAccuracy: {correct}/{total} ({100*correct/total:.0f}%)")
    print("Note: A fine-tuned BERT model would do MUCH better than keyword matching!\n")


# ============================================================
# EXAMPLE 2: Caribbean Sentiment Analysis
# ============================================================

def caribbean_sentiment_analysis():
    """
    Analyze sentiment of Caribbean hotel/tourism reviews.
    Using a simple bag-of-words approach to demonstrate the concept.
    With transformers, yuh would use a pre-trained sentiment model.
    """
    print("=" * 60)
    print("EXAMPLE 2: Caribbean Hotel Review Sentiment Analysis")
    print("=" * 60)

    reviews = [
        # Positive reviews from across the Caribbean
        {"text": "Di resort in Montego Bay was absolutely beautiful! Staff treated us like family.",
         "island": "Jamaica", "true_sentiment": "positive"},
        {"text": "Maracas Bay beach was stunning, and the bake and shark was the best I ever had!",
         "island": "Trinidad", "true_sentiment": "positive"},
        {"text": "The diving in Bonaire was world-class. Crystal clear water and healthy reefs.",
         "island": "Bonaire", "true_sentiment": "positive"},
        {"text": "Loved every minute in Barbuda. Pink sand beach is paradise on earth!",
         "island": "Antigua & Barbuda", "true_sentiment": "positive"},
        {"text": "The old San Juan charm is unmatched. Amazing food and friendly people.",
         "island": "Puerto Rico", "true_sentiment": "positive"},
        {"text": "Grand Cayman was perfect - Seven Mile Beach lived up to the hype completely!",
         "island": "Cayman Islands", "true_sentiment": "positive"},

        # Negative reviews
        {"text": "The hotel was dirty and overpriced. Very disappointed with the service.",
         "island": "Generic", "true_sentiment": "negative"},
        {"text": "Worst experience ever. Room had bugs and the food made us sick.",
         "island": "Generic", "true_sentiment": "negative"},
        {"text": "Not worth the money at all. Beach was crowded and full of seaweed.",
         "island": "Generic", "true_sentiment": "negative"},
        {"text": "Staff was rude and unhelpful. Would never come back to this place.",
         "island": "Generic", "true_sentiment": "negative"},

        # Mixed/Neutral reviews
        {"text": "The beach was nice but the hotel needs renovation. Food was okay.",
         "island": "Bahamas", "true_sentiment": "neutral"},
        {"text": "Beautiful scenery in Dominica but the roads were terrible and scary.",
         "island": "Dominica", "true_sentiment": "neutral"},
    ]

    # Simple sentiment lexicon
    positive_words = {"beautiful", "stunning", "amazing", "perfect", "loved", "best",
                      "paradise", "wonderful", "excellent", "fantastic", "great", "nice",
                      "world-class", "crystal", "healthy", "unmatched", "friendly", "charm"}
    negative_words = {"dirty", "overpriced", "disappointed", "worst", "bugs", "sick",
                      "rude", "unhelpful", "terrible", "scary", "never", "crowded"}

    print("\nReview Sentiment Analysis:")
    print("-" * 40)

    for review in reviews:
        words = set(review["text"].lower().split())
        pos_score = len(words & positive_words)
        neg_score = len(words & negative_words)

        if pos_score > neg_score:
            predicted = "positive"
        elif neg_score > pos_score:
            predicted = "negative"
        else:
            predicted = "neutral"

        match = "CORRECT" if predicted == review["true_sentiment"] else "WRONG"
        print(f"  [{review['island']:20s}] {predicted:8s} ({match}) - \"{review['text'][:55]}...\"")

    print("\n--- With HuggingFace Transformers (pseudocode) ---")
    print("""
    from transformers import pipeline

    # Load a pre-trained sentiment model
    sentiment = pipeline("sentiment-analysis")

    # Analyze Caribbean review
    result = sentiment("Di resort in Montego Bay was absolutely beautiful!")
    # Output: {'label': 'POSITIVE', 'score': 0.9998}

    # For better Caribbean results, fine-tune on Caribbean review data!
    """)


# ============================================================
# EXAMPLE 3: Caribbean Text Summarization
# ============================================================

def caribbean_text_summarization():
    """
    Demonstrate text summarization concepts with Caribbean news articles.
    """
    print("=" * 60)
    print("EXAMPLE 3: Caribbean News Summarization")
    print("=" * 60)

    # Sample Caribbean news articles (synthetic)
    articles = [
        {
            "title": "CARICOM Launches Regional AI Strategy",
            "text": """The Caribbean Community (CARICOM) today announced a comprehensive
            regional artificial intelligence strategy aimed at positioning Caribbean nations
            at the forefront of AI innovation. The strategy, developed over two years with
            input from all 15 member states, focuses on five key pillars: education and
            workforce development, infrastructure and connectivity, research and innovation,
            governance and ethics, and entrepreneurship. CARICOM Secretary-General emphasized
            that small island developing states must not be left behind in the AI revolution.
            The strategy allocates USD 50 million over five years, with funding from the
            Caribbean Development Bank and the Inter-American Development Bank. Jamaica,
            Trinidad and Tobago, and Barbados will host the first three regional AI centers
            of excellence. The strategy also calls for the development of Caribbean-specific
            AI models that understand local languages, cultures, and economic contexts.""",
        },
        {
            "title": "Hurricane Season AI Early Warning System Deployed",
            "text": """A new AI-powered hurricane early warning system has been deployed
            across the Eastern Caribbean, covering Dominica, St. Lucia, St. Vincent and
            the Grenadines, Grenada, and Antigua and Barbuda. The system, developed by
            Caribbean climate scientists in partnership with international researchers,
            uses deep learning models trained on 50 years of Caribbean weather data to
            predict hurricane paths and intensity with greater accuracy than traditional
            methods. The system sends alerts to mobile phones in English and local Creole
            languages, reaching communities that previously received limited warning.
            Initial tests showed the AI system predicted Hurricane Maria's impact on
            Dominica 48 hours earlier than conventional forecasting. The project was
            funded by the Green Climate Fund and the European Union.""",
        },
    ]

    def simple_extractive_summary(text, num_sentences=2):
        """Simple extractive summarization — pick the most important sentences."""
        import re
        sentences = re.split(r'[.!?]+', text.strip())
        sentences = [s.strip() for s in sentences if len(s.strip()) > 20]

        # Score by keyword importance (very simplified)
        important_words = {"ai", "caribbean", "system", "strategy", "developed",
                          "hurricane", "model", "deployed", "million", "warning",
                          "caricom", "first", "new", "key", "regional"}

        scored = []
        for sent in sentences:
            words = set(sent.lower().split())
            score = len(words & important_words)
            scored.append((score, sent))

        scored.sort(reverse=True)
        return ". ".join(s for _, s in scored[:num_sentences]) + "."

    for article in articles:
        print(f"\nTitle: {article['title']}")
        print(f"Original: {len(article['text'].split())} words")
        summary = simple_extractive_summary(article['text'])
        print(f"Summary ({len(summary.split())} words): {summary}")

    print("\n--- With HuggingFace (pseudocode) ---")
    print("""
    from transformers import pipeline

    summarizer = pipeline("summarization", model="facebook/bart-large-cnn")
    result = summarizer(article_text, max_length=100, min_length=30)
    print(result[0]['summary_text'])
    """)


# ============================================================
# EXAMPLE 4: Caribbean Chatbot (Rule-Based Demo)
# ============================================================

def caribbean_chatbot_demo():
    """
    A simple rule-based Caribbean knowledge chatbot.
    Demonstrates the concept — in production you'd use an LLM.
    """
    print("=" * 60)
    print("EXAMPLE 4: Caribbean Knowledge Chatbot (Demo)")
    print("=" * 60)

    knowledge_base = {
        "jamaica": "Jamaica is the third-largest Caribbean island. Capital: Kingston. Known for reggae, jerk chicken, Blue Mountains coffee, and being home to Usain Bolt — the fastest man ever!",
        "trinidad": "Trinidad and Tobago is the southernmost Caribbean nation. Capital: Port of Spain. Famous for Carnival, calypso, soca music, and having the world's largest natural asphalt lake (Pitch Lake).",
        "barbados": "Barbados is the easternmost Caribbean island. Capital: Bridgetown. Known as 'Little England', famous for rum, flying fish, cricket, and being the birthplace of Rihanna.",
        "haiti": "Haiti was the first Black republic (1804) and first Caribbean nation to gain independence. Capital: Port-au-Prince. Rich in art, music (kompa), and resilience.",
        "bahamas": "The Bahamas is an archipelago of 700+ islands. Capital: Nassau. Famous for crystal-clear waters, swimming pigs of Exuma, and being a major financial center.",
        "guyana": "Guyana is on the South American mainland but culturally Caribbean. Capital: Georgetown. Known for Kaieteur Falls (world's largest single-drop waterfall), gold, and diverse culture.",
        "cuba": "Cuba is the largest Caribbean island. Capital: Havana. Famous for classic cars, cigars, salsa music, and a world-class healthcare system.",
        "dominican republic": "The Dominican Republic shares Hispaniola with Haiti. Capital: Santo Domingo (oldest European settlement in Americas). Known for baseball, merengue, and beautiful beaches.",
        "cricket": "Cricket is HUGE in the Caribbean! The West Indies cricket team represents multiple nations. Legends include Sir Vivian Richards, Brian Lara (400 not out!), and Chris Gayle (Universe Boss).",
        "carnival": "Caribbean Carnival originated in Trinidad but is celebrated across the region. It features mas (masquerade), calypso/soca music, steelpan, and incredible costumes. Port of Spain Carnival is the biggest!",
        "caricom": "CARICOM (Caribbean Community) has 15 member states. Founded 1973. Works on economic integration, foreign policy coordination, and regional development. Headquartered in Georgetown, Guyana.",
    }

    sample_queries = [
        "Tell me about Jamaica",
        "What is Trinidad known for?",
        "Who plays cricket in the Caribbean?",
        "What is CARICOM?",
        "Tell me about Barbados",
    ]

    def chatbot_respond(query):
        query_lower = query.lower()
        for key, response in knowledge_base.items():
            if key in query_lower:
                return response
        return "Hmm, mi nuh too sure bout dat one. Try asking about a specific Caribbean country or topic like cricket, carnival, or CARICOM!"

    print("\nChatbot Demo (non-interactive):")
    print("-" * 40)
    for query in sample_queries:
        response = chatbot_respond(query)
        print(f"\n  You: {query}")
        print(f"  Bot: {response}")

    print("\n--- With an LLM (pseudocode) ---")
    print("""
    from transformers import pipeline

    chatbot = pipeline("text-generation", model="your-caribbean-finetuned-model")

    prompt = '''You are a knowledgeable Caribbean AI assistant.
    You speak with Caribbean warmth and know everything about
    the region's history, culture, economics, and sports.

    User: Tell me about Jamaica's contribution to world music.
    Assistant:'''

    response = chatbot(prompt, max_length=200)
    """)


# ============================================================
# RUN ALL EXAMPLES
# ============================================================

if __name__ == "__main__":
    print("\n" + "=" * 60)
    print("  CARIBBEAN AI ACADEMY — LLM EXAMPLES")
    print("  Designed by Adrian Dunkley (https://Adriandunkley.net)")
    print("  FREE fi all Caribbean students!")
    print("=" * 60 + "\n")

    creole_language_detection()
    print("\n")
    caribbean_sentiment_analysis()
    print("\n")
    caribbean_text_summarization()
    print("\n")
    caribbean_chatbot_demo()

    print("\n" + "=" * 60)
    print("  All examples complete! Now go build something amazing")
    print("  fi di Caribbean! 🌴")
    print("=" * 60)
