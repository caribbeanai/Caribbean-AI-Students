"""
=============================================================================
Caribbean AI Academy - Sixth Form Capstone Project
English <-> Jamaican Patois Rule-Based Translator
=============================================================================
A rule-based translation system between English and Jamaican Patois.

This is a RULE-BASED approach (not AI/ML). Students will:
1. Understand how rule-based NLP works
2. See its limitations compared to neural machine translation
3. Complete TODOs to expand the translator
4. Discuss how ML could improve Caribbean language technology

Author: Adrian Dunkley | Caribbean AI Academy
=============================================================================
"""

import re

print("=" * 60)
print("  Caribbean Creole Translator")
print("  English <-> Jamaican Patois")
print("=" * 60)


# =====================================================================
# SECTION 1: Vocabulary Dictionary
# =====================================================================

# English -> Patois word mappings
# Patois spelling varies — we use common conventions
ENGLISH_TO_PATOIS = {
    # Pronouns
    'i': 'mi',
    'me': 'mi',
    'my': 'mi',
    'you': 'yuh',
    'your': 'yuh',
    'he': 'im',
    'him': 'im',
    'his': 'im',
    'she': 'shi',
    'her': 'har',
    'we': 'wi',
    'us': 'wi',
    'our': 'wi',
    'they': 'dem',
    'them': 'dem',
    'their': 'dem',

    # Common verbs
    'is': 'a',
    'am': 'a',
    'are': 'a',
    'was': 'did',
    'were': 'did',
    'going': 'gwaan',
    'go': 'gwaan',
    'come': 'come',
    'know': 'know',
    'want': 'waan',
    'have': 'ave',
    'has': 'ave',
    'had': 'did ave',
    'give': 'gi',
    'tell': 'tell',
    'said': 'seh',
    'say': 'seh',
    'eat': 'nyam',
    'eating': 'nyamming',
    'look': 'look',
    'see': 'si',
    'think': 'tink',
    'thing': 'ting',
    'things': 'tings',
    'understand': 'ovahstand',
    'run': 'run',
    'walk': 'walk',
    'talk': 'chat',
    'talking': 'chatting',
    'like': 'like',
    'love': 'love',
    'make': 'mek',
    'take': 'tek',
    'bring': 'bring',
    'put': 'put',
    'get': 'get',
    'let': 'mek',
    'can': 'can',
    'cannot': 'cyah',
    "can't": 'cyah',
    'will': 'wi',
    "don't": 'nuh',
    'do': 'duh',
    "didn't": 'neva',
    'did': 'did',
    "won't": 'nah go',
    "isn't": 'nuh',
    "aren't": 'nuh',

    # Common nouns
    'man': 'man',
    'woman': 'ooman',
    'child': 'pickney',
    'children': 'pickney dem',
    'boy': 'bwoy',
    'girl': 'gyal',
    'friend': 'bredren',
    'friends': 'bredren dem',
    'mother': 'madda',
    'father': 'fadda',
    'brother': 'bredda',
    'sister': 'sista',
    'house': 'yaad',
    'home': 'yaad',
    'money': 'money',
    'food': 'food',
    'water': 'wata',
    'the': 'di',
    'this': 'dis',
    'that': 'dat',
    'what': 'wah',
    'where': 'weh',
    'when': 'wen',
    'why': 'why',
    'how': 'how',
    'here': 'yah',
    'there': 'deh',
    'with': 'wid',
    'from': 'from',
    'about': 'bout',
    'something': 'supm',
    'nothing': 'nutten',
    'everything': 'everyting',
    'everyone': 'everybody',
    'small': 'likkle',
    'little': 'likkle',
    'big': 'big',
    'good': 'good',
    'bad': 'bad',
    'very': 'well',
    'really': 'fi real',
    'always': 'always',
    'never': 'neva',
    'now': 'now',
    'today': 'today',
    'tomorrow': 'tomorrow',
    'yesterday': 'yesterday',
    'night': 'night',
    'morning': 'mawnin',
    'school': 'school',
    'work': 'work',
    'street': 'road',
    'car': 'cyar',

    # TODO 1: Add 20 more word mappings!
    # Think about: sports terms (cricket, football, track),
    # food (ackee, bammy, jerk), music (reggae, dancehall),
    # nature (river, mountain, sea), school subjects, etc.
    # Example:
    # 'beautiful': 'pretty',
    # 'mountain': 'mountain',
}

# Build reverse dictionary (Patois -> English)
# Note: Some Patois words map to multiple English words
# We take the first mapping found
PATOIS_TO_ENGLISH = {}
for eng, pat in ENGLISH_TO_PATOIS.items():
    if pat not in PATOIS_TO_ENGLISH:
        PATOIS_TO_ENGLISH[pat] = eng


# =====================================================================
# SECTION 2: Common Phrases
# =====================================================================

ENGLISH_PHRASES_TO_PATOIS = {
    "what's up": "wah gwaan",
    "what is up": "wah gwaan",
    "how are you": "how yuh stay",
    "how are you doing": "wah gwaan",
    "i am fine": "mi good",
    "i'm fine": "mi good",
    "good morning": "good mawnin",
    "good night": "good night",
    "thank you": "give thanks",
    "thanks": "give thanks",
    "you're welcome": "no problem",
    "i don't know": "mi nuh know",
    "i don't care": "mi nuh care",
    "come here": "come yah",
    "go away": "gweh",
    "leave me alone": "lef mi alone",
    "be quiet": "hush yuh mouth",
    "hurry up": "hurry up nuh",
    "what happened": "wah happen",
    "what is that": "a wah dat",
    "who is that": "a who dat",
    "where are you going": "weh yuh a go",
    "i am going home": "mi a go yaad",
    "i am hungry": "mi belly a hurt mi",
    "it's raining": "rain a fall",
    "no problem": "no problem mon",
    "that's great": "dat deh good",
    "i love you": "mi love yuh",
    "see you later": "likkle more",
    "goodbye": "walk good",
    "take care": "walk good",

    # TODO 2: Add 10 more common phrases!
    # Think about greetings, farewells, expressions of emotion,
    # common questions, and everyday Caribbean conversations.
}

PATOIS_PHRASES_TO_ENGLISH = {v: k for k, v in ENGLISH_PHRASES_TO_PATOIS.items()}


# =====================================================================
# SECTION 3: Grammar Transformation Rules
# =====================================================================

def apply_english_to_patois_grammar(text):
    """
    Apply Jamaican Patois grammar transformations to text.

    Key Patois grammar differences from Standard English:
    1. No conjugation: "I am / you are / he is" -> all use "a"
    2. "fi" replaces "to" (infinitive): "to go" -> "fi go"
    3. "a" marks continuous: "I am running" -> "mi a run"
    4. "did" marks past: "I was" -> "mi did"
    5. "nuh" for negation: "don't" -> "nuh"
    6. Plural with "dem": "the boys" -> "di bwoy dem"
    """
    # Rule: "to [verb]" -> "fi [verb]" (infinitive marker)
    text = re.sub(r'\bto (\w+)', r'fi \1', text)

    # Rule: "ing" endings often dropped or modified
    # "running" -> "run", "coming" -> "come"
    # (simplified — real Patois is more nuanced)
    text = re.sub(r'\b(\w{3,})ing\b', lambda m: m.group(1)
                  if len(m.group(1)) > 2 else m.group(0), text)

    # Rule: "th" sounds often become "d" or "t"
    text = re.sub(r'\bthe\b', 'di', text)
    text = re.sub(r'\bthat\b', 'dat', text)
    text = re.sub(r'\bthis\b', 'dis', text)
    text = re.sub(r'\bthere\b', 'deh', text)
    text = re.sub(r'\bthen\b', 'den', text)
    text = re.sub(r'\bthink\b', 'tink', text)
    text = re.sub(r'\bthing\b', 'ting', text)
    text = re.sub(r'\bthings\b', 'tings', text)
    text = re.sub(r'\bwith\b', 'wid', text)

    # TODO 3: Add more grammar rules!
    # Ideas:
    # - "er" endings -> "a" (e.g., "water" -> "wata", "sister" -> "sista")
    # - Double negatives are standard in Patois
    # - "ould" -> "woulda" patterns
    # - Question formation differences

    return text


def apply_patois_to_english_grammar(text):
    """
    Reverse grammar transformations: Patois -> English.

    TODO 4: Implement reverse grammar rules.
    This is harder because Patois grammar is more flexible!
    """
    # Rule: "fi [verb]" -> "to [verb]"
    text = re.sub(r'\bfi (\w+)', r'to \1', text)

    # Rule: "di" -> "the"
    text = re.sub(r'\bdi\b', 'the', text)

    # Rule: "dat" -> "that"
    text = re.sub(r'\bdat\b', 'that', text)

    # Rule: "dis" -> "this"
    text = re.sub(r'\bdis\b', 'this', text)

    # Rule: "deh" -> "there"
    text = re.sub(r'\bdeh\b', 'there', text)

    # TODO 5: Add more reverse grammar rules!

    return text


# =====================================================================
# SECTION 4: Translation Functions
# =====================================================================

def translate_english_to_patois(text):
    """
    Translate English text to Jamaican Patois.

    Process:
    1. Check for known phrases first (longer matches)
    2. Apply grammar transformations
    3. Replace individual words using dictionary
    """
    text_lower = text.lower().strip()

    # Step 1: Check phrase dictionary (longest match first)
    sorted_phrases = sorted(ENGLISH_PHRASES_TO_PATOIS.keys(),
                           key=len, reverse=True)
    for phrase in sorted_phrases:
        if phrase in text_lower:
            text_lower = text_lower.replace(
                phrase, ENGLISH_PHRASES_TO_PATOIS[phrase])

    # Step 2: Apply grammar rules
    text_lower = apply_english_to_patois_grammar(text_lower)

    # Step 3: Word-by-word replacement
    words = text_lower.split()
    translated = []
    for word in words:
        # Remove punctuation for lookup, keep it for output
        clean = re.sub(r'[^\w]', '', word)
        punct = word[len(clean):] if len(word) > len(clean) else ''

        if clean in ENGLISH_TO_PATOIS:
            translated.append(ENGLISH_TO_PATOIS[clean] + punct)
        else:
            translated.append(word)

    return ' '.join(translated)


def translate_patois_to_english(text):
    """
    Translate Jamaican Patois text to English.

    TODO 6: Improve this function!
    Currently basic — can you make it handle more cases?
    Consider: context-dependent translations, handling Patois
    spelling variations, dealing with words that have multiple
    English meanings.
    """
    text_lower = text.lower().strip()

    # Step 1: Check phrase dictionary
    sorted_phrases = sorted(PATOIS_PHRASES_TO_ENGLISH.keys(),
                           key=len, reverse=True)
    for phrase in sorted_phrases:
        if phrase in text_lower:
            text_lower = text_lower.replace(
                phrase, PATOIS_PHRASES_TO_ENGLISH[phrase])

    # Step 2: Apply reverse grammar rules
    text_lower = apply_patois_to_english_grammar(text_lower)

    # Step 3: Word-by-word replacement
    words = text_lower.split()
    translated = []
    for word in words:
        clean = re.sub(r'[^\w]', '', word)
        punct = word[len(clean):] if len(word) > len(clean) else ''

        if clean in PATOIS_TO_ENGLISH:
            translated.append(PATOIS_TO_ENGLISH[clean] + punct)
        else:
            translated.append(word)

    return ' '.join(translated)


# =====================================================================
# SECTION 5: Testing the Translator
# =====================================================================

print("\n--- English to Patois Translation ---\n")

english_test_sentences = [
    "How are you doing today?",
    "I am going to the store to buy food",
    "The children are playing in the street",
    "I don't know what happened",
    "She said that he is coming tomorrow",
    "Come here and eat the food",
    "I want to go home",
    "What is that thing over there?",
    "The boy can't find his mother",
    "We are going to school this morning",
    "I think you should understand this",
    "Leave me alone, I am eating",
]

for sentence in english_test_sentences:
    translated = translate_english_to_patois(sentence)
    print(f"  ENG: {sentence}")
    print(f"  PAT: {translated}")
    print()


print("--- Patois to English Translation ---\n")

patois_test_sentences = [
    "wah gwaan bredren",
    "mi a go yaad now",
    "di pickney dem a play",
    "mi nuh know wah happen",
    "shi seh im a come tomorrow",
    "come yah and nyam di food",
    "mi waan fi go yaad",
    "likkle more bredren",
    "mi love yuh fi real",
    "walk good mi bredren",
]

for sentence in patois_test_sentences:
    translated = translate_patois_to_english(sentence)
    print(f"  PAT: {sentence}")
    print(f"  ENG: {translated}")
    print()


# =====================================================================
# SECTION 6: Translation Accuracy Assessment
# =====================================================================
print("--- Translation Accuracy Check ---\n")

# Known correct translations for testing
test_pairs = [
    ("how are you", "how yuh stay"),
    ("come here", "come yah"),
    ("the children", "di pickney dem"),
    ("i don't know", "mi nuh know"),
    ("goodbye", "walk good"),
]

correct = 0
for eng, expected_pat in test_pairs:
    result = translate_english_to_patois(eng)
    match = expected_pat in result or result == expected_pat
    if match:
        correct += 1
    status = "PASS" if match else "FAIL"
    print(f"  [{status}] '{eng}' -> '{result}' (expected: '{expected_pat}')")

print(f"\nAccuracy: {correct}/{len(test_pairs)} "
      f"({100*correct/len(test_pairs):.0f}%)")


# =====================================================================
# SECTION 7: Limitations and Future Work
# =====================================================================
print("\n--- Limitations of Rule-Based Translation ---\n")
print("""
This translator is LIMITED because:

1. VOCABULARY: Only covers a fraction of words. Real Patois has
   thousands of unique words, expressions, and slang that evolve.

2. GRAMMAR: Patois grammar is complex and context-dependent.
   Rules can't capture every pattern.
   - "Mi a go" = "I am going" (present continuous)
   - "Mi go" = "I went" or "I will go" (depends on context!)

3. CONTEXT: The same Patois word can mean different things:
   - "Bad" can mean "bad" OR "really good" (like "wicked" in English)
   - "Run" can mean physical running OR managing something

4. REGIONAL VARIATION: Patois varies across Jamaica:
   - Kingston vs rural St Elizabeth vs Montego Bay
   - Each parish has its own flavour!

5. TONE AND EMPHASIS: Meaning changes with intonation
   - "Yuh MAD?" (are you crazy?) vs "Yuh mad?" (are you angry?)

HOW AI/ML COULD IMPROVE THIS:
- Train a neural machine translation model on parallel text
- Use a large language model fine-tuned on Patois text
- Collect a corpus of Patois text from social media, music lyrics,
  literature, and transcribed speech
- Community-driven data collection from across Jamaica

TODO 7: Write a paragraph discussing how you would collect
training data for a machine learning Patois translator.
Consider: ethics, consent, representation, and cultural sensitivity.

TODO 8: Research ONE other Caribbean Creole language (Trinidadian
Creole, Haitian Kreyol, Papiamento, or Garifuna) and add at least
5 word translations and 3 phrase translations to this system.
Create a new dictionary and translation function for that language.
""")


# =====================================================================
# SECTION 8: Interactive Mode
# =====================================================================
print("=" * 60)
print("  Interactive Translator")
print("  Type English or Patois text to translate.")
print("  Commands: 'eng2pat', 'pat2eng', 'quit'")
print("=" * 60)

# Uncomment below for interactive mode:
# mode = 'eng2pat'
# while True:
#     user_input = input(f"\n[{mode}] Enter text (or command): ").strip()
#     if user_input.lower() == 'quit':
#         print("Walk good! Likkle more!")
#         break
#     elif user_input.lower() == 'eng2pat':
#         mode = 'eng2pat'
#         print("Mode: English -> Patois")
#         continue
#     elif user_input.lower() == 'pat2eng':
#         mode = 'pat2eng'
#         print("Mode: Patois -> English")
#         continue
#
#     if mode == 'eng2pat':
#         result = translate_english_to_patois(user_input)
#     else:
#         result = translate_patois_to_english(user_input)
#     print(f"  -> {result}")

print("\n(Uncomment the interactive section in the code to try it!)")
print("\nProject complete! Yuh build a Caribbean language tool!")
print("Walk good, bredren!")
print("=" * 60)
