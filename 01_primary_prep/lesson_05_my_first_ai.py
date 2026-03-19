"""
=============================================================
  Lesson 5: My First AI — Caribbean Weather Predictor!
=============================================================

  Caribbean AI Curriculum — Primary/Prep School (Ages 6-11)
  Designed by Adrian Dunkley | Adriandunkley.net | FREE

  HOW TO RUN THIS PROGRAM:
  1. Make sure Python is installed on yuh computer
  2. Open a terminal (command prompt)
  3. Type: python lesson_05_my_first_ai.py
  4. Answer di questions and watch di AI predict!

  WHAT THIS PROGRAM DOES:
  - It asks yuh yes/no questions about di weather
  - Based on yuh answers, it PREDICTS if rain a come!
  - Dis is how real AI works — it uses information fi make decisions!

  CARIBBEAN WEATHER FACTS:
  - Di Caribbean have two main seasons: dry season and wet season
  - Wet season (hurricane season) is usually June to November
  - Di trade winds blow from di east and bring weather patterns
  - Caribbean islands can have rain on one side and sun on di other!
"""

# ============================================================
# STEP 1: Welcome message
# ============================================================

print("=" * 55)
print("  CARIBBEAN WEATHER PREDICTOR AI")
print("  ==============================")
print()
print("  Designed by Adrian Dunkley | Adriandunkley.net")
print("  Caribbean AI Curriculum — FREE fi everyone!")
print()
print("  Mi is a simple AI dat can predict if rain a come!")
print("  Just answer mi questions wid 'yes' or 'no'.")
print()
print("  Dis is how REAL AI works — it ask questions,")
print("  look at di answers, and make a prediction!")
print("=" * 55)
print()

# ============================================================
# STEP 2: Ask fi di player's name and island
# ============================================================

name = input("Wah yuh name? ")
print()
print(f"Welcome, {name}! Let mi predict yuh weather today!")
print()

island = input("Which Caribbean island yuh deh pon? ")
print()
print(f"Ah, {island}! Beautiful place! Let mi check di conditions...")
print()

# ============================================================
# SECRET WORD EASTER EGG!
# If someone type di secret word "mango" as their island,
# something special happen!
# ============================================================

if island.lower().strip() == "mango":
    print("*" * 50)
    print("  YUH FIND DI SECRET! 🥭🥭🥭")
    print()
    print("  'Mango' is di secret word!")
    print("  Here's a Caribbean AI joke fi yuh:")
    print()
    print("  Why did di AI go to Carnival?")
    print("  Because it wanted fi learn di STEEL-pan-dard")
    print("  algorithm fi wining! 😄🎶")
    print()
    print("  Fun fact: Di word 'algorithm' come from")
    print("  a mathematician named Al-Khwarizmi who lived")
    print("  over 1,000 years ago. Algorithms are di")
    print("  step-by-step instructions dat AI follow —")
    print("  just like a recipe fi cook curry chicken!")
    print("*" * 50)
    print()
    print("Now let's do di weather prediction anyway!")
    print()

# ============================================================
# STEP 3: Ask weather questions
# Each answer adds to a "rain score"
# Dis is how AI works — it collects data and uses it fi decide!
# ============================================================

# We start wid a rain_score of 0
# Every answer dat suggest rain will ADD points
# At di end, if di score high, we predict rain!

rain_score = 0  # Start at zero — no rain signs yet

print("-" * 40)
print("  QUESTION TIME! Answer 'yes' or 'no'")
print("-" * 40)
print()

# QUESTION 1: Clouds
print("Question 1:")
answer1 = input("Yuh see dark clouds in di sky? (yes/no): ")
print()

if answer1.lower().strip() in ["yes", "y", "yeah", "yeh", "yah"]:
    rain_score = rain_score + 3  # Dark clouds = BIG sign of rain!
    print("  Hmm, dark clouds... dat nah look good! ☁️")
else:
    print("  Clear sky so far... nice! ☀️")
print()

# QUESTION 2: Wind
print("Question 2:")
answer2 = input("Di wind a blow strong-strong? (yes/no): ")
print()

if answer2.lower().strip() in ["yes", "y", "yeah", "yeh", "yah"]:
    rain_score = rain_score + 2  # Strong wind = medium sign of rain
    print("  Strong wind can carry rain clouds in! 💨")
else:
    print("  Calm breeze... dat is a good sign! 🌿")
print()

# QUESTION 3: Humidity (Feeling sticky)
print("Question 3:")
answer3 = input("Di air feel sticky and hot-hot? (yes/no): ")
print()

if answer3.lower().strip() in ["yes", "y", "yeah", "yeh", "yah"]:
    rain_score = rain_score + 2  # Sticky = humidity = possible rain
    print("  Sticky air mean plenty moisture — rain might come! 🥵")
else:
    print("  Nice and comfortable! 😊")
print()

# QUESTION 4: Animals behaving funny
print("Question 4:")
answer4 = input("Di birds dem flying low-low to di ground? (yes/no): ")
print()

if answer4.lower().strip() in ["yes", "y", "yeah", "yeh", "yah"]:
    rain_score = rain_score + 2  # Old Caribbean wisdom!
    print("  Caribbean grandparents always say: low birds mean rain a come! 🐦")
else:
    print("  Birds flying high — usually a dry sign! 🦅")
print()

# QUESTION 5: Time of year
print("Question 5:")
answer5 = input("Is it between June and November right now? (yes/no): ")
print()

if answer5.lower().strip() in ["yes", "y", "yeah", "yeh", "yah"]:
    rain_score = rain_score + 2  # Rainy/hurricane season
    print("  Dat is hurricane season — more rain is normal! 🌊")
else:
    rain_score = rain_score - 1  # Dry season — less chance
    print("  Dry season! Less chance of rain usually. 🌞")
print()

# QUESTION 6: Morning dew
print("Question 6:")
answer6 = input("Was there dew (water drops) on di grass dis morning? (yes/no): ")
print()

if answer6.lower().strip() in ["yes", "y", "yeah", "yeh", "yah"]:
    print("  Interesting! Dew can mean moisture in di air. 💧")
    rain_score = rain_score + 1
else:
    print("  Dry grass — less moisture around. 🌾")
print()

# ============================================================
# STEP 4: Make di prediction!
# Dis is di "AI decision" part — using all di data fi decide!
# ============================================================

print("=" * 55)
print(f"  WEATHER PREDICTION FOR {name.upper()}")
print(f"  Location: {island}")
print("=" * 55)
print()
print(f"  Rain Score: {rain_score} out of 12")
print()

# Here is where di AI makes its decision!
# Just like real AI, we use di collected data fi predict

if rain_score >= 8:
    print("  🌧️🌧️🌧️  PREDICTION: HEAVY RAIN A COME!")
    print()
    print("  Mi STRONGLY predict rain today!")
    print("  Grab yuh umbrella and don't hang out")
    print("  no clothes on di line!")
    print(f"  {island} going get WET today!")
    confidence = "very high"

elif rain_score >= 5:
    print("  🌦️  PREDICTION: RAIN LIKELY!")
    print()
    print("  Good chance of rain today.")
    print("  Better carry yuh umbrella just in case!")
    print("  Yuh might get a quick shower.")
    confidence = "medium-high"

elif rain_score >= 3:
    print("  ⛅  PREDICTION: MAYBE A LITTLE DRIZZLE")
    print()
    print("  Could go either way!")
    print("  Might get a likkle drizzle, might stay dry.")
    print("  Keep one eye on di sky!")
    confidence = "medium"

elif rain_score >= 1:
    print("  🌤️  PREDICTION: MOSTLY DRY!")
    print()
    print("  Probably going stay dry today.")
    print("  Maybe a tiny sprinkle but nothing serious.")
    print(f"  Good beach day in {island}!")
    confidence = "low"

else:
    print("  ☀️☀️☀️  PREDICTION: NO RAIN TODAY!")
    print()
    print("  Blue sky all day long!")
    print("  Perfect weather fi go beach, play cricket,")
    print("  or run track!")
    print(f"  Enjoy di sunshine in {island}!")
    confidence = "very low"

print()
print(f"  Confidence dat rain a come: {confidence}")
print()

# ============================================================
# STEP 5: Explain how dis is like REAL AI
# ============================================================

print("-" * 55)
print("  HOW DIS IS LIKE REAL AI:")
print("-" * 55)
print()
print("  1. We COLLECTED DATA (yuh answers to 6 questions)")
print("  2. We gave each answer a SCORE (some signs of rain")
print("     are stronger than others)")
print("  3. We ADDED UP di score")
print("  4. We made a PREDICTION based on di total score")
print()
print("  Real weather AI (like on yuh phone) does di SAME")
print("  ting, but wid THOUSANDS of data points:")
print("  - Satellite images")
print("  - Temperature sensors")
print("  - Ocean buoy data")
print("  - Wind measurements")
print("  - Historical weather records")
print()
print("  Yuh just built yuh FIRST AI! How yuh feel? 🎉")
print()

# ============================================================
# STEP 6: Quiz Time!
# ============================================================

print("=" * 55)
print("  LESSON 5 QUIZ — Test Yuh Knowledge!")
print("=" * 55)
print()

quiz_score = 0

# Quiz Question 1
print("Quiz Question 1:")
print("What does our weather AI use fi make predictions?")
print("  A) Magic")
print("  B) Data from yuh answers to questions")
print("  C) Random guessing")
print("  D) It ask di clouds directly")
q1 = input("Yuh answer (A/B/C/D): ")
if q1.upper().strip() == "B":
    print("CORRECT! AI use DATA fi make predictions! ✅")
    quiz_score += 1
else:
    print("Di answer is B! AI use data (information) fi decide. ❌")
print()

# Quiz Question 2
print("Quiz Question 2:")
print("When is hurricane season in di Caribbean?")
print("  A) January to March")
print("  B) June to November")
print("  C) December only")
print("  D) All year round")
q2 = input("Yuh answer (A/B/C/D): ")
if q2.upper().strip() == "B":
    print("CORRECT! June to November is hurricane season! ✅")
    quiz_score += 1
else:
    print("Di answer is B! June to November — remember dat! ❌")
print()

# Quiz Question 3
print("Quiz Question 3:")
print("In our AI, what does a HIGHER rain score mean?")
print("  A) Less chance of rain")
print("  B) More chance of rain")
print("  C) It going snow")
print("  D) Nothing at all")
q3 = input("Yuh answer (A/B/C/D): ")
if q3.upper().strip() == "B":
    print("CORRECT! Higher score = more likely fi rain! ✅")
    quiz_score += 1
else:
    print("Di answer is B! Higher score means more rain signs! ❌")
print()

# Quiz Question 4
print("Quiz Question 4:")
print("What is di FIRST step in making an AI?")
print("  A) Make a guess right away")
print("  B) Collect data (information)")
print("  C) Turn off di computer")
print("  D) Draw a picture")
q4 = input("Yuh answer (A/B/C/D): ")
if q4.upper().strip() == "B":
    print("CORRECT! First yuh collect data, den yuh analyze it! ✅")
    quiz_score += 1
else:
    print("Di answer is B! Data collection always come first! ❌")
print()

# Quiz Question 5
print("Quiz Question 5:")
print("Why did wi give 'dark clouds' a HIGHER score dan 'morning dew'?")
print("  A) Because clouds are prettier")
print("  B) Because dark clouds are a STRONGER sign of rain")
print("  C) Because dew is boring")
print("  D) No reason, we just pick random numbers")
q5 = input("Yuh answer (A/B/C/D): ")
if q5.upper().strip() == "B":
    print("CORRECT! Some signs are stronger indicators dan others! ✅")
    quiz_score += 1
else:
    print("Di answer is B! In AI, some data is more important — we call")
    print("dat giving it more 'weight'! ❌")
print()

# Show quiz results
print("=" * 55)
print(f"  QUIZ RESULTS: {quiz_score} out of 5")
print("=" * 55)
print()

if quiz_score == 5:
    print("  PERFECT! Yuh is a real AI scientist! 🏆")
elif quiz_score >= 3:
    print("  GREAT JOB! Yuh understand how AI works! 🌟")
elif quiz_score >= 1:
    print("  GOOD TRY! Review di lesson and try again! 📚")
else:
    print("  No worries! Read through di lesson one more time! 💪")

print()
print("=" * 55)
print("  CONGRATULATIONS!")
print()
print(f"  {name}, yuh just completed di PRIMARY/PREP")
print("  level of di Caribbean AI Curriculum!")
print()
print("  Yuh learned:")
print("  ✅ What AI is (Lesson 1)")
print("  ✅ How fi find patterns (Lesson 2)")
print("  ✅ How fi sort and classify (Lesson 3)")
print("  ✅ How fi write yuh first code (Lesson 4)")
print("  ✅ How fi build a simple AI (Lesson 5)")
print()
print("  Yuh ready fi di next level!")
print()
print("  Caribbean AI Curriculum")
print("  Designed by Adrian Dunkley | Adriandunkley.net | FREE")
print("=" * 55)

# ============================================================
# WHAT YUH LEARNED IN DIS LESSON:
# ============================================================
# 1. AI collects DATA (information) fi make decisions
# 2. Different data have different WEIGHTS (importance)
# 3. AI adds up all di evidence fi make a PREDICTION
# 4. More data = better predictions!
# 5. Real weather AI works di same way, just wid MORE data
#
# YUH BUILT YUH FIRST AI! Be proud of yuhself! 🎉
# ============================================================
