# =============================================================================
# Lesson 5: My First AI — Caribbean Weather Predictor! 🌦️🌴
# =============================================================================
# Caribbean AI Curriculum — Primary/Prep School (Ages 6-11)
# Designed by Adrian Dunkley | Adriandunkley.net | FREE
#
# HOW TO RUN THIS PROGRAM:
# 1. Make sure Python is installed on yuh computer
# 2. Open a terminal (command prompt)
# 3. Type: python lesson_05_my_first_ai.py
# 4. Answer di questions and watch di "AI" predict di weather!
#
# WHAT THIS PROGRAM DOES:
# This is yuh FIRST AI! Well, it's a simple version.
# Real AI learns from millions of data points, but dis program
# shows yuh HOW AI thinks — using yes/no questions to make decisions.
# This is called a "Decision Tree" — one of the simplest forms of AI!
#
# CARIBBEAN WEATHER CONTEXT:
# Di Caribbean has two main seasons:
#   - Dry Season (January to May) — less rain, cooler breeze
#   - Wet/Rainy Season (June to December) — more rain, hurricanes possible
# Rain in di Caribbean often come quick — bright sunshine one minute,
# heavy rain di next! We call dat "sunshower" or "liquid sunshine"!
# =============================================================================

# --- Welcome Message ---
print("=" * 60)
print("  MY FIRST AI: CARIBBEAN WEATHER PREDICTOR!")
print("  Caribbean AI Curriculum — Lesson 5")
print("  Designed by Adrian Dunkley | Adriandunkley.net | FREE")
print("=" * 60)
print()
print("Wah gwaan! Welcome to yuh FIRST AI program!")
print()
print("Dis program going ask yuh some questions about di weather")
print("RIGHT NOW where yuh are, and den it going PREDICT if rain")
print("a come! Just like how a real AI weather app works!")
print()
print("Answer each question wid 'yes' or 'no'.")
print()
print("(Psst... if yuh know di secret word, type it anytime fi a surprise!)")
print()

# --- We collect information, just like a real AI collects data ---

# Question 1: Clouds
print("-" * 40)
print("QUESTION 1: Look outside right now!")
answer1 = input("Do yuh see dark clouds in di sky? (yes/no): ")

# Secret Easter Egg! If someone types "caribbean" they get a special message!
if answer1.lower().strip() == "caribbean":
    print()
    print("*" * 50)
    print("  YUH FIND DI SECRET! 🎉")
    print("  'Caribbean' is di magic word!")
    print("  Fun fact: Di word 'Caribbean' comes from")
    print("  di Kalinago and Taino people — di ORIGINAL")
    print("  people of dese beautiful islands!")
    print("  Dem was here LONG before Columbus!")
    print("  Respect to di indigenous Caribbean people! 🙏")
    print("*" * 50)
    print()
    print("Now let's continue wid di weather prediction!")
    print()
    answer1 = input("Do yuh see dark clouds in di sky? (yes/no): ")

# We use a "rain score" — di higher it is, di more likely rain a come
# This is like how AI gives things a "probability score"!
rain_score = 0

# If dark clouds, dat is a BIG sign of rain
if answer1.lower().strip() == "yes":
    rain_score = rain_score + 3  # Add 3 because clouds are a strong sign
    print("Hmm, dark clouds... dat is a sign! ☁️")
else:
    print("No dark clouds — good sign so far! ☀️")

print()

# Question 2: Wind
print("-" * 40)
print("QUESTION 2: Step outside or look at di trees!")
answer2 = input("Is di wind blowing hard? (yes/no): ")

# Check for secret word again
if answer2.lower().strip() == "caribbean":
    print()
    print("*" * 50)
    print("  YUH FIND DI SECRET! 🎉")
    print("  Caribbean magic activated!")
    print("*" * 50)
    print()
    answer2 = input("Is di wind blowing hard? (yes/no): ")

if answer2.lower().strip() == "yes":
    rain_score = rain_score + 2  # Wind is a medium sign
    print("Strong wind can push rain clouds yuh way! 💨")
else:
    print("Calm breeze — nice! 🌿")

print()

# Question 3: Humidity (explained fi kids)
print("-" * 40)
print("QUESTION 3: Think about how di air feels!")
print("(Humid = di air feel sticky and heavy, like before rain)")
answer3 = input("Does di air feel sticky and heavy? (yes/no): ")

if answer3.lower().strip() == "caribbean":
    print()
    print("*" * 50)
    print("  YUH FIND DI SECRET! 🎉")
    print("*" * 50)
    print()
    answer3 = input("Does di air feel sticky and heavy? (yes/no): ")

if answer3.lower().strip() == "yes":
    rain_score = rain_score + 2
    print("Sticky air means nuff moisture — rain might a come! 💧")
else:
    print("Air feel light — less chance of rain!")

print()

# Question 4: Season
print("-" * 40)
print("QUESTION 4: Think about what month it is!")
print("(In di Caribbean, rainy season is June to December)")
answer4 = input("Are we in rainy season right now? (June-December = yes): ")

if answer4.lower().strip() == "caribbean":
    print()
    print("*" * 50)
    print("  YUH FIND DI SECRET! 🎉")
    print("*" * 50)
    print()
    answer4 = input("Are we in rainy season right now? (June-December = yes): ")

if answer4.lower().strip() == "yes":
    rain_score = rain_score + 2
    print("Rainy season means more chance of rain! 🌧️")
else:
    print("Dry season — rain less likely but still possible! ☀️")

print()

# Question 5: Recent rain
print("-" * 40)
print("QUESTION 5: Think about yesterday and today!")
answer5 = input("Did it rain yesterday or earlier today? (yes/no): ")

if answer5.lower().strip() == "caribbean":
    print()
    print("*" * 50)
    print("  YUH FIND DI SECRET! 🎉")
    print("*" * 50)
    print()
    answer5 = input("Did it rain yesterday or earlier today? (yes/no): ")

if answer5.lower().strip() == "yes":
    rain_score = rain_score + 1
    print("If it rain recently, di ground wet and more moisture inna di air!")
else:
    print("No recent rain — ground dry!")

print()

# =============================================
# THE AI PREDICTION!
# =============================================
# Now we use di rain_score to make a prediction
# Maximum possible score = 3 + 2 + 2 + 2 + 1 = 10
# We use "if/elif/else" to decide — this is a DECISION TREE!

print("=" * 60)
print("  DI AI IS THINKING...")
print("=" * 60)
print()
print("Analyzing yuh answers...")
print("Checking cloud data... ☁️")
print("Checking wind data... 💨")
print("Checking humidity... 💧")
print("Checking season... 📅")
print("Checking recent weather... 🔍")
print()
print("Rain Score: " + str(rain_score) + " out of 10")
print()

# Decision time!
if rain_score >= 8:
    print("🌧️🌧️🌧️ PREDICTION: RAIN IS ALMOST CERTAIN! 🌧️🌧️🌧️")
    print()
    print("Mi prediction: HEAVY RAIN a come! Carry yuh umbrella,")
    print("put on yuh rain boots, and DON'T leave yuh clothes on di line!")
    print("If yuh in Jamaica, dis is wah we call 'wash out' weather!")
    print("If yuh in Trinidad, better cancel di lime!")
elif rain_score >= 5:
    print("🌦️🌦️ PREDICTION: RAIN IS LIKELY! 🌦️🌦️")
    print()
    print("Mi prediction: Good chance rain a come, maybe inna di afternoon.")
    print("Yuh know how it go inna di Caribbean — sunshine one minute,")
    print("rain di next! We call dat 'liquid sunshine!'")
    print("Better keep yuh umbrella close, just in case!")
elif rain_score >= 3:
    print("⛅ PREDICTION: MAYBE A LITTLE RAIN ⛅")
    print()
    print("Mi prediction: Yuh MIGHT get a likkle shower, but nuttin serious.")
    print("In Barbados dem would say: 'It might spit a likkle.'")
    print("Carry a light jacket just in case!")
else:
    print("☀️☀️☀️ PREDICTION: NO RAIN TODAY! ☀️☀️☀️")
    print()
    print("Mi prediction: Dry and sunny! Perfect day fi go beach,")
    print("play cricket, or lime wid yuh friends!")
    print("Enjoy di Caribbean sunshine! 🏖️")

print()

# =============================================
# HOW THIS AI WORKS — EXPLANATION
# =============================================
print("=" * 60)
print("  HOW DIS 'AI' WORKS — WHAT YUH JUST LEARNED!")
print("=" * 60)
print()
print("Dis program is a SIMPLE version of how real AI works:")
print()
print("1. COLLECT DATA: We asked yuh 5 questions (dat is di 'data')")
print("2. SCORE DI DATA: Each answer got a number (di 'weight')")
print("   - Dark clouds = 3 points (strong sign of rain)")
print("   - Wind, humidity, season = 2 points each (medium signs)")
print("   - Recent rain = 1 point (small sign)")
print("3. MAKE A DECISION: We add up all di points and decide")
print("   - 8-10 = Definitely rain")
print("   - 5-7 = Probably rain")
print("   - 3-4 = Maybe rain")
print("   - 0-2 = No rain")
print()
print("REAL AI weather apps do di SAME thing, but wid THOUSANDS")
print("of data points instead of just 5!")
print()

# =============================================
# LESSON 5 QUIZ (printed on screen)
# =============================================
print("=" * 60)
print("  LESSON 5 QUIZ!")
print("=" * 60)
print()
print("Answer dese in yuh notebook, den check quiz_answers.md!")
print()
print("Question 1: What is a 'Decision Tree' in AI?")
print("  A) A real tree dat makes decisions")
print("  B) A way of making decisions by asking yes/no questions")
print("  C) A tree yuh find in di Caribbean forest")
print("  D) A computer game about trees")
print()
print("Question 2: In dis weather predictor, which sign got di MOST")
print("            points (was di strongest sign of rain)?")
print("  A) Recent rain (1 point)")
print("  B) Strong wind (2 points)")
print("  C) Dark clouds (3 points)")
print("  D) Dem all di same")
print()
print("Question 3: What are di two main weather seasons in di Caribbean?")
print("  A) Summer and Winter")
print("  B) Dry Season and Rainy Season")
print("  C) Hot Season and Cold Season")
print("  D) Spring and Fall")
print()
print("Question 4: What does 'collecting data' mean in AI?")
print("  A) Picking up garbage")
print("  B) Gathering information dat di AI can use fi make decisions")
print("  C) Collecting stickers")
print("  D) Downloading games")
print()
print("Question 5: How is dis weather predictor DIFFERENT from a real AI")
print("            weather app?")
print("  A) Real AI uses thousands of data points, we only used 5")
print("  B) Real AI learns and improves over time")
print("  C) Real AI uses satellites, sensors, and historical data")
print("  D) All of di above!")
print()

# =============================================
# GOODBYE MESSAGE
# =============================================
print("=" * 60)
print("  YUH JUST BUILT YUH FIRST AI! 🎉")
print("=" * 60)
print()
print("Congratulations! Yuh now understand how AI makes decisions!")
print("Dis 'decision tree' approach is used in REAL AI systems")
print("all over di world — including right here inna di Caribbean!")
print()
print("Fun Challenge: Try running di program again wid DIFFERENT")
print("answers and see how di prediction change!")
print()
print("=" * 60)
print("  Caribbean AI Curriculum — Designed by Adrian Dunkley")
print("  Adriandunkley.net | FREE fi everyone")
print("=" * 60)
