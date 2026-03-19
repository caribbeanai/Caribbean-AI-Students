"""
=============================================================================
LESSON 1: Python Basics — Caribbean Style! 🐍🌴
=============================================================================
Caribbean AI Academy — Forms 1-3 (Ages 11-14)
Designed by Adrian Dunkley (https://Adriandunkley.net)
FREE for Caribbean students who want to be AI Engineers, Scientists,
and Entrepreneurs.

Yo! Welcome to yuh FIRST Python lesson!
We gonna learn di building blocks of coding using tings yuh already know —
Caribbean islands, food, people, and culture.

By di end of this lesson, yuh go know:
  - Variables (storing information)
  - Data types (numbers, text, true/false)
  - Lists (collections of tings)
  - Dictionaries (like a real dictionary — look up one ting, find another)
  - Loops (doing tings over and over)
  - If/else (making decisions)
  - Functions (reusable blocks of code)

Just run dis file and read along. Type it out yuhself fi practice!
=============================================================================
"""

# =============================================================================
# PART 1: VARIABLES — Storing Information
# =============================================================================
# A variable is like a box wid a label. Yuh put something inside,
# and yuh can use di label fi get it back later.

print("=" * 60)
print("PART 1: VARIABLES")
print("=" * 60)

# Let's store some Caribbean island populations (approximate, 2024)
jamaica_population = 2_828_000
trinidad_population = 1_405_000
barbados_population = 288_000
guyana_population = 808_000
bahamas_population = 412_000
haiti_population = 11_725_000
dominican_republic_population = 11_229_000
cuba_population = 11_212_000

# Yuh see dat? We just stored 8 numbers in 8 variables!
# Di underscore in numbers (like 2_828_000) is just fi mek it easier fi read.
# Python ignores dem — 2_828_000 is di same as 2828000.

print(f"Jamaica population: {jamaica_population:,}")
print(f"Trinidad & Tobago population: {trinidad_population:,}")
print(f"Haiti population: {haiti_population:,}")

# Variables can hold text too — we call dem "strings"
my_island = "Jamaica"
my_capital = "Kingston"
my_motto = "Out of Many, One People"

print(f"\nI come from {my_island}. Di capital is {my_capital}.")
print(f"Our motto: '{my_motto}'")

# Variables can hold True or False — we call dem "booleans"
is_island = True
has_volcano = False  # Jamaica nuh have active volcano
loves_cricket = True

print(f"\nIs Jamaica an island? {is_island}")
print(f"Does Jamaica have a volcano? {has_volcano}")
print(f"Does Jamaica love cricket? {loves_cricket}")

# Variables can hold decimal numbers — we call dem "floats"
jamaica_area_km2 = 10991.0
barbados_area_km2 = 430.0
dominica_area_km2 = 751.0

print(f"\nJamaica area: {jamaica_area_km2} km²")
print(f"Barbados area: {barbados_area_km2} km²")


# =============================================================================
# PART 2: LISTS — Collections of Tings
# =============================================================================
# A list is like a line of people waiting fi buy patty.
# Each person has a position (starting from 0!).

print("\n" + "=" * 60)
print("PART 2: LISTS")
print("=" * 60)

# List of Caribbean countries
caribbean_countries = [
    "Jamaica", "Trinidad & Tobago", "Barbados", "Guyana",
    "The Bahamas", "Haiti", "Dominican Republic", "Cuba",
    "Antigua & Barbuda", "Belize", "Dominica", "Grenada",
    "St. Kitts & Nevis", "St. Lucia", "St. Vincent & the Grenadines",
    "Suriname", "Puerto Rico", "U.S. Virgin Islands",
    "Cayman Islands", "Bermuda", "Turks & Caicos",
    "Curacao", "Aruba", "Montserrat"
]

print(f"We have {len(caribbean_countries)} Caribbean territories in we list!")
print(f"First country: {caribbean_countries[0]}")       # Index 0 = first
print(f"Last country: {caribbean_countries[-1]}")        # Index -1 = last
print(f"Third country: {caribbean_countries[2]}")        # Index 2 = third

# List of Caribbean fruits — yuh know dem!
caribbean_fruits = ["ackee", "mango", "soursop", "guinep", "june plum",
                    "star apple", "breadfruit", "jackfruit", "papaya",
                    "guava", "passion fruit", "tamarind", "coconut"]

print(f"\nCaribbean fruits: {caribbean_fruits}")
print(f"Number of fruits: {len(caribbean_fruits)}")

# Adding to a list
caribbean_fruits.append("sugar apple")
print(f"After adding sugar apple: {len(caribbean_fruits)} fruits")

# List of Caribbean dishes
caribbean_dishes = ["jerk chicken", "roti", "cou-cou", "pepperpot",
                    "conch fritters", "griot", "mangu", "rice and peas",
                    "doubles", "bake and shark", "callaloo"]

print(f"\nCaribbean dishes: {caribbean_dishes}")

# Slicing a list — get a piece of it
# Dis is like cutting a piece of di line
first_three = caribbean_countries[:3]
print(f"\nFirst 3 countries: {first_three}")

last_three = caribbean_countries[-3:]
print(f"Last 3 territories: {last_three}")


# =============================================================================
# PART 3: DICTIONARIES — Key-Value Pairs
# =============================================================================
# A dictionary is like a phone book. Yuh look up a name (key) and
# find di number (value).

print("\n" + "=" * 60)
print("PART 3: DICTIONARIES")
print("=" * 60)

# Caribbean capitals — country is di key, capital is di value
caribbean_capitals = {
    "Jamaica": "Kingston",
    "Trinidad & Tobago": "Port of Spain",
    "Barbados": "Bridgetown",
    "Guyana": "Georgetown",
    "The Bahamas": "Nassau",
    "Haiti": "Port-au-Prince",
    "Dominican Republic": "Santo Domingo",
    "Cuba": "Havana",
    "Antigua & Barbuda": "St. John's",
    "Belize": "Belmopan",
    "Dominica": "Roseau",
    "Grenada": "St. George's",
    "St. Kitts & Nevis": "Basseterre",
    "St. Lucia": "Castries",
    "St. Vincent & the Grenadines": "Kingstown",
    "Suriname": "Paramaribo",
    "Puerto Rico": "San Juan",
    "Cayman Islands": "George Town",
    "Bermuda": "Hamilton",
    "Curacao": "Willemstad",
    "Aruba": "Oranjestad",
    "Montserrat": "Brades",
    "Turks & Caicos": "Cockburn Town",
}

# Look up a capital
print(f"Capital of Jamaica: {caribbean_capitals['Jamaica']}")
print(f"Capital of Barbados: {caribbean_capitals['Barbados']}")
print(f"Capital of Haiti: {caribbean_capitals['Haiti']}")

# Caribbean island info — nested dictionary (a dictionary inside a dictionary!)
island_info = {
    "Jamaica": {
        "population": 2_828_000,
        "area_km2": 10_991,
        "currency": "Jamaican Dollar (JMD)",
        "languages": ["English", "Jamaican Patois"],
        "famous_for": ["reggae", "jerk chicken", "Blue Mountain Coffee", "sprinting"]
    },
    "Trinidad & Tobago": {
        "population": 1_405_000,
        "area_km2": 5_131,
        "currency": "Trinidad and Tobago Dollar (TTD)",
        "languages": ["English", "Trinidadian Creole"],
        "famous_for": ["Carnival", "soca", "doubles", "steelpan"]
    },
    "Barbados": {
        "population": 288_000,
        "area_km2": 430,
        "currency": "Barbadian Dollar (BBD)",
        "languages": ["English", "Bajan Creole"],
        "famous_for": ["flying fish", "rum", "cricket", "Rihanna"]
    },
}

# Access nested data
print(f"\nJamaica is famous for: {island_info['Jamaica']['famous_for']}")
print(f"Trinidad's currency: {island_info['Trinidad & Tobago']['currency']}")
print(f"Barbados population: {island_info['Barbados']['population']:,}")


# =============================================================================
# PART 4: LOOPS — Doing Tings Over and Over
# =============================================================================
# A loop is like a DJ playing a riddim on repeat.
# It does di same ting multiple times.

print("\n" + "=" * 60)
print("PART 4: LOOPS")
print("=" * 60)

# FOR loop — go through each item in a list
print("Caribbean countries:")
for country in caribbean_countries[:8]:  # Just first 8 fi now
    print(f"  🏝️  {country}")

# FOR loop with index — enumerate gives yuh di position too
print("\nNumbered list of Caribbean fruits:")
for i, fruit in enumerate(caribbean_fruits[:6], start=1):
    print(f"  {i}. {fruit}")

# Loop through a dictionary
print("\nCaribbean Capitals:")
for country, capital in list(caribbean_capitals.items())[:6]:
    print(f"  {country} → {capital}")

# WHILE loop — keep going until a condition is met
print("\nCountdown to Carnival:")
countdown = 5
while countdown > 0:
    print(f"  {countdown}...")
    countdown -= 1  # Same as: countdown = countdown - 1
print("  🎉 CARNIVAL TIME! 🎉")

# Loop with a calculation — total Caribbean population
populations = {
    "Jamaica": 2_828_000, "Trinidad & Tobago": 1_405_000,
    "Barbados": 288_000, "Guyana": 808_000,
    "The Bahamas": 412_000, "Haiti": 11_725_000,
    "Dominican Republic": 11_229_000, "Cuba": 11_212_000,
    "Antigua & Barbuda": 100_000, "Belize": 441_000,
    "Dominica": 73_000, "Grenada": 125_000,
    "St. Kitts & Nevis": 48_000, "St. Lucia": 180_000,
    "St. Vincent & the Grenadines": 101_000, "Suriname": 618_000,
}

total_population = 0
for country, pop in populations.items():
    total_population += pop

print(f"\nTotal Caribbean population (selected): {total_population:,}")


# =============================================================================
# PART 5: IF / ELSE — Making Decisions
# =============================================================================
# If/else is like asking a question and doing something based on di answer.
# "If it raining, carry umbrella. Else, wear shades."

print("\n" + "=" * 60)
print("PART 5: IF / ELSE")
print("=" * 60)

# Simple if/else
temperature = 32  # Celsius, typical Caribbean day!

if temperature > 30:
    print(f"It's {temperature}°C — Hot hot hot! 🔥 Drink plenty water!")
elif temperature > 25:
    print(f"It's {temperature}°C — Nice Caribbean weather! ☀️")
elif temperature > 20:
    print(f"It's {temperature}°C — Likkle cool breeze today.")
else:
    print(f"It's {temperature}°C — Cold? In di Caribbean?! Check di AC! 😂")

# If with lists
country = "Jamaica"
if country in caribbean_countries:
    print(f"\n{country} is a Caribbean country! 🏝️")
else:
    print(f"\n{country} is NOT in we Caribbean list.")

# Checking population size
for country, pop in populations.items():
    if pop > 5_000_000:
        category = "Large"
    elif pop > 500_000:
        category = "Medium"
    else:
        category = "Small"

    # Only print a few examples
    if country in ["Jamaica", "Haiti", "Barbados", "Dominica"]:
        print(f"  {country}: {pop:,} — {category} Caribbean nation")


# =============================================================================
# PART 6: FUNCTIONS — Reusable Blocks of Code
# =============================================================================
# A function is like a recipe. Yuh write it once, and yuh can use it
# anytime yuh want.

print("\n" + "=" * 60)
print("PART 6: FUNCTIONS")
print("=" * 60)


def greet_caribbean(name, island):
    """Greet somebody Caribbean style!"""
    return f"Wah gwaan, {name}! Big up {island}! 🇯🇲🌴"


def convert_jmd_to_usd(jmd_amount, exchange_rate=155.0):
    """Convert Jamaican Dollars to US Dollars.
    Default exchange rate is approximate — it change every day!"""
    usd = jmd_amount / exchange_rate
    return round(usd, 2)


def classify_island_size(area_km2):
    """Classify a Caribbean island by its area."""
    if area_km2 > 10000:
        return "Large island"
    elif area_km2 > 1000:
        return "Medium island"
    elif area_km2 > 300:
        return "Small island"
    else:
        return "Tiny island"


def calculate_population_density(population, area_km2):
    """Calculate how many people per square kilometer."""
    density = population / area_km2
    return round(density, 1)


# Using our functions!
print(greet_caribbean("Keisha", "Trinidad"))
print(greet_caribbean("Marcus", "Barbados"))

# Currency conversion
patty_price_jmd = 250
print(f"\nA patty cost J${patty_price_jmd}")
print(f"That's US${convert_jmd_to_usd(patty_price_jmd)}")

# Island classification
print(f"\nJamaica ({10991} km²): {classify_island_size(10991)}")
print(f"Barbados ({430} km²): {classify_island_size(430)}")
print(f"Montserrat ({102} km²): {classify_island_size(102)}")
print(f"Cuba ({109884} km²): {classify_island_size(109884)}")

# Population density
print(f"\nJamaica density: {calculate_population_density(2_828_000, 10_991)} people/km²")
print(f"Barbados density: {calculate_population_density(288_000, 430)} people/km²")
print(f"Barbados is one of di most densely populated places in di world!")


# =============================================================================
# PART 7: PUTTING IT ALL TOGETHER — Mini Project
# =============================================================================
# Let's combine everything we learned!

print("\n" + "=" * 60)
print("PART 7: PUTTING IT ALL TOGETHER")
print("=" * 60)


def caribbean_report(country_data):
    """Generate a report fi a Caribbean country."""
    print(f"\n📊 CARIBBEAN COUNTRY REPORT")
    print(f"{'='*40}")
    for country, data in country_data.items():
        density = calculate_population_density(data["population"], data["area_km2"])
        size = classify_island_size(data["area_km2"])
        print(f"\n🏝️  {country}")
        print(f"   Population: {data['population']:,}")
        print(f"   Area: {data['area_km2']:,} km²")
        print(f"   Density: {density} people/km²")
        print(f"   Classification: {size}")
        print(f"   Currency: {data['currency']}")
        print(f"   Languages: {', '.join(data['languages'])}")
        print(f"   Famous for: {', '.join(data['famous_for'])}")


# Run di report!
caribbean_report(island_info)

# BONUS: List comprehension — a shorter way fi mek lists
# Get all countries with population over 1 million
big_countries = [c for c, p in populations.items() if p > 1_000_000]
print(f"\nCaribbean countries with 1M+ population: {big_countries}")

# Get all country names in UPPERCASE
upper_countries = [c.upper() for c in caribbean_countries[:5]]
print(f"First 5 countries (uppercase): {upper_countries}")

# Calculate average population
avg_pop = sum(populations.values()) / len(populations)
print(f"Average population of listed countries: {avg_pop:,.0f}")


print("\n" + "=" * 60)
print("🎉 CONGRATS! Yuh finish Lesson 1!")
print("Now try di quiz below (check di comments at di bottom).")
print("=" * 60)

# 🧩 Puzzle Piece 2/12: The second word is a type of Power _____ (think morphin time!)

# =============================================================================
# QUIZ TIME! 🧠
# =============================================================================
# Try answer dese questions. Write yuh answers in a separate Python file
# or in yuh notebook. Check quiz_answers.md when yuh done!
#
# Q1: What data type is the variable: my_island = "Jamaica"?
#     a) integer  b) float  c) string  d) boolean
#
# Q2: If caribbean_fruits = ["ackee", "mango", "soursop", "guinep"],
#     what does caribbean_fruits[2] return?
#     a) ackee  b) mango  c) soursop  d) guinep
#
# Q3: What does len(caribbean_countries) tell yuh?
#     a) The last country  b) The first country
#     c) The number of items  d) The biggest country
#
# Q4: In a dictionary, what do we call "Jamaica" in:
#     caribbean_capitals["Jamaica"] = "Kingston"?
#     a) value  b) key  c) index  d) variable
#
# Q5: What will this print?
#     for i in range(3):
#         print(i)
#     a) 1 2 3  b) 0 1 2  c) 0 1 2 3  d) 1 2
#
# Q6: What does the 'append' method do to a list?
#     a) Remove the last item  b) Sort the list
#     c) Add an item to the end  d) Count the items
#
# Q7: Write a function called `island_greeting` that takes a country name
#     and returns: "Welcome to [country]! Enjoy di vibes!"
#     (Try it yuhself before checking the answer!)
#
# Q8: What is the output of:
#     x = 10
#     if x > 5:
#         print("Big")
#     else:
#         print("Small")
#     a) Big  b) Small  c) 10  d) Error
#
# Q9: Create a dictionary with 3 Caribbean countries and their populations.
#     Then use a loop to print each one.
#
# Q10: What Caribbean sector could use Python lists to store data?
#      Name TWO sectors and explain what data they would store.
#      (Think: tourism, agriculture, fishing, sports, healthcare, energy)
# =============================================================================
