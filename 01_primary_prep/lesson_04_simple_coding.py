"""
=============================================================
  Lesson 4: Simple Coding — Caribbean Island Guessing Game!
=============================================================

  Caribbean AI Curriculum — Primary/Prep School (Ages 6-11)
  Designed by Adrian Dunkley | Adriandunkley.net | FREE

  HOW TO RUN THIS PROGRAM:
  1. Make sure Python is installed on yuh computer
  2. Open a terminal (command prompt)
  3. Type: python lesson_04_simple_coding.py
  4. Follow di instructions on screen!

  WHAT THIS PROGRAM DOES:
  - It give yuh clues about a Caribbean island
  - Yuh try fi guess which island it is!
  - Yuh get 3 tries fi each island
  - At di end, yuh see yuh score!
"""

# 🧩 Puzzle Piece 1/12: The first word starts with a color found in traffic lights and Jamaican flag

# ============================================================
# STEP 1: We create a welcome message!
# "print" means "show dis on di screen"
# ============================================================

print("=" * 55)
print("  WELCOME TO DI CARIBBEAN ISLAND GUESSING GAME!")
print("  =============================================")
print()
print("  Designed by Adrian Dunkley | Adriandunkley.net")
print("  Caribbean AI Curriculum — FREE fi everyone!")
print()
print("  I going give yuh clues about Caribbean islands.")
print("  Yuh job is fi guess which island it is!")
print("  Yuh get 3 tries fi each island. Good luck!")
print("=" * 55)
print()

# ============================================================
# STEP 2: We set up di score counter
# A "variable" is like a box dat hold a value
# We start wid 0 points
# ============================================================

score = 0  # Dis hold how many islands yuh guess correctly
total_islands = 5  # We have 5 islands fi guess

# ============================================================
# STEP 3: We ask di player if dem ready!
# "input" means "wait fi di person fi type something"
# ============================================================

player_name = input("First, tell mi yuh name: ")
print()
print(f"Nice fi meet yuh, {player_name}! Let's start!")
print()

# ============================================================
# ISLAND 1: Jamaica
# ============================================================

print("-" * 40)
print("ISLAND #1")
print("-" * 40)
print()
print("CLUE 1: Dis island is home to di fastest man in di world — Usain Bolt!")
print("CLUE 2: Dem famous fi reggae music and Bob Marley.")
print("CLUE 3: Di capital city is Kingston.")
print()

# We give di player 3 tries
tries = 3
got_it = False  # Dis keep track of whether dem guess right

while tries > 0 and not got_it:
    guess = input(f"What island is it? ({tries} tries left): ")

    # We use .lower() and .strip() fi handle different ways people type
    # "Jamaica", "jamaica", "JAMAICA", " jamaica " all work!
    if guess.lower().strip() == "jamaica":
        print("YES! Yuh get it! Big up yuhself! 🇯🇲")
        print()
        score = score + 1  # Add 1 to di score
        got_it = True
    else:
        tries = tries - 1  # Lose one try
        if tries > 0:
            print(f"Nah, try again! Yuh have {tries} tries left.")
        else:
            print("Di answer was JAMAICA! No worries, let's keep going!")
    print()

# ============================================================
# ISLAND 2: Trinidad and Tobago
# ============================================================

print("-" * 40)
print("ISLAND #2")
print("-" * 40)
print()
print("CLUE 1: Dis country is TWO islands together!")
print("CLUE 2: Dem famous fi Carnival — one of di biggest parties in di world!")
print("CLUE 3: Steelpan music was invented here.")
print()

tries = 3
got_it = False

while tries > 0 and not got_it:
    guess = input(f"What island is it? ({tries} tries left): ")

    # We accept different ways fi write it
    answer = guess.lower().strip()
    if answer in ["trinidad", "tobago", "trinidad and tobago",
                   "trinidad & tobago", "trinidad and tobago",
                   "t&t", "trini"]:
        print("YES! Trini to di bone! Big up! 🇹🇹")
        print()
        score = score + 1
        got_it = True
    else:
        tries = tries - 1
        if tries > 0:
            print(f"Nah, try again! Yuh have {tries} tries left.")
        else:
            print("Di answer was TRINIDAD AND TOBAGO! On to di next one!")
    print()

# ============================================================
# ISLAND 3: Barbados
# ============================================================

print("-" * 40)
print("ISLAND #3")
print("-" * 40)
print()
print("CLUE 1: Dis island is known as 'Little England.'")
print("CLUE 2: Singer Rihanna was born here!")
print("CLUE 3: Dem LOVE cricket — Kensington Oval is der famous ground.")
print()

tries = 3
got_it = False

while tries > 0 and not got_it:
    guess = input(f"What island is it? ({tries} tries left): ")

    answer = guess.lower().strip()
    if answer in ["barbados", "bim", "bimshire"]:
        print("YES! Correct! Bim represent! 🇧🇧")
        print()
        score = score + 1
        got_it = True
    else:
        tries = tries - 1
        if tries > 0:
            print(f"Nah, try again! Yuh have {tries} tries left.")
        else:
            print("Di answer was BARBADOS! Keep going, yuh doing great!")
    print()

# ============================================================
# ISLAND 4: Grenada
# ============================================================

print("-" * 40)
print("ISLAND #4")
print("-" * 40)
print()
print("CLUE 1: Dis island is called 'Di Spice Isle.'")
print("CLUE 2: Nutmeg is on di flag!")
print("CLUE 3: Olympic champion Kirani James is from here.")
print()

tries = 3
got_it = False

while tries > 0 and not got_it:
    guess = input(f"What island is it? ({tries} tries left): ")

    answer = guess.lower().strip()
    if answer in ["grenada", "grenade", "spice isle"]:
        print("YES! Spice Isle massive! Well done! 🇬🇩")
        print()
        score = score + 1
        got_it = True
    else:
        tries = tries - 1
        if tries > 0:
            print(f"Nah, try again! Yuh have {tries} tries left.")
        else:
            print("Di answer was GRENADA! One more island to go!")
    print()

# ============================================================
# ISLAND 5: The Bahamas
# ============================================================

print("-" * 40)
print("ISLAND #5 — LAST ONE!")
print("-" * 40)
print()
print("CLUE 1: Dis country have OVER 700 islands!")
print("CLUE 2: Di water so clear yuh can see straight to di bottom.")
print("CLUE 3: Olympic sprinter Shaunae Miller-Uibo is from here.")
print()

tries = 3
got_it = False

while tries > 0 and not got_it:
    guess = input(f"What island is it? ({tries} tries left): ")

    answer = guess.lower().strip()
    if answer in ["bahamas", "the bahamas", "bahamaland"]:
        print("YES! Bahamas massive! Yuh a real Caribbean expert! 🇧🇸")
        print()
        score = score + 1
        got_it = True
    else:
        tries = tries - 1
        if tries > 0:
            print(f"Nah, try again! Yuh have {tries} tries left.")
        else:
            print("Di answer was THE BAHAMAS!")
    print()

# ============================================================
# STEP 4: Show di final score!
# ============================================================

print("=" * 55)
print(f"  GAME OVER, {player_name}!")
print(f"  Yuh scored {score} out of {total_islands} islands!")
print()

# We use if/elif/else fi give different messages based on di score
# Dis is like simple AI — di computer DECIDE what fi say!
if score == 5:
    print("  PERFECT SCORE! Yuh is a CARIBBEAN CHAMPION! 🏆")
    print("  Yuh know yuh islands like di back of yuh hand!")
elif score >= 3:
    print("  GREAT JOB! Yuh know yuh Caribbean geography well! 🌟")
    print("  Keep learning and yuh going get perfect next time!")
elif score >= 1:
    print("  GOOD TRY! Yuh getting there! 📚")
    print("  Read up on yuh Caribbean islands and try again!")
else:
    print("  NO WORRIES! Every expert was a beginner once! 💪")
    print("  Study yuh Caribbean islands and come back stronger!")

print()
print("  Thank yuh fi playing! 🌴")
print("  Remember: coding is FUN and yuh can do ANYTHING wid it!")
print()
print("  Caribbean AI Curriculum")
print("  Designed by Adrian Dunkley | Adriandunkley.net | FREE")
print("=" * 55)

# ============================================================
# WHAT YUH LEARNED IN DIS LESSON:
# ============================================================
# 1. print() — shows text on di screen
# 2. input() — lets di user type something
# 3. Variables — boxes dat hold values (like "score" and "tries")
# 4. if/elif/else — di computer makes decisions (like simple AI!)
# 5. while loops — doing something over and over until a condition met
# 6. .lower() and .strip() — cleaning up text so di computer understand it
#
# DESE ARE DI BUILDING BLOCKS OF AI!
# AI programs use all dese same tools, just in bigger, fancier ways.
# ============================================================
