# =============================================================================
# Lesson 4: Simple Coding — Caribbean Island Guessing Game! 🏝️
# =============================================================================
# Caribbean AI Curriculum — Primary/Prep School (Ages 6-11)
# Designed by Adrian Dunkley | Adriandunkley.net | FREE
#
# HOW TO RUN THIS PROGRAM:
# 1. Make sure Python is installed on yuh computer
# 2. Open a terminal (command prompt)
# 3. Type: python lesson_04_simple_coding.py
# 4. Follow di instructions on screen!
#
# WHAT THIS PROGRAM DOES:
# It gives yuh clues about a Caribbean island, and yuh have to guess
# which island it is! Just like how AI uses clues (data) to figure
# things out!
#
# 🧩 Puzzle Piece 1/12: The first word starts with a color found in traffic lights and Jamaican flag
# =============================================================================

# --- We start by printing a welcome message ---
# "print" means "show dis on the screen"
print("=" * 60)
print("  WELCOME TO DI CARIBBEAN ISLAND GUESSING GAME!")
print("  Caribbean AI Curriculum — Lesson 4")
print("  Designed by Adrian Dunkley | Adriandunkley.net | FREE")
print("=" * 60)
print()
print("Wah gwaan! Mi going give yuh clues about a Caribbean island.")
print("Yuh job is fi guess which island it is!")
print("Yuh get 3 clues fi each island. Try fi guess early!")
print()

# --- This variable keeps track of yuh score ---
# A "variable" is like a box dat holds a number or word
score = 0

# --- This variable tracks how many questions we ask ---
total_questions = 0

# =============================================
# ISLAND 1: JAMAICA
# =============================================
print("-" * 40)
print("ISLAND #1")
print("-" * 40)

# Clue 1
print("Clue 1: Dis island is famous fi sprinting — Usain Bolt come from here!")
guess = input("Yuh guess? (type di island name): ")

# "if" checks if something is true
# ".lower()" makes everything lowercase so "Jamaica" and "jamaica" both work
# ".strip()" removes extra spaces
if guess.lower().strip() == "jamaica":
    print("YES! Yuh get it on di first clue! Amazing! +3 points!")
    score = score + 3
else:
    # If dem never get it, give another clue
    print("Not quite! Here is another clue...")
    print("Clue 2: Dis island is home to reggae music and Bob Marley!")
    guess = input("Yuh guess? (type di island name): ")

    if guess.lower().strip() == "jamaica":
        print("YES! Yuh get it! Nice! +2 points!")
        score = score + 2
    else:
        print("One more clue!")
        print("Clue 3: Di national dish is ackee and saltfish!")
        guess = input("Yuh guess? (type di island name): ")

        if guess.lower().strip() == "jamaica":
            print("Yuh get it! +1 point!")
            score = score + 1
        else:
            print("Di answer was JAMAICA! No worries, keep trying!")

total_questions = total_questions + 1
print()

# =============================================
# ISLAND 2: TRINIDAD AND TOBAGO
# =============================================
print("-" * 40)
print("ISLAND #2")
print("-" * 40)

print("Clue 1: Dis country is famous fi Carnival — di biggest party in di Caribbean!")
guess = input("Yuh guess? (type di country name): ")

# We check for different ways people might type it
answer = guess.lower().strip()
if answer == "trinidad" or answer == "trinidad and tobago" or answer == "trinidad & tobago":
    print("YES! First clue! Brilliant! +3 points!")
    score = score + 3
else:
    print("Not yet! Try again...")
    print("Clue 2: Steelpan (steel drum) was invented here!")
    guess = input("Yuh guess? ")
    answer = guess.lower().strip()

    if answer == "trinidad" or answer == "trinidad and tobago" or answer == "trinidad & tobago":
        print("Correct! +2 points!")
        score = score + 2
    else:
        print("Last clue!")
        print("Clue 3: Doubles (a famous street food wid channa) come from here!")
        guess = input("Yuh guess? ")
        answer = guess.lower().strip()

        if answer == "trinidad" or answer == "trinidad and tobago" or answer == "trinidad & tobago":
            print("Yuh get it! +1 point!")
            score = score + 1
        else:
            print("Di answer was TRINIDAD AND TOBAGO! Yuh go get di next one!")

total_questions = total_questions + 1
print()

# =============================================
# ISLAND 3: BARBADOS
# =============================================
print("-" * 40)
print("ISLAND #3")
print("-" * 40)

print("Clue 1: Dis island is known as 'di land of di flying fish!'")
guess = input("Yuh guess? ")

if guess.lower().strip() == "barbados":
    print("YES! First try! +3 points!")
    score = score + 3
else:
    print("Nah, try again!")
    print("Clue 2: Rihanna was born on dis island!")
    guess = input("Yuh guess? ")

    if guess.lower().strip() == "barbados":
        print("Correct! +2 points!")
        score = score + 2
    else:
        print("One more!")
        print("Clue 3: Kensington Oval, a famous cricket ground, is here!")
        guess = input("Yuh guess? ")

        if guess.lower().strip() == "barbados":
            print("Yuh get it! +1 point!")
            score = score + 1
        else:
            print("Di answer was BARBADOS!")

total_questions = total_questions + 1
print()

# =============================================
# ISLAND 4: THE BAHAMAS
# =============================================
print("-" * 40)
print("ISLAND #4")
print("-" * 40)

print("Clue 1: Dis country has over 700 islands and cays!")
guess = input("Yuh guess? ")

answer = guess.lower().strip()
if answer == "bahamas" or answer == "the bahamas":
    print("First clue! Yuh smart! +3 points!")
    score = score + 3
else:
    print("Not yet!")
    print("Clue 2: Nassau is di capital, and Junkanoo is di famous festival!")
    guess = input("Yuh guess? ")
    answer = guess.lower().strip()

    if answer == "bahamas" or answer == "the bahamas":
        print("Nice one! +2 points!")
        score = score + 2
    else:
        print("Last chance!")
        print("Clue 3: It famous fi conch salad, pink sand beaches, and swimming pigs!")
        guess = input("Yuh guess? ")
        answer = guess.lower().strip()

        if answer == "bahamas" or answer == "the bahamas":
            print("Yuh get it! +1 point!")
            score = score + 1
        else:
            print("Di answer was THE BAHAMAS!")

total_questions = total_questions + 1
print()

# =============================================
# ISLAND 5: GRENADA
# =============================================
print("-" * 40)
print("ISLAND #5")
print("-" * 40)

print("Clue 1: Dis island is called 'The Spice Isle' because it grows")
print("         nutmeg, cinnamon, cloves, and more!")
guess = input("Yuh guess? ")

if guess.lower().strip() == "grenada":
    print("Yes! Yuh on FIRE! +3 points!")
    score = score + 3
else:
    print("Not quite!")
    print("Clue 2: Yuh can find a nutmeg right on di national flag!")
    guess = input("Yuh guess? ")

    if guess.lower().strip() == "grenada":
        print("Correct! +2 points!")
        score = score + 2
    else:
        print("One more clue!")
        print("Clue 3: Di capital is St. George's, and 'Oil Down' is di national dish!")
        guess = input("Yuh guess? ")

        if guess.lower().strip() == "grenada":
            print("Yuh get it! +1 point!")
            score = score + 1
        else:
            print("Di answer was GRENADA!")

total_questions = total_questions + 1
print()

# =============================================
# FINAL SCORE
# =============================================
# Now we calculate and show di final score!
# Di maximum possible score is 15 (3 points x 5 islands)

print("=" * 60)
print("  GAME OVER! Let's see how yuh do!")
print("=" * 60)
print()
print("Yuh score: " + str(score) + " out of 15!")
print()

# We use "if/elif/else" to give different messages based on the score
# "elif" is short for "else if"
if score >= 13:
    print("🌟 CARIBBEAN CHAMPION! Yuh know yuh islands inside out!")
    print("   Yuh could teach AI about di Caribbean!")
elif score >= 9:
    print("🏆 GREAT JOB! Yuh know nuff about di Caribbean!")
    print("   Keep learning and yuh will be a champion!")
elif score >= 5:
    print("👍 GOOD EFFORT! Yuh learning!")
    print("   Read up on yuh Caribbean islands and try again!")
else:
    print("📚 KEEP TRYING! Every expert was once a beginner!")
    print("   Ask yuh teacher or parents about di Caribbean islands!")

print()
print("-" * 60)
print("WHAT YUH JUST LEARNED:")
print("-" * 60)
print("1. 'print' shows a message on di screen")
print("2. 'input' lets di computer ask yuh a question")
print("3. 'if/else' lets di computer make DECISIONS (like AI!)")
print("4. Variables (like 'score') store information")
print("5. Di computer checks yuh answer against di correct one")
print()
print("Dis is how AI works at a basic level:")
print("  INPUT (yuh guess) -> PROCESS (check if correct) -> OUTPUT (response)")
print()
print("=" * 60)
print("  Caribbean AI Curriculum — Designed by Adrian Dunkley")
print("  Adriandunkley.net | FREE fi everyone")
print("=" * 60)
