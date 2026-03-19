"""
=============================================================================
CAPSTONE PROJECT: Caribbean Quiz Bot! 🤖🌴🧠
=============================================================================
Caribbean AI Academy — Forms 1-3 (Ages 11-14)
Designed by Adrian Dunkley (https://Adriandunkley.net)
FREE for Caribbean students who want to be AI Engineers, Scientists,
and Entrepreneurs.

FINAL PROJECT! Yuh go BUILD a quiz bot dat tests knowledge of
Caribbean geography, history, culture, and sports!

DIS IS A FILL-IN-THE-BLANKS PROJECT!
Look fi di TODO comments — dat's where YUH write code.
Di template and structure done already — yuh just need fi complete it.

When yuh done, yuh go have a working quiz bot dat:
  - Has questions about Caribbean geography, history, culture, and sports
  - Keeps score
  - Gives hints
  - Shows correct answers
  - Gives a final report wid a grade

Tips:
  - Read each TODO carefully
  - Look at di examples already in di code fi guidance
  - Run di file after each TODO fi test yuh changes
  - If yuh stuck, check di hints in di comments!

Let's build! 🔨
=============================================================================
"""

import random
import time


# =============================================================================
# PART 1: THE QUESTION BANK
# =============================================================================
# Each question is a dictionary with:
#   - "question": The question text
#   - "options": A list of 4 choices
#   - "answer": The correct answer (must match one of the options EXACTLY)
#   - "category": geography, history, culture, or sports
#   - "hint": A helpful hint
#   - "difficulty": "easy", "medium", or "hard"

question_bank = [
    # =========== GEOGRAPHY ===========
    {
        "question": "What is the capital of Jamaica?",
        "options": ["Montego Bay", "Kingston", "Ocho Rios", "Mandeville"],
        "answer": "Kingston",
        "category": "geography",
        "hint": "It's on the southeastern coast of the island.",
        "difficulty": "easy",
    },
    {
        "question": "Which is the LARGEST Caribbean island by area?",
        "options": ["Jamaica", "Hispaniola", "Cuba", "Puerto Rico"],
        "answer": "Cuba",
        "category": "geography",
        "hint": "This island is also a country with Havana as its capital.",
        "difficulty": "easy",
    },
    {
        "question": "What is the capital of Trinidad & Tobago?",
        "options": ["San Fernando", "Port of Spain", "Chaguanas", "Scarborough"],
        "answer": "Port of Spain",
        "category": "geography",
        "hint": "Think about di name — it's a 'Port' city.",
        "difficulty": "easy",
    },
    {
        "question": "Which Caribbean country is on the South American mainland?",
        "options": ["Barbados", "Jamaica", "Guyana", "Bahamas"],
        "answer": "Guyana",
        "category": "geography",
        "hint": "Its capital is Georgetown and it borders Venezuela, Brazil, and Suriname.",
        "difficulty": "medium",
    },
    {
        "question": "What ocean borders the Caribbean on the east?",
        "options": ["Pacific Ocean", "Indian Ocean", "Atlantic Ocean", "Arctic Ocean"],
        "answer": "Atlantic Ocean",
        "category": "geography",
        "hint": "Hurricanes come from this ocean into the Caribbean.",
        "difficulty": "easy",
    },

    # TODO 1: Add TWO more geography questions!
    # Follow di same format as above. Include questions about any
    # Caribbean country — Barbados, Bahamas, Haiti, Dominica, St. Lucia,
    # Belize, Grenada, Antigua, Suriname, Dominican Republic, etc.
    # HINT: Think about capitals, locations, borders, or natural features.
    #
    # Example structure:
    # {
    #     "question": "YOUR QUESTION HERE?",
    #     "options": ["Option A", "Option B", "Option C", "Option D"],
    #     "answer": "THE CORRECT OPTION",
    #     "category": "geography",
    #     "hint": "A helpful hint.",
    #     "difficulty": "easy",  # or "medium" or "hard"
    # },

    # =========== HISTORY ===========
    {
        "question": "In what year did Jamaica gain independence from Britain?",
        "options": ["1958", "1962", "1966", "1970"],
        "answer": "1962",
        "category": "history",
        "hint": "It was in the early 1960s. Same year as Trinidad & Tobago!",
        "difficulty": "medium",
    },
    {
        "question": "Who was the first leader of independent Trinidad & Tobago?",
        "options": ["Basdeo Panday", "Eric Williams", "ANR Robinson", "George Chambers"],
        "answer": "Eric Williams",
        "category": "history",
        "hint": "He was a famous historian and author of 'Capitalism and Slavery'.",
        "difficulty": "hard",
    },
    {
        "question": "Which Caribbean country was the FIRST to gain independence (1804)?",
        "options": ["Jamaica", "Cuba", "Haiti", "Barbados"],
        "answer": "Haiti",
        "category": "history",
        "hint": "It was the first free Black republic in the world!",
        "difficulty": "medium",
    },
    {
        "question": "What was the name of the political federation of Caribbean countries (1958-1962)?",
        "options": ["CARICOM", "West Indies Federation", "Caribbean Union", "Island Alliance"],
        "answer": "West Indies Federation",
        "category": "history",
        "hint": "It had 'West Indies' in the name, like di cricket team.",
        "difficulty": "hard",
    },

    # TODO 2: Add TWO more history questions!
    # Think about: Caribbean independence dates, famous leaders,
    # the sugar trade, emancipation, Marcus Garvey, Toussaint Louverture,
    # CARICOM formation, important events.

    # =========== CULTURE ===========
    {
        "question": "Which Caribbean country is the birthplace of reggae music?",
        "options": ["Trinidad & Tobago", "Barbados", "Jamaica", "Cuba"],
        "answer": "Jamaica",
        "category": "culture",
        "hint": "Think Bob Marley! 🎵",
        "difficulty": "easy",
    },
    {
        "question": "What musical instrument was invented in Trinidad & Tobago?",
        "options": ["Guitar", "Steelpan", "Bongo drum", "Harmonica"],
        "answer": "Steelpan",
        "category": "culture",
        "hint": "It's made from oil drums! 🥁",
        "difficulty": "easy",
    },
    {
        "question": "What is the national dish of Jamaica?",
        "options": ["Jerk Chicken", "Curry Goat", "Ackee and Saltfish", "Rice and Peas"],
        "answer": "Ackee and Saltfish",
        "category": "culture",
        "hint": "One ingredient is a fruit dat grows on a tree. 🍳",
        "difficulty": "medium",
    },
    {
        "question": "What famous street food from Trinidad features chickpea filling in fried dough?",
        "options": ["Roti", "Doubles", "Bake and Shark", "Pelau"],
        "answer": "Doubles",
        "category": "culture",
        "hint": "Yuh buy dem from a street vendor — 'bara and channa'! 🍽️",
        "difficulty": "medium",
    },
    {
        "question": "Which Caribbean festival is held before Lent and features costumes and music?",
        "options": ["Crop Over", "Carnival", "Junkanoo", "Reggae Sumfest"],
        "answer": "Carnival",
        "category": "culture",
        "hint": "Trinidad's is di most famous one! Soca and mas! 🎭",
        "difficulty": "easy",
    },

    # TODO 3: Add TWO more culture questions!
    # Think about: Caribbean food, music genres (soca, calypso, dancehall,
    # zouk, kompa), festivals (Crop Over, Junkanoo, Carnival), art,
    # literature, famous Caribbean people (Rihanna, Bob Marley, etc.)

    # =========== SPORTS ===========
    {
        "question": "Who holds the men's 100m world record at 9.58 seconds?",
        "options": ["Yohan Blake", "Asafa Powell", "Usain Bolt", "Tyson Gay"],
        "answer": "Usain Bolt",
        "category": "sports",
        "hint": "Him name sound like lightning! ⚡ From Trelawny, Jamaica.",
        "difficulty": "easy",
    },
    {
        "question": "Which country does Chris Gayle ('Universe Boss') represent in cricket?",
        "options": ["Barbados", "Trinidad & Tobago", "Jamaica", "Guyana"],
        "answer": "Jamaica",
        "category": "sports",
        "hint": "Same island as Usain Bolt! 🏏",
        "difficulty": "easy",
    },
    {
        "question": "What sport do the 'West Indies' play as a combined Caribbean team?",
        "options": ["Football", "Basketball", "Cricket", "Tennis"],
        "answer": "Cricket",
        "category": "sports",
        "hint": "Bat, ball, wickets — yuh know dis one! 🏏",
        "difficulty": "easy",
    },
    {
        "question": "Which Caribbean sprinter is known as 'The Pocket Rocket'?",
        "options": ["Elaine Thompson-Herah", "Shericka Jackson", "Shelly-Ann Fraser-Pryce", "Veronica Campbell-Brown"],
        "answer": "Shelly-Ann Fraser-Pryce",
        "category": "sports",
        "hint": "She's 5'0\" and one of di fastest women EVER! 🚀",
        "difficulty": "medium",
    },
    {
        "question": "What is the name of the Caribbean T20 cricket league?",
        "options": ["IPL", "Big Bash", "Caribbean Premier League (CPL)", "The Hundred"],
        "answer": "Caribbean Premier League (CPL)",
        "category": "sports",
        "hint": "It has teams like Trinbago Knight Riders and Jamaica Tallawahs.",
        "difficulty": "medium",
    },

    # TODO 4: Add TWO more sports questions!
    # Think about: Caribbean athletes, cricket records, Olympic medals,
    # football (soccer), netball, swimming, or any Caribbean sport.
]


# =============================================================================
# PART 2: THE QUIZ BOT CLASS
# =============================================================================

class CaribbeanQuizBot:
    """A quiz bot dat tests yuh knowledge of di Caribbean! 🌴"""

    def __init__(self, questions, num_questions=10):
        """Set up di quiz bot.

        Args:
            questions: List of question dictionaries
            num_questions: How many questions fi ask (default 10)
        """
        self.all_questions = questions
        self.num_questions = min(num_questions, len(questions))
        self.score = 0
        self.total_asked = 0
        self.results = []  # Track each question's result
        self.hints_used = 0

    def welcome(self):
        """Print di welcome message."""
        print("\n" + "🌴" * 25)
        print("   WELCOME TO DI CARIBBEAN QUIZ BOT!")
        print("🌴" * 25)
        print(f"""
   Designed by Adrian Dunkley (Adriandunkley.net)
   FREE fi Caribbean students!

   Yuh go get {self.num_questions} questions about:
     🗺️  Caribbean Geography
     📜  Caribbean History
     🎵  Caribbean Culture
     🏏  Caribbean Sports

   Type di NUMBER of yuh answer (1, 2, 3, or 4).
   Type 'H' fi a hint (but it cost yuh half a point!).
   Type 'Q' fi quit early.

   Let's see how much yuh know! 🧠
        """)
        input("   Press ENTER fi start...")

    def select_questions(self, category=None, difficulty=None):
        """Select random questions fi di quiz.

        TODO 5: Complete this function!
        It should:
        1. Filter questions by category (if given)
        2. Filter by difficulty (if given)
        3. Randomly select self.num_questions from di filtered list
        4. Return di selected questions

        HINT: Use list comprehension fi filtering, and random.sample() fi selection.
        """
        filtered = self.all_questions

        # TODO 5a: If category is not None, filter questions to only include
        # that category. (HINT: use list comprehension)
        # Example: filtered = [q for q in filtered if q["category"] == category]
        if category is not None:
            # YOUR CODE HERE — replace 'pass' with the filter
            pass

        # TODO 5b: If difficulty is not None, filter by difficulty too
        if difficulty is not None:
            # YOUR CODE HERE — replace 'pass' with the filter
            pass

        # Select random questions (don't pick more than available)
        count = min(self.num_questions, len(filtered))
        selected = random.sample(filtered, count)
        return selected

    def ask_question(self, question_dict, question_num):
        """Ask a single question and get di answer.

        Args:
            question_dict: The question dictionary
            question_num: The question number (1, 2, 3, ...)

        Returns:
            True if correct, False if wrong
        """
        print(f"\n{'='*50}")
        print(f"  Question {question_num}/{self.num_questions}")
        print(f"  Category: {question_dict['category'].upper()}")
        print(f"  Difficulty: {question_dict['difficulty']}")
        print(f"{'='*50}")
        print(f"\n  {question_dict['question']}\n")

        # Display options
        for i, option in enumerate(question_dict["options"], 1):
            print(f"    {i}. {option}")

        print(f"\n  (Type 1-4 fi answer, 'H' fi hint, 'Q' fi quit)")

        # Get answer from user
        while True:
            user_input = input("\n  Yuh answer: ").strip().upper()

            if user_input == "Q":
                return None  # Signal to quit

            if user_input == "H":
                print(f"\n  💡 HINT: {question_dict['hint']}")
                self.hints_used += 1
                continue

            # TODO 6: Validate di user's input and check if it's correct!
            # Steps:
            # 1. Check if user_input is a number between 1 and 4
            # 2. If valid, get di selected option from question_dict["options"]
            # 3. Compare it to question_dict["answer"]
            # 4. Return True if correct, False if wrong
            #
            # HINT: user_input is a string, so convert to int with int()
            #       Remember: options are indexed 0-3 but user types 1-4

            try:
                choice = int(user_input)
                if 1 <= choice <= 4:
                    selected_option = question_dict["options"][choice - 1]

                    if selected_option == question_dict["answer"]:
                        print(f"\n  ✅ CORRECT! Big up yuhself! 🎉")
                        return True
                    else:
                        print(f"\n  ❌ Wrong! Di answer was: {question_dict['answer']}")
                        return False
                else:
                    print("  Please type a number between 1 and 4.")
            except ValueError:
                print("  Invalid input. Type 1, 2, 3, 4, H, or Q.")

    def calculate_grade(self):
        """Calculate di student's grade based on score.

        TODO 7: Complete this function!
        Return a grade string based on percentage:
          90-100% → "A+ — OUTSTANDING! Yuh a Caribbean genius! 🌟"
          80-89%  → "A — EXCELLENT! Yuh know yuh ting! 🎉"
          70-79%  → "B — GOOD JOB! Keep learning! 💪"
          60-69%  → "C — Not bad! Study a likkle more! 📚"
          50-59%  → "D — Yuh need fi study more! Hit di books! 📖"
          Below 50% → "F — Nuh worry! Try again and yuh go improve! 🔄"

        HINT: Calculate percentage = (self.score / self.total_asked) * 100
        """
        if self.total_asked == 0:
            return "No questions answered."

        percentage = (self.score / self.total_asked) * 100

        # TODO 7: Replace dis wid yuh if/elif/else chain
        # based on di percentage ranges above
        if percentage >= 90:
            return "A+ — OUTSTANDING! Yuh a Caribbean genius! 🌟"
        elif percentage >= 80:
            return "A — EXCELLENT! Yuh know yuh ting! 🎉"
        elif percentage >= 70:
            return "B — GOOD JOB! Keep learning! 💪"
        elif percentage >= 60:
            return "C — Not bad! Study a likkle more! 📚"
        elif percentage >= 50:
            return "D — Yuh need fi study more! Hit di books! 📖"
        else:
            return "F — Nuh worry! Try again and yuh go improve! 🔄"

    def show_results(self):
        """Show di final results.

        TODO 8: Complete dis function to show:
        1. Total score (self.score / self.total_asked)
        2. Percentage
        3. Grade (use self.calculate_grade())
        4. Number of hints used
        5. Results breakdown by category

        HINT: Look at self.results — each entry is a dict with:
              {"question": ..., "correct": True/False, "category": ...}
        """
        print("\n" + "🏆" * 25)
        print("   QUIZ RESULTS!")
        print("🏆" * 25)

        if self.total_asked == 0:
            print("\n  Yuh didn't answer any questions! Try again!")
            return

        percentage = (self.score / self.total_asked) * 100
        grade = self.calculate_grade()

        print(f"""
   Score: {self.score}/{self.total_asked}
   Percentage: {percentage:.1f}%
   Grade: {grade}
   Hints used: {self.hints_used}
        """)

        # TODO 8: Show results by category
        # Count correct answers per category
        # HINT: Loop through self.results and count by category
        categories = {}
        for result in self.results:
            cat = result["category"]
            if cat not in categories:
                categories[cat] = {"correct": 0, "total": 0}
            categories[cat]["total"] += 1
            if result["correct"]:
                categories[cat]["correct"] += 1

        print("   Results by Category:")
        for cat, stats in categories.items():
            emoji = {"geography": "🗺️", "history": "📜",
                     "culture": "🎵", "sports": "🏏"}.get(cat, "📝")
            pct = (stats["correct"] / stats["total"] * 100) if stats["total"] > 0 else 0
            print(f"     {emoji} {cat.capitalize()}: {stats['correct']}/{stats['total']} ({pct:.0f}%)")

        print(f"\n   {'🌴' * 25}")

    def run(self, category=None, difficulty=None):
        """Run di entire quiz!"""
        self.welcome()

        # Select questions
        questions = self.select_questions(category, difficulty)

        if not questions:
            print("\n  No questions found fi dat category/difficulty! Try again.")
            return

        self.num_questions = len(questions)  # Update in case fewer available

        # Ask each question
        for i, q in enumerate(questions, 1):
            result = self.ask_question(q, i)

            if result is None:  # User quit
                print("\n  Yuh quit early! Let's see how yuh did so far...")
                break

            self.total_asked += 1
            if result:
                self.score += 1

            self.results.append({
                "question": q["question"],
                "correct": result,
                "category": q["category"],
            })

            # Small pause between questions
            time.sleep(0.5)

        # Show results
        self.show_results()

        # Ask to play again
        print("\n  Want fi play again? Run di file again! 🔄")
        print("  Or try a specific category: geography, history, culture, sports")


# =============================================================================
# PART 3: RUN DI QUIZ!
# =============================================================================

if __name__ == "__main__":
    print("""
    ╔═══════════════════════════════════════════════════╗
    ║     Caribbean Quiz Bot — Capstone Project         ║
    ║     Caribbean AI Academy (Forms 1-3)              ║
    ║     By Adrian Dunkley (Adriandunkley.net)          ║
    ╚═══════════════════════════════════════════════════╝

    Choose yuh quiz mode:
      1. All categories (mixed)
      2. Geography only
      3. History only
      4. Culture only
      5. Sports only
      6. Easy questions only
      7. Hard questions only
    """)

    choice = input("    Enter yuh choice (1-7): ").strip()

    category = None
    difficulty = None

    if choice == "2":
        category = "geography"
    elif choice == "3":
        category = "history"
    elif choice == "4":
        category = "culture"
    elif choice == "5":
        category = "sports"
    elif choice == "6":
        difficulty = "easy"
    elif choice == "7":
        difficulty = "hard"

    # How many questions?
    try:
        num = int(input("    How many questions? (5-20): ").strip())
        num = max(5, min(20, num))
    except ValueError:
        num = 10

    # Create and run di bot!
    bot = CaribbeanQuizBot(question_bank, num_questions=num)
    bot.run(category=category, difficulty=difficulty)


# =============================================================================
# TODO CHECKLIST — Make Sure Yuh Complete All of Dese!
# =============================================================================
# [ ] TODO 1: Add 2 more geography questions to di question bank
# [ ] TODO 2: Add 2 more history questions to di question bank
# [ ] TODO 3: Add 2 more culture questions to di question bank
# [ ] TODO 4: Add 2 more sports questions to di question bank
# [ ] TODO 5: Complete di select_questions() method (filtering)
# [ ] TODO 6: Validate user input in ask_question() (already done as example)
# [ ] TODO 7: Complete di calculate_grade() method (already done as example)
# [ ] TODO 8: Complete di show_results() method (already done as example)
#
# BONUS CHALLENGES:
# [ ] Add a timer fi each question (use time.time())
# [ ] Add a "lifeline" feature (50/50 — remove 2 wrong answers)
# [ ] Save high scores to a file
# [ ] Add questions about ALL 24 Caribbean territories
# [ ] Add an "explain" feature dat gives more info after each answer
# [ ] Make di questions harder as yuh get more right (adaptive difficulty)
# =============================================================================
