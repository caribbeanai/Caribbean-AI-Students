#!/usr/bin/env python3
"""
Caribbean AI Puzzle Portal -- Answer Checker
Designed by Adrian Dunkley (Adriandunkley.net) | FREE Curriculum

Run this script and enter your assembled passphrase to check if you've
solved the Caribbean AI Puzzle!

Usage:
    python check_answer.py
"""

import hashlib
import sys
import time
import random

# =============================================================================
# The correct answer hash (SHA-256)
# We NEVER store the plaintext answer -- only its hash.
# =============================================================================
CORRECT_HASH = "5f2c457e3811e3659e14ac0e80a8a7927de3b794cab77b7952db6a72f1a30050"

REWARD_EMAIL = "founder@starapple.ai"

BANNER = r"""
 ___________________________________________________________________________
|                                                                           |
|         *** THE CARIBBEAN AI PUZZLE PORTAL ***                            |
|         ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~                            |
|                                                                           |
|         Designed by Adrian Dunkley | Adriandunkley.net | FREE             |
|                                                                           |
|              .  *  .     .  *  .     .  *  .                              |
|           *    ___    *    ___    *    ___    *                            |
|          .   /   \  .   /   \  .   /   \   .                             |
|             / === \     / === \     / === \                                |
|            |  _|_  |   |  _|_  |   |  _|_  |                             |
|            |_|   |_|   |_|   |_|   |_|   |_|                             |
|            Steel Drums   Steel Drums   Steel Drums                        |
|                                                                           |
|     "Every puzzle piece yuh find bring yuh closer                         |
|      to di top ah di mountain, champion!"                                 |
|                                                                           |
|___________________________________________________________________________|
"""

CELEBRATION = r"""

    *  .  *  .  *  .  *  .  *  .  *  .  *  .  *  .  *  .  *  .  *

     $$$$$$   $$$$$$  $$    $$  $$$$$$  $$$$$   $$$$$$  $$$$$$  $$  $$
    $$       $$    $$ $$$   $$ $$      $$   $$ $$    $$ $$  $$  $$ $$
    $$       $$    $$ $$ $$ $$ $$  $$$ $$$$$   $$$$$$$$ $$  $$  $$$$
    $$       $$    $$ $$   $$$ $$    $$ $$  $$ $$    $$ $$  $$  $$ $$
     $$$$$$   $$$$$$  $$    $$  $$$$$$  $$  $$ $$    $$ $$$$$$  $$  $$

    *  .  *  .  *  .  *  .  *  .  *  .  *  .  *  .  *  .  *  .  *

         __/\__         __/\__         __/\__         __/\__
        /      \       /      \       /      \       /      \
       / PALM   \     / PALM   \     / PALM   \     / PALM   \
      /  TREE    \   /  TREE    \   /  TREE    \   /  TREE    \
     /    |||     \ /    |||     \ /    |||     \ /    |||     \
    ~~~~~~|||~~~~~~ ~~~~~~|||~~~~~~ ~~~~~~|||~~~~~~ ~~~~~~|||~~~~~~
    ~~~~~~~~~~~~~~~ ~~~~~~~~~~~~~~~ ~~~~~~~~~~~~~~~ ~~~~~~~~~~~~~~~
          CARIBBEAN WAVES OF CELEBRATION!

    *  .  *  .  *  .  *  .  *  .  *  .  *  .  *  .  *  .  *  .  *

        YUH DID IT, CHAMPION!!!

        You solved the Caribbean AI Puzzle!
        You completed the ENTIRE curriculum!

        From Primary Prep to Graduate Level --
        every lesson, every notebook, every exercise.

        The Caribbean is PROUD of you!

    *  .  *  .  *  .  *  .  *  .  *  .  *  .  *  .  *  .  *  .  *
"""

WRONG_MESSAGES = [
    "Hmm, dat nah right. But doh give up! Try again, champion!",
    "Close but no cigar... or should we say, no sugarcane! Try again!",
    "Dat ain't it, fam. Go back and check yuh puzzle pieces!",
    "Nah, star. But remember -- every wrong answer teach yuh something!",
    "Not quite! But di journey is di reward. Keep pushing!",
    "Wah gwaan?! Dat nuh match. Check yuh clues again!",
    "Nope! But every great cricketer get bowl out sometimes. Try again!",
    "Ehhh... not dis time. But yuh getting warmer? Maybe? Check di hints!",
    "Di spirits ah di Caribbean ancestors say... TRY AGAIN!",
    "Wrong answer, but right spirit! Keep dat energy, champion!",
]

CARIBBEAN_PROVERBS = [
    '"Cockroach nuh business inna fowl fight." -- But THIS fight is yours!',
    '"Every day bucket go ah well, one day di bottom mus drop out." -- Keep trying!',
    '"Wha sweet nanny goat ah go run him belly." -- Di answer sweet, keep searching!',
    '"One one cocoa full basket." -- One clue at a time, champion!',
    '"If yuh want good, yuh nose haffi run." -- Work hard fi di answer!',
]


def slow_print(text, delay=0.02):
    """Print text character by character for dramatic effect."""
    for char in text:
        sys.stdout.write(char)
        sys.stdout.flush()
        time.sleep(delay)
    print()


def compute_hash(text):
    """Compute SHA-256 hash of a string."""
    return hashlib.sha256(text.encode()).hexdigest()


def get_hint(answer):
    """Give contextual hints based on how close the answer is."""
    answer = answer.lower().strip()
    hints = []

    if not answer:
        return "Yuh didn't type anything! Go collect dem puzzle pieces first!"

    # Check formatting issues first
    if " " in answer:
        hints.append(
            "FORMAT: Remove ALL spaces! Di answer is one continuous string "
            "with no gaps. Read Piece 11 again!"
        )

    if answer != answer.lower():
        hints.append(
            "FORMAT: Everything should be LOWERCASE. No capital letters anywhere! "
            "Check Piece 11!"
        )

    # Check individual components
    has_green = "green" in answer
    has_ranger = "ranger" in answer
    has_05 = "05" in answer
    has_swordart = "swordart" in answer

    if not has_green:
        hints.append(
            "PIECE 1: Think about traffic lights when they say GO! "
            "Think about the color of di Jamaican hills, di sugarcane fields, "
            "the vegetation all over di Caribbean..."
        )
    else:
        hints.append("PIECE 1: Correct! Yuh got di color right!")

    if not has_ranger:
        if "rangers" in answer:
            hints.append(
                "PIECE 2: Almost! But use the SINGULAR form. "
                "One _____, not multiple."
            )
        elif "power" in answer:
            hints.append(
                "PIECE 2: Yuh don't need the word 'Power' -- just what comes AFTER it! "
                "Power _____ -- what's the blank?"
            )
        else:
            hints.append(
                "PIECE 2: 'It's Morphin Time!' Power _____ ! "
                "What were those heroes called? Fill in the ONE word!"
            )
    else:
        hints.append("PIECE 2: Correct! Yuh got di hero type right!")

    if not has_05:
        if "5" in answer and "05" not in answer:
            hints.append(
                "PIECES 3-4: Yuh got the right DIGIT, but remember -- "
                "the passphrase needs TWO digits! How do computers pad "
                "a single digit? Think zero-padding... think 01, 02, 03..."
            )
        elif "45" in answer:
            hints.append(
                "PIECES 3-4: 45 is Gayle's jersey number, but yuh need to "
                "SUBTRACT 40 from it first! Then make it two digits."
            )
        else:
            hints.append(
                "PIECE 3: Chris Gayle, the Universe Boss! Jersey number 45. "
                "Subtract 40 from dat. Then Piece 4 says make it two digits!"
            )
    else:
        hints.append("PIECES 3-4: Correct! Yuh got di number right!")

    if not has_swordart:
        if "sword" in answer and "art" in answer and "swordart" not in answer:
            hints.append(
                "PIECE 5: Yuh have the right words but check the ORDER! "
                "The weapon comes first, then 'art'. No space between them."
            )
        elif "sao" in answer or "online" in answer:
            hints.append(
                "PIECE 5: Don't use abbreviations or 'Online'! "
                "Just the first two words of the anime name, no spaces: "
                "[weapon][art]"
            )
        else:
            hints.append(
                "PIECE 5: Famous anime -- people trapped in a VR game. "
                "The name has a weapon + 'art'. Think: what do knights carry? "
                "That weapon + 'art' = the anime name (drop the 'Online')."
            )
    else:
        hints.append("PIECE 5: Correct! Yuh got di anime right!")

    # Length check
    correct_count = sum([has_green, has_ranger, has_05, has_swordart])
    clean = answer.replace(" ", "").lower()
    if len(clean) != 20:
        hints.append(
            f"LENGTH: Your answer is {len(clean)} characters (after cleanup). "
            f"Di correct answer is exactly 20 characters."
        )

    if correct_count == 4 and len(clean) != 20:
        hints.append(
            "Yuh have all the RIGHT pieces but something extra crept in! "
            "Make sure there's NOTHING else -- just the four parts joined together."
        )

    # Caribbean proverb for encouragement
    hints.append(f"\n{random.choice(CARIBBEAN_PROVERBS)}")

    return "\n".join(hints)


def check_answer(answer):
    """Check if the answer is correct using secure hash comparison."""
    answer_clean = answer.lower().strip()
    answer_hash = compute_hash(answer_clean)
    return answer_hash == CORRECT_HASH


def reveal_reward():
    """Reveal the reward after correct answer."""
    print(CELEBRATION)
    time.sleep(1)

    slow_print("=" * 65, 0.008)
    print()
    slow_print("  YOUR REWARD", 0.05)
    print()
    slow_print("=" * 65, 0.008)
    print()
    slow_print("  You have earned the right to contact the curriculum creator!", 0.03)
    print()
    slow_print(f"  Email:   {REWARD_EMAIL}", 0.05)
    slow_print("  Website: https://Adriandunkley.net", 0.03)
    print()
    slow_print("  Send an email to Adrian Dunkley at the address above", 0.03)
    slow_print("  to claim your Certificate of Completion!", 0.03)
    print()
    slow_print("  Subject: 'Caribbean AI Puzzle Complete!'", 0.03)
    slow_print("  Include: Your name, your island, your AI journey story!", 0.03)
    print()
    slow_print("=" * 65, 0.008)
    print()
    slow_print("  This curriculum was designed by Adrian Dunkley", 0.03)
    slow_print("  It is FREE. Share it with every student in di Caribbean!", 0.03)
    print()
    slow_print("  BIG UP YUHSELF, CHAMPION! DI CARIBBEAN IS PROUD OF YOU!", 0.04)
    print()
    slow_print("=" * 65, 0.008)
    print()


def main():
    """Main puzzle checker loop."""
    print(BANNER)
    slow_print("  Welcome to di Caribbean AI Puzzle Portal!", 0.03)
    print()
    slow_print("  Yuh collected all 12 puzzle pieces?", 0.03)
    slow_print("  Time fi put dem together and claim yuh reward!", 0.03)
    print()
    print("=" * 65)
    print()
    print("  Type your passphrase below.")
    print("  (All lowercase, no spaces, letters and digits only)")
    print()

    attempts = 0
    max_attempts = 15

    while attempts < max_attempts:
        try:
            answer = input("  >>> Enter yuh passphrase (or 'quit' fi leave): ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\n\n  Lata, champion! Come back when yuh ready!")
            sys.exit(0)

        if answer.lower() == "quit":
            print()
            print("  No problem! Come back when yuh ready, champion!")
            print("  Remember: di puzzle pieces are scattered throughout")
            print("  the ENTIRE Caribbean AI curriculum -- every level!")
            print()
            print(f"  {random.choice(CARIBBEAN_PROVERBS)}")
            print()
            sys.exit(0)

        attempts += 1

        if check_answer(answer):
            reveal_reward()
            sys.exit(0)
        else:
            print()
            print(f"  {random.choice(WRONG_MESSAGES)}")
            print()
            print("-" * 65)
            print(get_hint(answer))
            print("-" * 65)
            remaining = max_attempts - attempts
            if remaining > 0:
                print(f"\n  Attempts remaining: {remaining}")
            print()

    print()
    print("  Yuh used all your attempts! But doh worry --")
    print("  just run di script again. We believe in second chances")
    print("  and third chances and fourth chances in di Caribbean!")
    print("  Go review di curriculum and come back stronger!")
    print()
    print(f"  {random.choice(CARIBBEAN_PROVERBS)}")
    print()


if __name__ == "__main__":
    main()
