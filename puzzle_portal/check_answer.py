#!/usr/bin/env python3
"""
Caribbean AI Puzzle Portal -- Answer Checker
Designed by Adrian Dunkley (Adriandunkley.net) | FREE Curriculum

Run this script and enter your assembled passphrase to check if you've
solved the Caribbean AI Puzzle!
"""

import hashlib
import sys
import time

# SHA-256 hash of the correct answer
CORRECT_HASH = hashlib.sha256("greenranger05swordart".encode()).hexdigest()

REWARD_EMAIL = "founder@starapple.ai"

BANNER = r"""
╔══════════════════════════════════════════════════════════════════╗
║                                                                  ║
║     THE CARIBBEAN AI PUZZLE PORTAL                               ║
║     ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~                               ║
║                                                                  ║
║     Designed by Adrian Dunkley | Adriandunkley.net | FREE        ║
║                                                                  ║
║     "Every puzzle piece yuh find bring yuh closer                ║
║      to di top ah di mountain, champion!"                        ║
║                                                                  ║
╚══════════════════════════════════════════════════════════════════╝
"""

CELEBRATION = r"""
╔══════════════════════════════════════════════════════════════════╗
║                                                                  ║
║   ***  CONGRATULATIONS, CHAMPION!!!  ***                         ║
║                                                                  ║
║        .---.         .---.         .---.                          ║
║       /     \       /     \       /     \                         ║
║      | () () |     | () () |     | () () |                       ║
║       \  ^  /       \  ^  /       \  ^  /                        ║
║        '---'         '---'         '---'                          ║
║                                                                  ║
║   ~~~~~~~~~~~~~ CARIBBEAN VIBES ~~~~~~~~~~~~~                    ║
║                                                                  ║
║       __         __         __                                   ║
║      /  \  ~~~  /  \  ~~~  /  \                                  ║
║     / /\ \     / /\ \     / /\ \                                 ║
║    /_/  \_\   /_/  \_\   /_/  \_\                                ║
║                                                                  ║
║   Palm trees swaying! Steel drums playing!                       ║
║   You did it! You completed the ENTIRE                           ║
║   Caribbean AI Curriculum!                                       ║
║                                                                  ║
╚══════════════════════════════════════════════════════════════════╝
"""

WRONG_MESSAGES = [
    "Hmm, dat nah right. But doh give up! Try again, champion!",
    "Close but no cigar... or should we say, no sugarcane! Try again!",
    "Dat ain't it, fam. Go back and check yuh puzzle pieces!",
    "Nah, star. But remember -- every wrong answer teach yuh something!",
    "Not quite! But di journey is di reward. Keep trying!",
]


def slow_print(text, delay=0.02):
    """Print text character by character for dramatic effect."""
    for char in text:
        sys.stdout.write(char)
        sys.stdout.flush()
        time.sleep(delay)
    print()


def get_hint(answer):
    """Give a hint based on how close the answer is."""
    answer = answer.lower().strip()
    hints = []

    if not answer:
        return "Yuh didn't type anything! Go collect dem puzzle pieces first!"

    # Check for common mistakes
    if " " in answer:
        hints.append("HINT: Remove ALL spaces! Di answer is one continuous string.")

    if answer != answer.lower():
        hints.append("HINT: Everything should be LOWERCASE. No capital letters!")

    if "green" not in answer:
        hints.append("HINT (Piece 1): Think about the color of traffic lights "
                      "when they say GO. Think about the Jamaican hills...")
    else:
        hints.append("Good: Yuh got di color right!")

    if "ranger" not in answer:
        hints.append("HINT (Piece 2): 'It's Morphin Time!' -- Power _____ !")
    else:
        hints.append("Good: Yuh got di hero type right!")

    if "05" not in answer:
        if "5" in answer:
            hints.append("HINT (Piece 3-4): Yuh got the right digit, but "
                          "remember -- two digits! How do computers pad numbers?")
        else:
            hints.append("HINT (Piece 3): Chris Gayle jersey = 45. "
                          "Subtract 40. Then make it two digits!")
    else:
        hints.append("Good: Yuh got di number right!")

    if "sword" not in answer and "art" not in answer:
        hints.append("HINT (Piece 5): An anime about a virtual reality game. "
                      "The name contains 'art' and a weapon...")
    elif "swordart" not in answer:
        hints.append("HINT (Piece 5): Yuh close with di anime! Remember: "
                      "just the two words, no spaces, no 'Online'.")
    else:
        hints.append("Good: Yuh got di anime right!")

    if len(answer) != 20:
        hints.append(f"HINT: Your answer is {len(answer)} characters. "
                      f"Di correct answer is 20 characters.")

    return "\n".join(hints)


def check_answer(answer):
    """Check if the answer is correct using hash comparison."""
    answer_clean = answer.lower().strip()
    answer_hash = hashlib.sha256(answer_clean.encode()).hexdigest()
    return answer_hash == CORRECT_HASH


def main():
    """Main puzzle checker loop."""
    print(BANNER)
    slow_print("Welcome to di Caribbean AI Puzzle Portal!", 0.03)
    print()
    slow_print("Yuh collected all 12 puzzle pieces? Time fi put dem together!", 0.03)
    print()
    print("=" * 60)
    print()

    attempts = 0
    max_attempts = 10

    while attempts < max_attempts:
        answer = input("Enter yuh passphrase (or 'quit' to exit): ").strip()

        if answer.lower() == "quit":
            print("\nNo problem! Come back when yuh ready, champion!")
            print("Remember: di puzzle pieces are scattered throughout")
            print("the entire Caribbean AI curriculum.")
            sys.exit(0)

        attempts += 1

        if check_answer(answer):
            print(CELEBRATION)
            time.sleep(0.5)
            slow_print("=" * 60, 0.01)
            print()
            slow_print("YUH DID IT!!! You solved the Caribbean AI Puzzle!", 0.04)
            print()
            slow_print("You have completed the ENTIRE Caribbean AI Curriculum", 0.04)
            slow_print("from Primary Prep all the way to Graduate Level!", 0.04)
            print()
            slow_print("=" * 60, 0.01)
            print()
            slow_print("YOUR REWARD:", 0.05)
            print()
            slow_print(f"   Email: {REWARD_EMAIL}", 0.05)
            print()
            slow_print("Send an email to Adrian Dunkley at the address above", 0.04)
            slow_print("to claim your Certificate of Completion!", 0.04)
            print()
            slow_print("Subject line: 'Caribbean AI Puzzle Complete!'", 0.04)
            slow_print("Include your name and which island you represent!", 0.04)
            print()
            slow_print("=" * 60, 0.01)
            print()
            slow_print("This curriculum was designed by Adrian Dunkley", 0.03)
            slow_print("Website: Adriandunkley.net", 0.03)
            slow_print("This curriculum is FREE. Share it with everyone!", 0.03)
            print()
            slow_print("BIG UP YUHSELF, CHAMPION! DI CARIBBEAN IS PROUD!", 0.04)
            print()
            sys.exit(0)
        else:
            import random
            print()
            print(random.choice(WRONG_MESSAGES))
            print()
            print(get_hint(answer))
            print()
            remaining = max_attempts - attempts
            if remaining > 0:
                print(f"Attempts remaining: {remaining}")
            print()

    print("Yuh used all your attempts! But doh worry --")
    print("just run di script again. We believe in second chances")
    print("in di Caribbean! Go review di curriculum and come back stronger!")


if __name__ == "__main__":
    main()
