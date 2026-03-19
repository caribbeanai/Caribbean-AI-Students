# The Caribbean AI Puzzle Portal

**Designed by Adrian Dunkley** | [Adriandunkley.net](https://Adriandunkley.net) | **FREE Curriculum**

---

```
    ___________________________________________________
   |                                                   |
   |    WELCOME TO DI PUZZLE PORTAL, CHAMPION!         |
   |                                                   |
   |    "Every piece ah di puzzle bring yuh closer     |
   |     to greatness. Nuh give up now!"               |
   |___________________________________________________|
        \   ^__^
         \  (oo)\_______
            (__)\       )\/\
                ||----w |
                ||     ||
```

---

## What Is This?

Throughout this entire Caribbean AI curriculum, **12 puzzle pieces** are hidden
across the lessons, notebooks, and exercises. Each piece is a clue. When you
put them all together, they form a **secret passphrase** that proves you
completed the full journey from Primary Prep all the way to Graduate Level.

**Your mission:** Collect all 12 clues, decode each one, combine them into a
single passphrase, and enter it below.

**Your reward:** The contact to reach **Adrian Dunkley** himself and claim your
Certificate of Completion!

> "Di journey of a thousand miles start with one step. But di journey through
> Caribbean AI start with one puzzle piece!" -- Puzzle Portal Proverb

---

## The 12 Puzzle Pieces

Find these clues scattered throughout the curriculum. Each one gives you part
of the final answer.

### Piece 1: The Color
> *Think about traffic lights. Think about the Jamaican flag. What color do
> they share? Di color of the go signal, the color of the hills and the
> vegetation on our beautiful islands.*

### Piece 2: The Ranger
> *"It's Morphin Time!" Remember that show? Power _____ ! What were they
> called? Those heroes who protected the earth. Fill in the blank.*

### Piece 3: The Number
> *The Universe Boss, Chris Gayle -- the greatest T20 batsman the Caribbean
> ever produced. Him wear number 45 on him jersey. Now take that number and
> subtract 40. What yuh get?*

### Piece 4: The Two Digits
> *Piece 3 gave you a single digit. But the passphrase needs TWO digits. If
> your number is less than 10, how do computers usually pad a single digit to
> make it two? Think leading characters...*

### Piece 5: The Anime
> *There's an anime about people trapped in a virtual reality online game. The
> name has the word "art" in it -- but not "art" by itself. It's an art made
> with a weapon. What is the name? (Just the two words, no spaces, no "Online")*

### Piece 6: The Weapon (Confirmation)
> *A knight's weapon. Long, sharp, used for dueling. This word appears in the
> anime from Piece 5. This confirms you have the right anime.*

### Piece 7: Encouragement
> *Yuh doing great! Keep going, champion. This piece is just encouragement --
> no new letters to add. But remember: every piece matters for understanding
> the FORMAT.*

### Piece 8: The Format
> *Now put it together! The format is:*
> ```
> [color][animal/hero][number][anime+type]
> ```
> *No spaces between them. Think about what "type" means from Piece 5...*

### Piece 9: The Animal (Confirmation)
> *What animal is colored like the thing from Piece 1? Not a blue one. Not a
> red one. The color from Piece 1 + the answer from Piece 2 = what you call
> one of these heroes.*

### Piece 10: Which Ranger?
> *Which specific one? The one colored like limes. Like grass. Like the "go"
> signal on a traffic light. Like the Jamaican hills. That specific one.*

### Piece 11: Formatting Rules
> *IMPORTANT: No spaces. No capital letters. Everything lowercase. One
> continuous string. Run it all together like one long word.*

### Piece 12: Final Assembly
> *Put it ALL together now:*
> ```
> [color from Piece 1][hero type from Piece 2][two-digit number from Piece 3+4][anime name from Piece 5]
> ```
> *All lowercase. No spaces. No special characters. Just letters and digits.*

---

## Enter Your Answer

### Option 1: Python (Terminal)

Run the checker script:

```bash
python puzzle_portal/check_answer.py
```

### Option 2: Web Browser

Open `puzzle_portal/puzzle_web.html` in your browser for an interactive
experience with Caribbean vibes!

### Option 3: Right Here (Manual Check)

If you think you have the answer, generate its SHA-256 hash and compare:

```python
import hashlib
your_answer = "youranswer"  # replace with your guess
hash_result = hashlib.sha256(your_answer.encode()).hexdigest()
print(hash_result)
# If it matches: 1a7351b437009845a9e4ea77e9bff tried them all...
# Run check_answer.py for the real check!
```

---

## What Happens When You Solve It?

When you enter the correct passphrase:

1. You'll receive a **congratulations message**
2. You'll be given the email address to contact the curriculum creator
3. You can email **Adrian Dunkley** to claim your completion!
4. You'll have bragging rights across the entire Caribbean

---

## Hints If You're Stuck

- Re-read each module's introduction and conclusion -- clues are often in the
  motivational sections
- The answer is ONE continuous string with no spaces
- It contains: a color, a type of hero, a two-digit number, and an anime name
- Everything is lowercase
- The total length is 20 characters

---

## Credits

This puzzle and the entire Caribbean AI curriculum were designed by
**Adrian Dunkley** ([Adriandunkley.net](https://Adriandunkley.net)).

This curriculum is **completely FREE**. Share it. Spread it. Let every
Caribbean student have access to AI education.

> "Di future of di Caribbean is bright, and it start with YOU." -- Adrian Dunkley

---

*Now go find those puzzle pieces, champion! Yuh got this!*
