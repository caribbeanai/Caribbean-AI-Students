# Quiz Answers — Forms 1-3 (Lower Secondary)

### *"Nuh peek til yuh TRY first! Dat's how yuh learn fi real."*

---

> **Caribbean AI Academy** — Forms 1-3 (Ages 11-14)
> Designed by **[Adrian Dunkley](https://Adriandunkley.net)** | 100% FREE for Caribbean students

---

## Lesson 1: Python Basics

**Q1:** What data type is `my_island = "Jamaica"`?
> **c) string** — Text wrapped in quotes is always a string.

**Q2:** What does `caribbean_fruits[2]` return if the list is `["ackee", "mango", "soursop", "guinep"]`?
> **c) soursop** — Remember, Python lists start counting at 0! Index 0 = "ackee", index 1 = "mango", index 2 = "soursop".

**Q3:** What does `len(caribbean_countries)` tell yuh?
> **c) The number of items** — `len()` returns how many items are in a list, string, or other collection.

**Q4:** What do we call "Jamaica" in `caribbean_capitals["Jamaica"] = "Kingston"`?
> **b) key** — In a dictionary, the part inside the brackets is the key. "Kingston" is the value.

**Q5:** What will this print: `for i in range(3): print(i)`?
> **b) 0 1 2** — `range(3)` produces 0, 1, 2. It starts at 0 and stops BEFORE 3.

**Q6:** What does `append` do to a list?
> **c) Add an item to the end** — `my_list.append("new item")` puts it at the very end.

**Q7:** Write a function called `island_greeting`:
> ```python
> def island_greeting(country):
>     return f"Welcome to {country}! Enjoy di vibes!"
>
> # Test it:
> print(island_greeting("Barbados"))
> # Output: "Welcome to Barbados! Enjoy di vibes!"
> ```

**Q8:** What is the output when x = 10?
> **a) Big** — Since 10 > 5 is True, it prints "Big".

**Q9:** Dictionary with 3 countries and populations:
> ```python
> populations = {
>     "Jamaica": 2_828_000,
>     "Barbados": 288_000,
>     "Trinidad & Tobago": 1_405_000,
> }
> for country, pop in populations.items():
>     print(f"{country}: {pop:,}")
> ```

**Q10:** Two Caribbean sectors that use Python lists:
> - **Tourism:** Lists of hotel bookings, visitor arrival dates, cruise ship schedules, tourist attractions
> - **Fishing:** Lists of fish species caught, daily catch weights, boat registrations, fishing zones
> - (Also acceptable: Agriculture — crop yields by month; Healthcare — patient records; Sports — player stats)

---

## Lesson 2: Data with Pandas

**Q1:** What is a DataFrame?
> **b) A table of data wid rows and columns** — Think of it like a spreadsheet inside Python.

**Q2:** What does `df.head(3)` do?
> **c) Shows the first 3 rows** — `.head(n)` shows the first n rows. Default is 5.

**Q3:** How do yuh filter for Jamaica data?
> **b) `df[df["Country"] == "Jamaica"]`** — This creates a boolean mask and filters rows where Country equals "Jamaica".

**Q4:** What does `df.shape` return?
> **c) A tuple with (rows, columns)** — For example, `(48, 5)` means 48 rows and 5 columns.

**Q5:** What does `df.groupby("Country")["Visitors"].sum()` do?
> **b) Groups data by country and adds up visitors** — It groups all rows by country, then sums the visitor column for each group.

**Q6:** Which country had the MOST tourists in 2023?
> **Dominican Republic** — with 7,200 thousand (7.2 million) visitors, followed by The Bahamas with 6,200 thousand.

**Q7:** Two other sectors where pandas is useful:
> - **Agriculture:** Analyzing crop yields by region, season, and crop type. Tracking rainfall, soil quality, and harvest data for sugarcane, bananas, cocoa, etc.
> - **Fishing:** Tracking daily catch data by species, location, and boat. Analyzing seasonal fishing patterns to support sustainable fishing in Caribbean waters.
> - (Also acceptable: Healthcare — patient data, disease tracking; Sports — player performance stats; Energy — electricity usage patterns)

**Q8:** Finding countries where tourism decreased:
> **c) filter, calculate change, filter negative** — You would filter 2020 and 2023 data, calculate the percentage change, then filter for negative values.

---

## Lesson 3: Intro to ML Concepts

**Q1:** What is machine learning?
> **b) A computer dat learns patterns from data and makes predictions** — Instead of writing every rule by hand, ML lets computers learn from examples.

**Q2:** Which type is like teaching a child to identify birds?
> **c) Supervised Learning** — You provide labeled examples (bird name + image), and the model learns the patterns.

**Q3:** Sorting spices without knowing names is which type?
> **b) Unsupervised Learning** — The algorithm finds natural groups/clusters without any labels.

**Q4:** What tells the computer if it did good or bad in reinforcement learning?
> **c) The reward signal** — Like getting points for staying on the surfboard or losing points for falling off.

**Q5:** What is overfitting?
> **b) When the model memorizes the training data instead of learning real patterns** — An overfit model does great on training data but poorly on new data, like a student who memorizes answers but can't solve new problems.

**Q6:** Three Caribbean sectors for supervised learning:
> 1. **Tourism:** Predict next month's hotel occupancy based on historical data, season, events, and flight bookings. Labels = actual occupancy numbers.
> 2. **Agriculture:** Predict crop disease from leaf images. Labels = "healthy" or "diseased" for each image.
> 3. **Healthcare:** Predict dengue outbreak risk based on rainfall, temperature, and mosquito population data. Labels = outbreak/no outbreak.

**Q7:** AI vs ML vs Deep Learning:
> - **AI** = All vehicles (any computer system that does "intelligent" tasks)
> - **ML** = Cars (a subset of AI that learns from data)
> - **Deep Learning** = Electric cars (a subset of ML using neural networks)
> All deep learning is ML, all ML is AI. But not all AI is ML, and not all ML is deep learning.

**Q8:** Grenada nutmeg prediction:
> **Supervised Learning** — The scientist has input features (tree age, rainfall, soil quality, sunlight) and a target to predict (spice yield). This is a regression problem because they're predicting a continuous number (how much spice). They would train on historical data where both the inputs and actual yields are known.

---

## Lesson 4: First ML Model

**Q1:** Difference between regression and classification?
> **a) Regression predicts numbers, classification predicts categories** — Regression: "The price will be $2.50." Classification: "This fruit is a Mango."

**Q2:** Why split into training and testing?
> **c) To check if the model can handle NEW data it never seen** — If we only test on data it already saw, we can't tell if it truly learned or just memorized.

**Q3:** What does LabelEncoder do?
> **b) Converts text/categories into numbers fi ML** — ML models need numbers, so "Jamaica" becomes 5, "Mango" becomes 7, etc.

**Q4:** What does 95% accuracy mean?
> **c) It correctly predicts 95 out of 100 examples** — The model gets the right answer 95% of the time on test data.

**Q5:** What is a Decision Tree?
> **b) A model dat makes decisions using yes/no questions in a tree structure** — Like a flowchart: "Is sweetness > 7?" → "Is acidity < 3?" → "Tropical Sweet!"

**Q6:** Three features in the fruit price model:
> 1. **Fruit_Code** (which fruit it is)
> 2. **Island_Code** (which island/country)
> 3. **Season** (time of year: 1-4)

**Q7:** Grenada nutmeg prices — regression or classification?
> **Regression** — because price is a continuous number (e.g., $12.50 per pound), not a category. You want to predict the exact price value.

**Q8:** Is MAE of $0.50 good or bad for prices $0.50-$5.00?
> It's **moderate**. The price range is $4.50, so an average error of $0.50 is about 11% of the range. For a first model, that's decent! But for a market vendor, being off by $0.50 on a $1.00 item (50% error) is significant. The model could be improved with more data, more features, or a more complex algorithm.

**Q9:** Classification in a Caribbean sector:
> **Healthcare:** Classify patients as "Low Risk", "Medium Risk", or "High Risk" for dengue fever based on symptoms, location, recent rainfall, and travel history. This helps doctors prioritize patients across clinics in Trinidad, Jamaica, and other islands.

**Q10:** Why set random_state=42?
> **c) It ensures we get the same results each time we run it** — Random operations in ML (like splitting data) use a seed. Setting random_state means everyone running this code gets identical results, making it reproducible.

---

## Lesson 5: Data Visualization

**Q1:** Best chart for change over time?
> **c) Line graph** — Line graphs show trends and patterns over time beautifully. Each point connects to the next, showing the flow.

**Q2:** When to use a pie chart?
> **c) Showing parts of a whole (percentages)** — Pie charts work best when showing how a total breaks down into categories (e.g., 35% tourism, 12% agriculture).

**Q3:** What does plt.savefig() do?
> **b) Saves the chart as an image file** — You can save as PNG, JPG, PDF, or SVG.

**Q4:** Why use different colors?
> **b) To make it easier to tell dem apart** — Color coding helps viewers quickly distinguish between different categories, countries, or data series.

**Q5:** Which decade had the most major hurricanes?
> **2000s** — with 8 Category 3-5 hurricanes in the Caribbean. Climate scientists warn this trend may continue due to warming ocean temperatures.

**Q6:** Which sector benefits most from data visualization?
> Multiple valid answers. **Tourism** is strong because: tourism boards need to communicate visitor trends to government, show seasonal patterns to hotels, compare destinations, and present economic impact data to investors. A clear chart can convince a government to invest millions in tourism infrastructure.
> Also excellent: **Climate/Hurricane preparedness** — visualizing hurricane paths and intensity helps save lives.

**Q7:** Two things every chart should have?
> **b) A title and axis labels** — Without a title, nobody knows what the chart shows. Without axis labels, nobody knows what the numbers mean.

**Q8:** Temperature vs tourist arrivals chart type:
> **Scatter plot** — because you're looking at the RELATIONSHIP between two numerical variables. Each dot represents one data point (e.g., one month), with temperature on one axis and tourist arrivals on the other. You could see if warmer months correlate with more visitors.

---

## Lesson 6: Sports Analytics

**Q1:** All-time leading West Indies Test run scorer?
> **b) Brian Lara** — 11,953 runs in 131 matches. He also holds the record for the highest individual Test score of 400 not out (vs England, 2004)!

**Q2:** Country with most WI cricket legends in our dataset?
> **d) Guyana** — Garfield Sobers (born Barbados but...) — actually looking at the data: **Barbados and Guyana** are tied with multiple players each. Barbados has Sobers, Greenidge, Haynes, Roach, Holder, and Brathwaite. Guyana has Chanderpaul, Lloyd, Kanhai, Sarwan, Hooper. **Barbados** has the most.

**Q3:** How many Olympic Gold medals did Usain Bolt win?
> **c) 8** — Three consecutive Olympic 100m golds (2008, 2012, 2016), three consecutive 200m golds, plus two 4x100m relay golds (one was later stripped due to a teammate's positive test, but he won 8 at the time).

**Q4:** What does batting average measure?
> **b) Average runs scored per dismissal** — Calculated as Total Runs / Times Out. A higher average means the batsman scores more runs before getting out.

**Q5:** Which CPL team has the most titles?
> **c) Trinbago Knight Riders** — with 4 titles. They are the most successful CPL franchise, based in Trinidad & Tobago.

**Q6:** Three Caribbean track & field athletes NOT from Jamaica:
> - **Hasely Crawford** (Trinidad & Tobago) — 1976 Olympic 100m Gold
> - **Kirani James** (Grenada) — 2012 Olympic 400m Gold
> - **Kim Collins** (St. Kitts & Nevis) — 2003 World 100m Champion
> - Also: Ato Boldon (Trinidad), Pauline Davis-Thompson (Bahamas), Obadele Thompson (Barbados), Steven Gardiner (Bahamas), Julien Alfred (St. Lucia), Anderson Peters (Grenada)

**Q7:** Wickets-per-match rate:
> 300 wickets / 80 matches = **3.75 wickets per match**. That's an excellent rate! For reference, Courtney Walsh took 519 wickets in 132 matches = 3.93 per match.

**Q8:** How AI could help Caribbean cricket:
> 1. **Opposition Analysis:** AI can analyze thousands of hours of footage to find patterns in opposing batsmen's weaknesses. For example, analyzing which deliveries a particular batsman struggles against, helping bowlers plan their strategy.
> 2. **Player Fitness Monitoring:** AI can track player workload, injury risk factors, and recovery patterns. Wearable sensors combined with ML can predict when a fast bowler is at risk of injury based on bowling speed, action biomechanics, and training load.

**Q9:** Bolt vs Fraser-Pryce height difference:
> (196 - 152) / 152 * 100 = 44 / 152 * 100 = **28.9%** — Bolt is almost 29% taller than Fraser-Pryce! Yet both are among the fastest humans ever. Speed isn't just about height!

**Q10:** Which sport benefits most from data analytics?
> Multiple valid answers. **Cricket** is a strong choice because: it generates massive amounts of statistical data (every ball is recorded), has complex strategic decisions (field placement, bowling changes, batting order), and Caribbean teams could use analytics to compete with wealthier nations who already use advanced data science. The CPL could also use analytics for player auctions, team strategy, and fan engagement.

---

## Capstone Project: Caribbean Quiz Bot

The quiz bot is a hands-on project — no fixed "answers" since yuh writing yuh own code! But here's guidance on the TODOs:

**TODO 1-4 (Adding questions):** Any valid questions about Caribbean geography, history, culture, and sports are correct. Make sure the "answer" field EXACTLY matches one of the "options".

**TODO 5 (select_questions):**
```python
if category is not None:
    filtered = [q for q in filtered if q["category"] == category]
if difficulty is not None:
    filtered = [q for q in filtered if q["difficulty"] == difficulty]
```

**TODO 6 (ask_question validation):** Already provided as an example in the template. The key logic is converting the user's input to an integer, checking it's 1-4, and comparing the selected option to the answer.

**TODO 7 (calculate_grade):** Already provided as an example. Use if/elif/else with percentage thresholds.

**TODO 8 (show_results):** Already provided as an example. Count correct/total by category using a dictionary.

---

> *"Di answers nah di end — dem just di beginning of deeper understanding. Keep learning, keep building, keep pushing di Caribbean forward!"*
>
> — [Adrian Dunkley](https://Adriandunkley.net)

---

<p align="center"><em>Built with love in the Caribbean, for the Caribbean. 100% FREE.</em></p>
