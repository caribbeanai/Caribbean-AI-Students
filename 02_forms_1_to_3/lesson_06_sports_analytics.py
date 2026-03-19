"""
=============================================================================
LESSON 6: Sports Analytics — Caribbean Sporting Legends! 🏏🏃‍♂️🌴
=============================================================================
Caribbean AI Academy — Forms 1-3 (Ages 11-14)
Designed by Adrian Dunkley (https://Adriandunkley.net)
FREE for Caribbean students who want to be AI Engineers, Scientists,
and Entrepreneurs.

BIG LESSON! We analyzing Caribbean sports data!
West Indies cricket, Caribbean Premier League, track & field legends,
and more.

In dis lesson, yuh go:
  - Analyze West Indies cricket batting and bowling stats
  - Look at CPL (Caribbean Premier League) team performance
  - Study track & field data (Bolt, Fraser-Pryce, and more!)
  - Calculate averages, strike rates, and other sports stats
  - Build simple visualizations of sports data

Dis is REAL sports analytics — di same ting professional teams use! 🏆
=============================================================================
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# =============================================================================
# PART 1: WEST INDIES CRICKET — BATTING LEGENDS
# =============================================================================

print("=" * 65)
print("PART 1: WEST INDIES CRICKET BATTING LEGENDS 🏏")
print("=" * 65)

# West Indies cricket legends — batting statistics (career Test stats, approximate)
batting_data = {
    "Player": [
        "Brian Lara", "Vivian Richards", "Garfield Sobers",
        "Chris Gayle", "Shivnarine Chanderpaul", "Clive Lloyd",
        "Gordon Greenidge", "Desmond Haynes", "Rohan Kanhai",
        "Ramnaresh Sarwan", "Carl Hooper", "Richie Richardson",
        "Darren Bravo", "Kraigg Brathwaite", "Shai Hope",
    ],
    "Country": [
        "Trinidad & Tobago", "Antigua & Barbuda", "Barbados",
        "Jamaica", "Guyana", "Guyana",
        "Barbados", "Barbados", "Guyana",
        "Guyana", "Guyana", "Antigua & Barbuda",
        "Trinidad & Tobago", "Barbados", "Barbados",
    ],
    "Matches": [131, 121, 93, 103, 164, 110,
                108, 116, 79, 87, 102, 86,
                58, 80, 52],
    "Innings": [232, 182, 160, 176, 280, 175,
                185, 202, 137, 154, 173, 146,
                99, 148, 92],
    "Runs": [11953, 8540, 8032, 7214, 11867, 7515,
             7558, 7487, 6227, 5842, 5762, 5949,
             3943, 4540, 3520],
    "Highest_Score": [400, 291, 365, 333, 203, 242,
                      226, 184, 256, 291, 233, 194,
                      218, 212, 170],
    "Centuries": [34, 24, 26, 15, 30, 19,
                  19, 18, 15, 15, 13, 16,
                  9, 10, 8],
    "Average": [52.89, 50.24, 57.78, 42.18, 51.37, 46.68,
                44.72, 42.30, 47.53, 40.01, 36.47, 44.39,
                40.03, 33.14, 39.77],
    "Era": ["1990s-2000s", "1970s-1990s", "1950s-1970s", "2000s-2010s",
            "1990s-2010s", "1960s-1980s", "1970s-1990s", "1970s-1990s",
            "1950s-1970s", "2000s-2010s", "1980s-2000s", "1980s-1990s",
            "2010s-2020s", "2010s-2020s", "2010s-2020s"],
}

bat_df = pd.DataFrame(batting_data)

print("\n--- West Indies Test Cricket Batting Legends ---")
print(bat_df[["Player", "Country", "Matches", "Runs", "Average", "Centuries"]].to_string(index=False))

# Analysis: Top run scorers
print("\n--- Top 5 Run Scorers ---")
top_scorers = bat_df.nlargest(5, "Runs")
for i, (_, row) in enumerate(top_scorers.iterrows(), 1):
    print(f"  {i}. {row['Player']} ({row['Country']}): {row['Runs']:,} runs in {row['Matches']} matches")

# Highest average (minimum 50 innings)
print("\n--- Highest Batting Averages ---")
qualified = bat_df[bat_df["Innings"] >= 50].nlargest(5, "Average")
for _, row in qualified.iterrows():
    print(f"  {row['Player']}: {row['Average']:.2f} ({row['Runs']:,} runs)")

# Country contribution
print("\n--- Runs by Country ---")
country_runs = bat_df.groupby("Country")["Runs"].sum().sort_values(ascending=False)
for country, runs in country_runs.items():
    bar = "█" * (runs // 1000)
    print(f"  {country:25s}: {runs:>6,} runs {bar}")

# Best strike rate of centuries (centuries per innings)
bat_df["Century_Rate"] = (bat_df["Centuries"] / bat_df["Innings"] * 100).round(1)
print("\n--- Century Rate (centuries per 100 innings) ---")
for _, row in bat_df.nlargest(5, "Century_Rate").iterrows():
    print(f"  {row['Player']}: {row['Century_Rate']}% (scored {row['Centuries']} centuries)")


# =============================================================================
# PART 2: WEST INDIES BOWLING LEGENDS
# =============================================================================

print("\n\n" + "=" * 65)
print("PART 2: WEST INDIES BOWLING LEGENDS 🏏🔥")
print("=" * 65)

bowling_data = {
    "Player": [
        "Curtly Ambrose", "Courtney Walsh", "Malcolm Marshall",
        "Joel Garner", "Michael Holding", "Andy Roberts",
        "Lance Gibbs", "Kemar Roach", "Jason Holder",
        "Shannon Gabriel", "Jerome Taylor", "Fidel Edwards",
    ],
    "Country": [
        "Antigua & Barbuda", "Jamaica", "Barbados",
        "Barbados", "Jamaica", "Antigua & Barbuda",
        "Guyana", "Barbados", "Barbados",
        "Trinidad & Tobago", "Jamaica", "Barbados",
    ],
    "Matches": [98, 132, 81, 58, 60, 47, 79, 71, 55, 56, 46, 55],
    "Wickets": [405, 519, 376, 259, 249, 202, 309, 258, 133, 162, 128, 165],
    "Best_Bowling": ["8/45", "7/37", "7/22", "6/56", "8/92", "7/54",
                     "8/38", "6/48", "6/42", "5/11", "5/48", "6/90"],
    "Bowling_Average": [20.99, 24.44, 20.95, 20.98, 23.69, 25.61,
                        29.09, 27.67, 33.56, 30.86, 35.47, 37.87],
    "Five_Wicket_Hauls": [22, 22, 22, 7, 13, 11, 18, 11, 2, 5, 4, 5],
    "Type": ["Fast", "Fast", "Fast", "Fast", "Fast", "Fast",
             "Spin", "Fast", "Fast-Medium", "Fast", "Fast", "Fast"],
}

bowl_df = pd.DataFrame(bowling_data)

print("\n--- West Indies Test Bowling Legends ---")
print(bowl_df[["Player", "Country", "Wickets", "Bowling_Average", "Five_Wicket_Hauls"]].to_string(index=False))

# Top wicket takers
print("\n--- Top 5 Wicket Takers ---")
for i, (_, row) in enumerate(bowl_df.nlargest(5, "Wickets").iterrows(), 1):
    print(f"  {i}. {row['Player']} ({row['Country']}): {row['Wickets']} wickets @ {row['Bowling_Average']:.2f}")

# Best bowling averages (lower is better!)
print("\n--- Best Bowling Averages (lower = better) ---")
for _, row in bowl_df.nsmallest(5, "Bowling_Average").iterrows():
    print(f"  {row['Player']}: {row['Bowling_Average']:.2f} ({row['Wickets']} wickets)")

# Fast bowlers vs Spinners
print("\n--- Fast vs Spin ---")
for btype in ["Fast", "Spin", "Fast-Medium"]:
    group = bowl_df[bowl_df["Type"] == btype]
    if not group.empty:
        print(f"  {btype}: {len(group)} bowlers, {group['Wickets'].sum()} total wickets")


# =============================================================================
# PART 3: CARIBBEAN PREMIER LEAGUE (CPL)
# =============================================================================

print("\n\n" + "=" * 65)
print("PART 3: CARIBBEAN PREMIER LEAGUE (CPL) 🏏🎉")
print("=" * 65)

# CPL teams and their performance (approximate/fictional season data)
cpl_data = {
    "Team": [
        "Trinbago Knight Riders", "Jamaica Tallawahs",
        "Barbados Royals", "Guyana Amazon Warriors",
        "St Kitts & Nevis Patriots", "St Lucia Kings",
    ],
    "Home_Country": [
        "Trinidad & Tobago", "Jamaica",
        "Barbados", "Guyana",
        "St. Kitts & Nevis", "St. Lucia",
    ],
    "Titles_Won": [4, 1, 0, 0, 0, 0],
    "Seasons_Played": [11, 11, 11, 11, 11, 11],
    "Total_Wins": [62, 48, 43, 55, 38, 36],
    "Total_Losses": [40, 54, 55, 48, 56, 60],
    "Highest_Team_Score": [267, 243, 236, 258, 225, 219],
    "Star_Players": [
        "Sunil Narine, Kieron Pollard, DJ Bravo",
        "Chris Gayle, Andre Russell, Rovman Powell",
        "Jason Holder, Kyle Mayers, Quinton de Kock",
        "Shimron Hetmyer, Imran Tahir, Brandon King",
        "Evin Lewis, Dwayne Bravo, Faf du Plessis",
        "Tim David, Roston Chase, Faf du Plessis",
    ]
}

cpl_df = pd.DataFrame(cpl_data)

# Win percentage
cpl_df["Win_Pct"] = round(cpl_df["Total_Wins"] / (cpl_df["Total_Wins"] + cpl_df["Total_Losses"]) * 100, 1)

print("\n--- CPL Team Performance ---")
print(cpl_df[["Team", "Titles_Won", "Total_Wins", "Total_Losses", "Win_Pct"]].to_string(index=False))

# Rankings by win percentage
print("\n--- CPL Rankings by Win Percentage ---")
for i, (_, row) in enumerate(cpl_df.sort_values("Win_Pct", ascending=False).iterrows(), 1):
    print(f"  {i}. {row['Team']}: {row['Win_Pct']}% wins ({row['Titles_Won']} titles)")

# Chris Gayle: Universe Boss!
# 🧩 Puzzle Piece 3/12: Add two digits: the jersey number of a cricket all-rounder who is 'Universe Boss' minus 40
# (Chris Gayle wears #333... but his JERSEY number is 45. 45 - 40 = 05)

print(f"\n🌟 Fun Fact: Chris Gayle ('Universe Boss') has hit di most sixes")
print(f"   in T20 cricket history! Jamaica represent! 🇯🇲")


# =============================================================================
# PART 4: TRACK & FIELD — CARIBBEAN SPEED DEMONS! 🏃‍♂️💨
# =============================================================================

print("\n\n" + "=" * 65)
print("PART 4: TRACK & FIELD LEGENDS 🏃‍♂️🏃‍♀️💨")
print("=" * 65)

# Caribbean track & field legends
track_data = {
    "Athlete": [
        "Usain Bolt", "Shelly-Ann Fraser-Pryce", "Elaine Thompson-Herah",
        "Yohan Blake", "Asafa Powell", "Veronica Campbell-Brown",
        "Shericka Jackson", "Dina Asher-Smith", "Hasely Crawford",
        "Kim Collins", "Pauline Davis-Thompson", "Obadele Thompson",
        "Ato Boldon", "Kirani James", "Steven Gardiner",
        "Julien Alfred", "Anderson Peters",
    ],
    "Country": [
        "Jamaica", "Jamaica", "Jamaica",
        "Jamaica", "Jamaica", "Jamaica",
        "Jamaica", "Jamaica (born UK)", "Trinidad & Tobago",
        "St. Kitts & Nevis", "Bahamas", "Barbados",
        "Trinidad & Tobago", "Grenada", "Bahamas",
        "St. Lucia", "Grenada",
    ],
    "Event": [
        "100m/200m", "100m", "100m/200m",
        "100m/200m", "100m", "100m/200m",
        "100m/200m", "100m/200m", "100m",
        "100m", "200m", "100m",
        "100m/200m", "400m", "400m",
        "100m", "Javelin",
    ],
    "Personal_Best": [
        "9.58s/19.19s", "10.60s", "10.54s/21.53s",
        "9.69s/19.26s", "9.72s", "10.76s/21.74s",
        "10.65s/21.45s", "10.83s/21.88s", "10.06s",
        "9.93s", "21.97s", "9.87s",
        "9.86s/19.80s", "43.74s", "43.48s",
        "10.72s", "93.07m",
    ],
    "Olympic_Gold": [8, 3, 5, 1, 0, 2, 2, 0, 1, 0, 1, 0, 0, 1, 1, 1, 0],
    "Olympic_Silver": [0, 4, 0, 1, 0, 0, 1, 0, 0, 0, 0, 1, 1, 0, 0, 1, 0],
    "Olympic_Bronze": [0, 1, 0, 0, 0, 1, 0, 1, 0, 0, 1, 0, 3, 0, 0, 0, 1],
    "World_Champs_Gold": [11, 5, 0, 1, 0, 2, 3, 1, 0, 0, 0, 0, 0, 1, 1, 1, 1],
}

track_df = pd.DataFrame(track_data)

# Total medals
track_df["Total_Olympic_Medals"] = track_df["Olympic_Gold"] + track_df["Olympic_Silver"] + track_df["Olympic_Bronze"]

print("\n--- Caribbean Track & Field Legends ---")
print(track_df[["Athlete", "Country", "Event", "Personal_Best", "Total_Olympic_Medals"]].to_string(index=False))

# Most decorated
print("\n--- Most Olympic Medals (Caribbean Athletes) ---")
for _, row in track_df.nlargest(8, "Total_Olympic_Medals").iterrows():
    medals = f"🥇x{row['Olympic_Gold']} 🥈x{row['Olympic_Silver']} 🥉x{row['Olympic_Bronze']}"
    print(f"  {row['Athlete']} ({row['Country']}): {row['Total_Olympic_Medals']} medals ({medals})")

# Country medal count
print("\n--- Olympic Medals by Country ---")
country_medals = track_df.groupby("Country").agg(
    Gold=("Olympic_Gold", "sum"),
    Silver=("Olympic_Silver", "sum"),
    Bronze=("Olympic_Bronze", "sum"),
    Total=("Total_Olympic_Medals", "sum"),
).sort_values("Total", ascending=False)
print(country_medals)

# Usain Bolt special stats
print("\n" + "⚡" * 30)
print("USAIN BOLT — The Greatest Sprinter of All Time! 🇯🇲")
print("⚡" * 30)
bolt_stats = {
    "100m World Record": "9.58s (Berlin, 2009)",
    "200m World Record": "19.19s (Berlin, 2009)",
    "Olympic Gold Medals": 8,
    "World Championship Gold": 11,
    "Home Parish": "Trelawny, Jamaica",
    "Height": "6'5\" (196 cm)",
    "Nickname": "Lightning Bolt ⚡",
    "Top Speed Recorded": "44.72 km/h (27.8 mph)",
}
for stat, value in bolt_stats.items():
    print(f"  {stat}: {value}")

# Shelly-Ann Fraser-Pryce special stats
print(f"\n🏃‍♀️ SHELLY-ANN FRASER-PRYCE — The Pocket Rocket! 🇯🇲")
safp_stats = {
    "100m Personal Best": "10.60s",
    "Olympic Medals": "3 Gold, 4 Silver, 1 Bronze (8 total)",
    "World Championship Gold": 5,
    "Home": "Kingston, Jamaica",
    "Height": "5'0\" (152 cm) — di fastest short person ever!",
    "Nickname": "Pocket Rocket / Mommy Rocket 🚀",
    "Notable": "Became fastest woman alive at age 35!",
}
for stat, value in safp_stats.items():
    print(f"  {stat}: {value}")


# =============================================================================
# PART 5: SPORTS DATA VISUALIZATION
# =============================================================================

print("\n\n" + "=" * 65)
print("PART 5: SPORTS DATA VISUALIZATION 📊")
print("=" * 65)

# Chart 1: West Indies batting legends — runs comparison
fig, axes = plt.subplots(2, 2, figsize=(15, 12))
fig.suptitle("Caribbean Sports Analytics Dashboard", fontsize=18, fontweight='bold')

# Panel 1: Top batsmen by runs
ax1 = axes[0, 0]
top_bats = bat_df.nlargest(8, "Runs")
colors_bat = plt.cm.YlOrRd(np.linspace(0.3, 0.9, len(top_bats)))
ax1.barh(top_bats["Player"], top_bats["Runs"], color=colors_bat)
ax1.set_title("West Indies: Top Test Run Scorers", fontweight='bold')
ax1.set_xlabel("Total Runs")
for i, (_, row) in enumerate(top_bats.iterrows()):
    ax1.text(row["Runs"] + 100, i, f'{row["Runs"]:,}', va='center', fontsize=9)

# Panel 2: Top bowlers by wickets
ax2 = axes[0, 1]
top_bowl = bowl_df.nlargest(8, "Wickets")
colors_bowl = plt.cm.Blues(np.linspace(0.3, 0.9, len(top_bowl)))
ax2.barh(top_bowl["Player"], top_bowl["Wickets"], color=colors_bowl)
ax2.set_title("West Indies: Top Test Wicket Takers", fontweight='bold')
ax2.set_xlabel("Total Wickets")
for i, (_, row) in enumerate(top_bowl.iterrows()):
    ax2.text(row["Wickets"] + 5, i, str(row["Wickets"]), va='center', fontsize=9)

# Panel 3: CPL win percentages
ax3 = axes[1, 0]
cpl_sorted = cpl_df.sort_values("Win_Pct", ascending=True)
team_short = [t.split()[-1] if len(t.split()) > 1 else t for t in cpl_sorted["Team"]]
colors_cpl = ["#E63946", "#009B3A", "#006994", "#FFD100", "#FF6B35", "#00CED1"]
ax3.barh(team_short, cpl_sorted["Win_Pct"], color=colors_cpl)
ax3.set_title("CPL: Win Percentage by Team", fontweight='bold')
ax3.set_xlabel("Win %")
ax3.set_xlim(0, 100)
for i, pct in enumerate(cpl_sorted["Win_Pct"]):
    ax3.text(pct + 1, i, f'{pct}%', va='center', fontsize=10, fontweight='bold')

# Panel 4: Track & field medals by country
ax4 = axes[1, 1]
medal_countries = country_medals.head(6)
x = np.arange(len(medal_countries))
width = 0.25
ax4.bar(x - width, medal_countries["Gold"], width, label='Gold', color='#FFD700')
ax4.bar(x, medal_countries["Silver"], width, label='Silver', color='#C0C0C0')
ax4.bar(x + width, medal_countries["Bronze"], width, label='Bronze', color='#CD7F32')
ax4.set_xticks(x)
short_labels = [c[:10] for c in medal_countries.index]
ax4.set_xticklabels(short_labels, rotation=30, ha='right', fontsize=9)
ax4.set_title("Olympic Track & Field Medals (Caribbean)", fontweight='bold')
ax4.legend(fontsize=9)
ax4.set_ylabel("Medal Count")

plt.tight_layout()
plt.savefig("/home/user/Caribbean-AI-Students/02_forms_1_to_3/chart_sports_dashboard.png", dpi=150)
plt.close()
print("Sports dashboard saved! ✅ (chart_sports_dashboard.png)")


# =============================================================================
# PART 6: SIMPLE SPORTS STATISTICS CALCULATIONS
# =============================================================================

print("\n" + "=" * 65)
print("PART 6: SPORTS STATISTICS 📈")
print("=" * 65)

# Batting statistics explained
print("\n--- Cricket Batting Statistics Explained ---")
print("""
  Batting Average = Total Runs / Times Out
    - Brian Lara: 11,953 runs at 52.89 average
    - Dat mean fi every time him bat and get out, him score about 53 runs!

  Strike Rate = (Runs / Balls Faced) x 100
    - Higher strike rate = scoring faster

  Century Rate = (Centuries / Innings) x 100
    - How often a batsman scores 100+ in an innings
""")

# Calculate some stats
print("--- Caribbean Sport Facts ---")
total_test_runs = bat_df["Runs"].sum()
total_test_wickets = bowl_df["Wickets"].sum()
total_olympic_gold = track_df["Olympic_Gold"].sum()

print(f"  Total Test runs by WI legends in our data: {total_test_runs:,}")
print(f"  Total Test wickets by WI legends: {total_test_wickets:,}")
print(f"  Total Olympic Gold (Track & Field): {total_olympic_gold}")
print(f"  Average batting average: {bat_df['Average'].mean():.2f}")
print(f"  Average bowling average: {bowl_df['Bowling_Average'].mean():.2f}")

# Which country produces the best batsmen?
print("\n--- Average Batting Average by Country ---")
country_avg = bat_df.groupby("Country")["Average"].mean().sort_values(ascending=False)
for country, avg in country_avg.items():
    print(f"  {country}: {avg:.2f}")

# Fun comparisons
print("\n--- Fun Comparisons ---")
bolt_100m = 9.58
print(f"  Usain Bolt runs 100m in {bolt_100m}s")
print(f"  A cricket pitch is 22 yards (20.12m)")
bolt_pitch_time = bolt_100m * (20.12/100)
print(f"  Bolt could run di cricket pitch in about {bolt_pitch_time:.2f}s!")
print(f"  Most batsmen take about 3-4 seconds fi a quick single 😂")


print("\n" + "=" * 65)
print("🎉 YUH A SPORTS ANALYST NOW! Big tings! 🏏🏃‍♂️")
print("Caribbean sports is FULL of data fi analyze.")
print("Next up: Build yuh own Caribbean Quiz Bot (Capstone Project)!")
print("=" * 65)


# =============================================================================
# QUIZ TIME! 🧠
# =============================================================================
# Check quiz_answers.md when yuh done!
#
# Q1: Who is di all-time leading Test run scorer fi West Indies in our data?
#     a) Vivian Richards  b) Brian Lara  c) Chris Gayle  d) Garfield Sobers
#
# Q2: What country has di most West Indies cricket legends in our dataset?
#     a) Jamaica  b) Trinidad & Tobago  c) Barbados  d) Guyana
#
# Q3: How many Olympic Gold medals did Usain Bolt win?
#     a) 6  b) 7  c) 8  d) 9
#
# Q4: What does "batting average" measure in cricket?
#     a) How fast yuh score  b) Average runs scored per dismissal
#     c) Total runs in career  d) Number of centuries
#
# Q5: Which CPL team has the MOST titles in our data?
#     a) Jamaica Tallawahs  b) Guyana Amazon Warriors
#     c) Trinbago Knight Riders  d) Barbados Royals
#
# Q6: Name THREE Caribbean track & field athletes NOT from Jamaica.
#
# Q7: If a bowler took 300 wickets in 80 matches, what is their
#     wickets-per-match rate?
#
# Q8: How could AI and data analytics help Caribbean cricket teams
#     improve their performance? Give TWO specific examples.
#     (Think about: player selection, bowling strategy, fitness,
#      opposition analysis, field placement)
#
# Q9: Shelly-Ann Fraser-Pryce is 5'0" (152cm). Usain Bolt is 6'5" (196cm).
#     Bolt is what percentage taller than Fraser-Pryce?
#     (Calculate: (196 - 152) / 152 * 100)
#
# Q10: What Caribbean sport sector (cricket, track, football, swimming, etc.)
#      do yuh think would benefit MOST from data analytics? Why?
# =============================================================================
