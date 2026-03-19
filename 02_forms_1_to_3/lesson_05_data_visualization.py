"""
=============================================================================
LESSON 5: Data Visualization — Paint di Caribbean Wid Data! 🎨📊🌴
=============================================================================
Caribbean AI Academy — Forms 1-3 (Ages 11-14)
Designed by Adrian Dunkley (https://Adriandunkley.net)
FREE for Caribbean students who want to be AI Engineers, Scientists,
and Entrepreneurs.

Wah gwaan! Time fi mek some BEAUTIFUL charts!

A chart can tell a story faster than a thousand numbers. In dis lesson,
we go use matplotlib fi create:
  - Bar charts (Caribbean populations)
  - Line graphs (tourism trends over time)
  - Pie charts (Caribbean economic sectors)
  - Scatter plots (comparing island data)
  - Hurricane frequency plots

We use Caribbean color schemes — di colors of we flags and culture! 🇯🇲🇹🇹🇧🇧

Run dis file and look at di charts dat pop up. Beautiful, right?
=============================================================================
"""

import matplotlib.pyplot as plt
import matplotlib
import numpy as np
import pandas as pd

# Use a clean style
matplotlib.rcParams.update({'font.size': 11})

# =============================================================================
# Caribbean Color Palettes — We Use We Own Colors!
# =============================================================================
# Inspired by Caribbean flags, sunsets, and ocean colors

CARIBBEAN_COLORS = {
    "ocean": ["#006994", "#0099CC", "#00CED1", "#40E0D0", "#48D1CC"],
    "sunset": ["#FF6B35", "#F7C548", "#E63946", "#FFB703", "#FB8500"],
    "flags": ["#009B3A", "#FED100", "#000000", "#CE1126", "#00209F"],
    "tropical": ["#2EC4B6", "#E71D36", "#FF9F1C", "#011627", "#FDFFFC"],
    "rasta": ["#E03C31", "#FFD100", "#009B3A"],
    "ocean_deep": ["#003049", "#D62828", "#F77F00", "#FCBF49", "#EAE2B7"],
}

# =============================================================================
# CHART 1: BAR CHART — Caribbean Island Populations
# =============================================================================

print("=" * 60)
print("CHART 1: Caribbean Population Bar Chart")
print("=" * 60)

# Population data (approximate, 2024)
countries = [
    "Cuba", "Dominican\nRepublic", "Haiti", "Jamaica",
    "Trinidad &\nTobago", "Guyana", "Suriname", "Bahamas",
    "Barbados", "St. Lucia", "Grenada", "St. Vincent",
    "Antigua &\nBarbuda", "Dominica", "St. Kitts\n& Nevis",
    "Belize",
]

populations = [
    11_212, 11_229, 11_725, 2_828,
    1_405, 808, 618, 412,
    288, 180, 125, 101,
    100, 73, 48,
    441,
]  # in thousands

fig, ax = plt.subplots(figsize=(14, 7))

# Create color gradient based on population size
colors = plt.cm.YlOrRd(np.linspace(0.3, 0.9, len(countries)))
# Sort by population for better visualization
sorted_pairs = sorted(zip(populations, countries, colors), reverse=True)
sorted_pops = [p for p, c, col in sorted_pairs]
sorted_countries = [c for p, c, col in sorted_pairs]
sorted_colors = [col for p, c, col in sorted_pairs]

bars = ax.bar(sorted_countries, sorted_pops, color=sorted_colors, edgecolor="white", linewidth=0.5)

# Add value labels on top of each bar
for bar, pop in zip(bars, sorted_pops):
    ax.text(bar.get_x() + bar.get_width()/2., bar.get_height() + 100,
            f'{pop:,}K', ha='center', va='bottom', fontsize=8, fontweight='bold')

ax.set_title("Caribbean Island Populations (thousands, 2024)", fontsize=16, fontweight='bold', pad=15)
ax.set_ylabel("Population (thousands)", fontsize=12)
ax.set_xlabel("Country/Territory", fontsize=12)
ax.tick_params(axis='x', rotation=45, labelsize=9)
ax.grid(axis='y', alpha=0.3)
ax.set_axisbelow(True)

# Add a note
ax.text(0.98, 0.95, "Data: Approximate 2024 estimates\nCaribbean AI Academy",
        transform=ax.transAxes, fontsize=8, ha='right', va='top',
        style='italic', alpha=0.6)

plt.tight_layout()
plt.savefig("/home/user/Caribbean-AI-Students/02_forms_1_to_3/chart_population.png", dpi=150)
plt.close()
print("Chart saved! ✅ (chart_population.png)")


# =============================================================================
# CHART 2: LINE GRAPH — Tourism Trends Over Time
# =============================================================================

print("\n" + "=" * 60)
print("CHART 2: Tourism Trends Line Graph")
print("=" * 60)

# Tourism visitor data (thousands) — shows COVID impact and recovery
years = [2018, 2019, 2020, 2021, 2022, 2023]

tourism_trends = {
    "Jamaica":             [2_473, 2_681, 838,   1_215, 2_503, 3_450],
    "Dominican Republic":  [6_569, 7_135, 2_400, 3_800, 5_600, 7_200],
    "Bahamas":             [6_620, 7_250, 1_800, 2_100, 4_500, 6_200],
    "Barbados":            [680,   718,   125,   218,   520,   625],
    "Trinidad & Tobago":   [395,   410,   112,   98,    245,   378],
    "St. Lucia":           [395,   423,   89,    155,   340,   425],
    "Cuba":                [4_684, 4_276, 1_100, 356,   890,   1_560],
}

fig, ax = plt.subplots(figsize=(12, 7))

colors_line = ["#009B3A", "#CE1126", "#00CED1", "#006994",
               "#FF6B35", "#F7C548", "#E63946"]

for (country, visitors), color in zip(tourism_trends.items(), colors_line):
    ax.plot(years, visitors, marker='o', linewidth=2.5, label=country,
            color=color, markersize=6)

# Highlight COVID year
ax.axvspan(2019.5, 2020.5, alpha=0.15, color='red', label='COVID-19 Impact')
ax.annotate('COVID-19\nPandemic', xy=(2020, 500), fontsize=10,
            ha='center', color='red', fontweight='bold')

ax.set_title("Caribbean Tourism: Visitors Over Time (thousands)", fontsize=15, fontweight='bold')
ax.set_xlabel("Year", fontsize=12)
ax.set_ylabel("Visitors (thousands)", fontsize=12)
ax.legend(loc='upper left', fontsize=9, framealpha=0.9)
ax.grid(alpha=0.3)
ax.set_axisbelow(True)
ax.set_xticks(years)

plt.tight_layout()
plt.savefig("/home/user/Caribbean-AI-Students/02_forms_1_to_3/chart_tourism_trends.png", dpi=150)
plt.close()
print("Chart saved! ✅ (chart_tourism_trends.png)")


# =============================================================================
# CHART 3: PIE CHART — Caribbean Economic Sectors
# =============================================================================

print("\n" + "=" * 60)
print("CHART 3: Caribbean Economic Sectors Pie Chart")
print("=" * 60)

# Average Caribbean economy breakdown (simplified)
sectors = ["Tourism &\nHospitality", "Agriculture\n& Fishing", "Financial\nServices",
           "Manufacturing", "Mining &\nEnergy", "Construction",
           "Other\nServices"]
percentages = [35, 12, 15, 8, 10, 7, 13]

fig, ax = plt.subplots(figsize=(10, 8))

colors_pie = ["#00CED1", "#009B3A", "#FFD100", "#FF6B35",
              "#8B4513", "#E63946", "#006994"]

wedges, texts, autotexts = ax.pie(
    percentages, labels=sectors, autopct='%1.0f%%',
    colors=colors_pie, startangle=90,
    textprops={'fontsize': 10},
    pctdistance=0.78,
    wedgeprops=dict(linewidth=2, edgecolor='white')
)

for autotext in autotexts:
    autotext.set_fontweight('bold')
    autotext.set_fontsize(11)

ax.set_title("Caribbean Economic Sectors\n(Average across CARICOM nations)",
             fontsize=15, fontweight='bold')

plt.tight_layout()
plt.savefig("/home/user/Caribbean-AI-Students/02_forms_1_to_3/chart_economic_sectors.png", dpi=150)
plt.close()
print("Chart saved! ✅ (chart_economic_sectors.png)")


# =============================================================================
# CHART 4: SCATTER PLOT — Population vs Tourism Revenue
# =============================================================================

print("\n" + "=" * 60)
print("CHART 4: Population vs Tourism Revenue Scatter Plot")
print("=" * 60)

# Data for scatter plot
scatter_data = {
    "Country":    ["Jamaica", "Trinidad", "Barbados", "Bahamas", "Dom. Rep.",
                   "Cuba", "St. Lucia", "Grenada", "Guyana", "Belize",
                   "Antigua", "Dominica", "St. Kitts"],
    "Population": [2828, 1405, 288, 412, 11229,
                   11212, 180, 125, 808, 441,
                   100, 73, 48],
    "Tourism_Revenue_M": [5120, 580, 1350, 11500, 12800,
                          2350, 1020, 330, 260, 620,
                          1150, 85, 420],
}

sdf = pd.DataFrame(scatter_data)

fig, ax = plt.subplots(figsize=(11, 7))

# Size of dot based on revenue per capita
sdf["Rev_Per_Capita"] = sdf["Tourism_Revenue_M"] / sdf["Population"] * 1000
sizes = sdf["Rev_Per_Capita"] * 20  # Scale for visibility

scatter = ax.scatter(sdf["Population"], sdf["Tourism_Revenue_M"],
                     s=sizes, c=CARIBBEAN_COLORS["ocean_deep"],
                     alpha=0.7, edgecolors='white', linewidth=1.5)

# Label each point
for _, row in sdf.iterrows():
    offset = (15, 10) if row["Country"] != "Dom. Rep." else (15, -15)
    ax.annotate(row["Country"], (row["Population"], row["Tourism_Revenue_M"]),
                xytext=offset, textcoords='offset points',
                fontsize=9, fontweight='bold', alpha=0.8)

ax.set_title("Population vs Tourism Revenue (2023)\nBubble size = Revenue per capita",
             fontsize=14, fontweight='bold')
ax.set_xlabel("Population (thousands)", fontsize=12)
ax.set_ylabel("Tourism Revenue (Millions USD)", fontsize=12)
ax.grid(alpha=0.3)
ax.set_axisbelow(True)

plt.tight_layout()
plt.savefig("/home/user/Caribbean-AI-Students/02_forms_1_to_3/chart_pop_vs_tourism.png", dpi=150)
plt.close()
print("Chart saved! ✅ (chart_pop_vs_tourism.png)")


# =============================================================================
# CHART 5: HURRICANE FREQUENCY PLOT
# =============================================================================

print("\n" + "=" * 60)
print("CHART 5: Caribbean Hurricane Frequency")
print("=" * 60)

# Named hurricanes affecting Caribbean by decade (approximate historical data)
decades = ["1960s", "1970s", "1980s", "1990s", "2000s", "2010s", "2020s\n(so far)"]
cat1_2 = [8, 6, 7, 8, 10, 7, 5]      # Category 1-2
cat3_5 = [4, 3, 5, 5, 8, 6, 4]        # Category 3-5 (major)

fig, ax = plt.subplots(figsize=(11, 7))

x = np.arange(len(decades))
width = 0.35

bars1 = ax.bar(x - width/2, cat1_2, width, label='Category 1-2',
               color='#F7C548', edgecolor='white', linewidth=0.5)
bars2 = ax.bar(x + width/2, cat3_5, width, label='Category 3-5 (Major)',
               color='#E63946', edgecolor='white', linewidth=0.5)

# Add value labels
for bar in bars1:
    ax.text(bar.get_x() + bar.get_width()/2., bar.get_height() + 0.1,
            str(int(bar.get_height())), ha='center', va='bottom', fontsize=10, fontweight='bold')
for bar in bars2:
    ax.text(bar.get_x() + bar.get_width()/2., bar.get_height() + 0.1,
            str(int(bar.get_height())), ha='center', va='bottom', fontsize=10, fontweight='bold')

ax.set_title("Caribbean Hurricanes by Decade\nCategory 1-2 vs Major (3-5)",
             fontsize=15, fontweight='bold')
ax.set_xlabel("Decade", fontsize=12)
ax.set_ylabel("Number of Named Hurricanes", fontsize=12)
ax.set_xticks(x)
ax.set_xticklabels(decades, fontsize=10)
ax.legend(fontsize=11)
ax.grid(axis='y', alpha=0.3)
ax.set_axisbelow(True)

# Add note about climate
ax.text(0.02, 0.97, "Note: Climate change may increase\nhurricane intensity in coming decades.",
        transform=ax.transAxes, fontsize=9, ha='left', va='top',
        style='italic', alpha=0.6,
        bbox=dict(boxstyle='round,pad=0.3', facecolor='lightyellow', alpha=0.8))

plt.tight_layout()
plt.savefig("/home/user/Caribbean-AI-Students/02_forms_1_to_3/chart_hurricanes.png", dpi=150)
plt.close()
print("Chart saved! ✅ (chart_hurricanes.png)")


# =============================================================================
# CHART 6: HORIZONTAL BAR — Caribbean Agricultural Exports
# =============================================================================

print("\n" + "=" * 60)
print("CHART 6: Caribbean Agricultural Exports")
print("=" * 60)

products = ["Sugarcane", "Bananas", "Cocoa", "Coffee", "Nutmeg",
            "Rice", "Rum", "Citrus Fruits", "Coconut Products",
            "Spices (other)", "Tobacco", "Seafood"]
export_values = [850, 520, 340, 280, 120,
                 410, 680, 190, 155,
                 95, 210, 730]  # Millions USD (approximate combined)

# Sort
sorted_pairs = sorted(zip(export_values, products))
sorted_values = [v for v, p in sorted_pairs]
sorted_products = [p for v, p in sorted_pairs]

fig, ax = plt.subplots(figsize=(11, 7))

colors_h = plt.cm.Greens(np.linspace(0.3, 0.9, len(products)))
bars = ax.barh(sorted_products, sorted_values, color=colors_h, edgecolor='white', height=0.7)

for bar, val in zip(bars, sorted_values):
    ax.text(bar.get_width() + 10, bar.get_y() + bar.get_height()/2.,
            f'${val}M', ha='left', va='center', fontsize=10, fontweight='bold')

ax.set_title("Caribbean Agricultural & Food Exports\n(Combined CARICOM, approximate, Millions USD)",
             fontsize=14, fontweight='bold')
ax.set_xlabel("Export Value (Millions USD)", fontsize=12)
ax.grid(axis='x', alpha=0.3)
ax.set_axisbelow(True)

plt.tight_layout()
plt.savefig("/home/user/Caribbean-AI-Students/02_forms_1_to_3/chart_agriculture.png", dpi=150)
plt.close()
print("Chart saved! ✅ (chart_agriculture.png)")


# =============================================================================
# CHART 7: MULTI-PANEL — Caribbean Data Dashboard
# =============================================================================

print("\n" + "=" * 60)
print("CHART 7: Caribbean Data Dashboard (Multi-Panel)")
print("=" * 60)

fig, axes = plt.subplots(2, 2, figsize=(14, 10))
fig.suptitle("Caribbean Data Dashboard", fontsize=18, fontweight='bold', y=1.02)

# Panel 1: Top 5 tourism destinations (2023)
ax1 = axes[0, 0]
top5_countries = ["Dom. Republic", "Bahamas", "Jamaica", "Cuba", "Barbados"]
top5_visitors = [7200, 6200, 3450, 1560, 625]
ax1.barh(top5_countries, top5_visitors, color=CARIBBEAN_COLORS["ocean"])
ax1.set_title("Top 5 Tourism Destinations (2023)", fontweight='bold')
ax1.set_xlabel("Visitors (thousands)")

# Panel 2: Temperature across islands
ax2 = axes[0, 1]
temp_islands = ["Jamaica", "Trinidad", "Bahamas", "Barbados", "Guyana", "Belize"]
avg_temps = [27.5, 28.0, 25.5, 27.0, 28.5, 26.5]
ax2.bar(temp_islands, avg_temps, color=CARIBBEAN_COLORS["sunset"][:6])
ax2.set_title("Average Temperature (Celsius)", fontweight='bold')
ax2.set_ylim(20, 32)
ax2.tick_params(axis='x', rotation=30)

# Panel 3: Renewable energy adoption
ax3 = axes[1, 0]
energy_countries = ["Barbados", "Jamaica", "Guyana", "Trinidad", "Bahamas", "Belize"]
renewable_pct = [35, 18, 12, 5, 8, 42]
ax3.bar(energy_countries, renewable_pct, color=CARIBBEAN_COLORS["flags"][:6])
ax3.set_title("Renewable Energy (% of total)", fontweight='bold')
ax3.set_ylabel("Percentage (%)")
ax3.tick_params(axis='x', rotation=30)

# Panel 4: Internet penetration
ax4 = axes[1, 1]
net_countries = ["Bahamas", "Barbados", "Trinidad", "Jamaica", "Dom. Rep.",
                 "Guyana", "Cuba", "Haiti"]
internet_pct = [87, 82, 77, 68, 75, 55, 62, 35]
colors_bar = plt.cm.cool(np.linspace(0.2, 0.8, len(net_countries)))
ax4.barh(net_countries, internet_pct, color=colors_bar)
ax4.set_title("Internet Penetration (%)", fontweight='bold')
ax4.set_xlabel("% of Population")
ax4.set_xlim(0, 100)

plt.tight_layout()
plt.savefig("/home/user/Caribbean-AI-Students/02_forms_1_to_3/chart_dashboard.png", dpi=150)
plt.close()
print("Dashboard saved! ✅ (chart_dashboard.png)")


# =============================================================================
# SUMMARY
# =============================================================================

print("\n" + "=" * 60)
print("🎉 Yuh just created 7 PROFESSIONAL charts!")
print("=" * 60)
print("""
Charts created:
  1. chart_population.png      — Bar chart of Caribbean populations
  2. chart_tourism_trends.png  — Line graph of tourism over time
  3. chart_economic_sectors.png — Pie chart of economic sectors
  4. chart_pop_vs_tourism.png  — Scatter plot: population vs revenue
  5. chart_hurricanes.png      — Hurricane frequency by decade
  6. chart_agriculture.png     — Agricultural export values
  7. chart_dashboard.png       — Multi-panel Caribbean dashboard

Key matplotlib skills yuh learned:
  - plt.bar() / plt.barh() — Bar charts (vertical and horizontal)
  - plt.plot() — Line graphs
  - plt.pie() — Pie charts
  - plt.scatter() — Scatter plots
  - plt.subplots() — Multi-panel dashboards
  - Customizing colors, labels, titles, and legends
  - Saving charts as image files

Dis is exactly what data analysts and data scientists do fi
Caribbean tourism boards, government agencies, and businesses!
""")

print("Now try di quiz and move on to Lesson 6 (Sports Analytics)! 🏏")


# =============================================================================
# QUIZ TIME! 🧠
# =============================================================================
# Check quiz_answers.md when yuh done!
#
# Q1: Which chart type is BEST fi showing change over time?
#     a) Pie chart  b) Bar chart  c) Line graph  d) Scatter plot
#
# Q2: When would yuh use a pie chart?
#     a) Showing trends over time  b) Comparing two variables
#     c) Showing parts of a whole (percentages)  d) Showing relationships
#
# Q3: What does plt.savefig() do?
#     a) Saves data to a CSV  b) Saves di chart as an image file
#     c) Saves Python code  d) Prints di chart
#
# Q4: Why did we use different colors fi different countries/categories?
#     a) Just fi fun  b) To make it easier fi tell dem apart
#     c) Python requires it  d) To save memory
#
# Q5: Look at di hurricane chart. Which decade had di MOST major
#     (Category 3-5) hurricanes?
#
# Q6: Which Caribbean sector (tourism, agriculture, fishing, healthcare,
#     energy, climate) would benefit MOST from data visualization? Why?
#
# Q7: What TWO things should EVERY chart have?
#     a) Colors and lines  b) A title and axis labels
#     c) Pie slices and bars  d) Legends and grids
#
# Q8: If yuh wanted fi show di relationship between temperature and
#     tourist arrivals, which chart type would yuh use? Why?
# =============================================================================
