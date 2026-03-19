"""
=============================================================================
LESSON 2: Data with Pandas — Caribbean Tourism Edition! 🐼🌴
=============================================================================
Caribbean AI Academy — Forms 1-3 (Ages 11-14)
Designed by Adrian Dunkley (https://Adriandunkley.net)
FREE for Caribbean students who want to be AI Engineers, Scientists,
and Entrepreneurs.

Ayyy! Yuh mek it to Lesson 2! Big up yuhself! 🎉

Pandas is one of di most powerful tools in Python fi working wid data.
Think of it like a super-powered spreadsheet (like Excel) but inside
Python. Data scientists and AI engineers use it EVERY DAY.

In dis lesson we go:
  - Learn what a DataFrame is (it's like a table)
  - Create Caribbean tourism data
  - Explore di data (peek at it, count it, describe it)
  - Filter and sort data
  - Do basic analysis (averages, totals, grouping)

Let's goooo! 🚀
=============================================================================
"""

import pandas as pd

# =============================================================================
# PART 1: CREATING A DATAFRAME — Our Caribbean Tourism Dataset
# =============================================================================

print("=" * 65)
print("PART 1: CREATING A DATAFRAME")
print("=" * 65)

# A DataFrame is like a table wid rows and columns.
# We go create a dataset about Caribbean tourism.
# Dis data is made up but realistic — based on real tourism trends!

tourism_data = {
    "Country": [
        "Jamaica", "Jamaica", "Jamaica", "Jamaica",
        "Trinidad & Tobago", "Trinidad & Tobago", "Trinidad & Tobago", "Trinidad & Tobago",
        "Barbados", "Barbados", "Barbados", "Barbados",
        "The Bahamas", "The Bahamas", "The Bahamas", "The Bahamas",
        "Dominican Republic", "Dominican Republic", "Dominican Republic", "Dominican Republic",
        "Cuba", "Cuba", "Cuba", "Cuba",
        "St. Lucia", "St. Lucia", "St. Lucia", "St. Lucia",
        "Antigua & Barbuda", "Antigua & Barbuda", "Antigua & Barbuda", "Antigua & Barbuda",
        "Grenada", "Grenada", "Grenada", "Grenada",
        "Guyana", "Guyana", "Guyana", "Guyana",
        "Haiti", "Haiti", "Haiti", "Haiti",
        "Belize", "Belize", "Belize", "Belize",
    ],
    "Year": [
        2020, 2021, 2022, 2023,
        2020, 2021, 2022, 2023,
        2020, 2021, 2022, 2023,
        2020, 2021, 2022, 2023,
        2020, 2021, 2022, 2023,
        2020, 2021, 2022, 2023,
        2020, 2021, 2022, 2023,
        2020, 2021, 2022, 2023,
        2020, 2021, 2022, 2023,
        2020, 2021, 2022, 2023,
        2020, 2021, 2022, 2023,
        2020, 2021, 2022, 2023,
    ],
    "Visitors_Thousands": [
        # Jamaica: big tourism drop in 2020 (COVID), recovering after
        838, 1_215, 2_503, 3_450,
        # Trinidad: smaller tourism sector, slower recovery
        112, 98, 245, 378,
        # Barbados: strong comeback
        125, 218, 520, 625,
        # Bahamas: cruise tourism hit hard, strong recovery
        1_800, 2_100, 4_500, 6_200,
        # Dominican Republic: largest in Caribbean
        2_400, 3_800, 5_600, 7_200,
        # Cuba: sanctions + COVID double whammy
        1_100, 356, 890, 1_560,
        # St. Lucia: steady growth
        89, 155, 340, 425,
        # Antigua & Barbuda: cruise dependent
        117, 85, 310, 490,
        # Grenada: small but growing
        48, 62, 135, 178,
        # Guyana: eco-tourism on the rise
        128, 103, 195, 315,
        # Haiti: affected by instability
        180, 72, 55, 68,
        # Belize: eco and adventure tourism
        131, 195, 340, 425,
    ],
    "Revenue_Millions_USD": [
        # Revenue correlates with visitors but varies by spending habits
        1_250, 1_820, 3_750, 5_120,
        180, 145, 385, 580,
        340, 520, 1_100, 1_350,
        3_200, 3_800, 8_200, 11_500,
        4_100, 6_200, 9_500, 12_800,
        1_800, 520, 1_280, 2_350,
        220, 365, 790, 1_020,
        290, 195, 720, 1_150,
        78, 105, 245, 330,
        95, 78, 155, 260,
        120, 42, 35, 45,
        180, 275, 490, 620,
    ],
    "Top_Sector": [
        "Beach", "Beach", "Beach", "All-Inclusive",
        "Carnival", "Carnival", "Carnival", "Carnival",
        "Beach", "Beach", "Beach", "Beach",
        "Cruise", "Cruise", "Cruise", "Cruise",
        "All-Inclusive", "All-Inclusive", "All-Inclusive", "All-Inclusive",
        "Culture", "Culture", "Beach", "Beach",
        "Beach", "Beach", "Eco-Tourism", "Eco-Tourism",
        "Cruise", "Cruise", "Cruise", "Cruise",
        "Eco-Tourism", "Eco-Tourism", "Eco-Tourism", "Eco-Tourism",
        "Eco-Tourism", "Eco-Tourism", "Eco-Tourism", "Eco-Tourism",
        "Culture", "Culture", "Culture", "Culture",
        "Eco-Tourism", "Eco-Tourism", "Eco-Tourism", "Adventure",
    ]
}

# Create the DataFrame — dis is where di magic happen!
df = pd.DataFrame(tourism_data)

print("We just created a DataFrame! Here's what it looks like:\n")
print(df)


# =============================================================================
# PART 2: EXPLORING THE DATA — Get fi Know It
# =============================================================================

print("\n" + "=" * 65)
print("PART 2: EXPLORING THE DATA")
print("=" * 65)

# .head() — see di first few rows (default 5)
print("\n--- First 5 rows (head) ---")
print(df.head())

# .tail() — see di last few rows
print("\n--- Last 3 rows (tail) ---")
print(df.tail(3))

# .shape — how many rows and columns?
print(f"\n--- Shape: {df.shape} ---")
print(f"That means {df.shape[0]} rows and {df.shape[1]} columns!")

# .columns — what are di column names?
print(f"\n--- Columns: {list(df.columns)} ---")

# .dtypes — what type of data is in each column?
print(f"\n--- Data Types ---")
print(df.dtypes)

# .describe() — quick statistics fi all number columns
print(f"\n--- Quick Stats (describe) ---")
print(df.describe())

# .info() — overall summary
print(f"\n--- DataFrame Info ---")
df.info()

# How many unique countries?
print(f"\n--- Unique countries: {df['Country'].nunique()} ---")
print(f"They are: {df['Country'].unique().tolist()}")


# =============================================================================
# PART 3: FILTERING DATA — Find Wha Yuh Looking For
# =============================================================================

print("\n" + "=" * 65)
print("PART 3: FILTERING DATA")
print("=" * 65)

# Filter: only Jamaica data
jamaica_df = df[df["Country"] == "Jamaica"]
print("--- Jamaica Tourism Data ---")
print(jamaica_df)

# Filter: only 2023 data
data_2023 = df[df["Year"] == 2023]
print("\n--- All Countries in 2023 ---")
print(data_2023[["Country", "Visitors_Thousands", "Revenue_Millions_USD"]])

# Filter: countries with more than 1 million visitors in 2023
big_tourism_2023 = data_2023[data_2023["Visitors_Thousands"] > 1000]
print("\n--- Countries with 1M+ visitors in 2023 ---")
print(big_tourism_2023[["Country", "Visitors_Thousands"]])

# Filter: Eco-Tourism destinations
eco_tourism = df[df["Top_Sector"] == "Eco-Tourism"]
print("\n--- Eco-Tourism Destinations ---")
print(eco_tourism[["Country", "Year", "Visitors_Thousands"]].to_string(index=False))

# Multiple filters: Jamaica in 2022 or 2023
jamaica_recent = df[(df["Country"] == "Jamaica") & (df["Year"] >= 2022)]
print("\n--- Jamaica 2022-2023 ---")
print(jamaica_recent)


# =============================================================================
# PART 4: SORTING DATA — Put Tings in Order
# =============================================================================

print("\n" + "=" * 65)
print("PART 4: SORTING DATA")
print("=" * 65)

# Sort by visitors in 2023 — who get di most tourists?
sorted_2023 = data_2023.sort_values("Visitors_Thousands", ascending=False)
print("--- 2023 Tourism Ranking (Most to Least Visitors) ---")
for i, (_, row) in enumerate(sorted_2023.iterrows(), 1):
    print(f"  {i}. {row['Country']}: {row['Visitors_Thousands']:,}K visitors")

# Sort by revenue per visitor (value tourism)
data_2023_copy = data_2023.copy()
data_2023_copy["Revenue_Per_Visitor_USD"] = (
    data_2023_copy["Revenue_Millions_USD"] * 1000 / data_2023_copy["Visitors_Thousands"]
)
sorted_by_value = data_2023_copy.sort_values("Revenue_Per_Visitor_USD", ascending=False)
print("\n--- 2023 Revenue Per Visitor (Highest Spending Tourists) ---")
for _, row in sorted_by_value.iterrows():
    print(f"  {row['Country']}: ${row['Revenue_Per_Visitor_USD']:,.0f} per visitor")


# =============================================================================
# PART 5: GROUPING AND AGGREGATION — Summarize Di Data
# =============================================================================

print("\n" + "=" * 65)
print("PART 5: GROUPING AND AGGREGATION")
print("=" * 65)

# Total visitors per country (all years combined)
total_by_country = df.groupby("Country")["Visitors_Thousands"].sum()
print("--- Total Visitors 2020-2023 (in thousands) ---")
print(total_by_country.sort_values(ascending=False))

# Average revenue per year
avg_revenue_by_year = df.groupby("Year")["Revenue_Millions_USD"].mean()
print(f"\n--- Average Revenue Per Country By Year ---")
print(avg_revenue_by_year)

# Total by sector
sector_totals = df.groupby("Top_Sector")["Visitors_Thousands"].sum()
print(f"\n--- Total Visitors by Tourism Sector ---")
print(sector_totals.sort_values(ascending=False))

# Multiple aggregations at once
summary = df.groupby("Country").agg(
    Total_Visitors=("Visitors_Thousands", "sum"),
    Avg_Revenue=("Revenue_Millions_USD", "mean"),
    Max_Visitors=("Visitors_Thousands", "max"),
).sort_values("Total_Visitors", ascending=False)

print(f"\n--- Country Summary (2020-2023) ---")
print(summary)


# =============================================================================
# PART 6: CREATING NEW COLUMNS — Add More Info
# =============================================================================

print("\n" + "=" * 65)
print("PART 6: CREATING NEW COLUMNS")
print("=" * 65)

# Calculate revenue per 1000 visitors (in millions USD)
df["Revenue_Per_1K_Visitors"] = round(
    df["Revenue_Millions_USD"] / df["Visitors_Thousands"], 2
)

# Classify tourism size
def classify_tourism(visitors_k):
    """Classify based on visitor numbers (in thousands)."""
    if visitors_k > 3000:
        return "Major Destination"
    elif visitors_k > 500:
        return "Growing Destination"
    elif visitors_k > 100:
        return "Emerging Destination"
    else:
        return "Small Destination"

df["Tourism_Class"] = df["Visitors_Thousands"].apply(classify_tourism)

print("--- Data with New Columns ---")
print(df[["Country", "Year", "Visitors_Thousands", "Tourism_Class", "Revenue_Per_1K_Visitors"]].head(12))

# Year-over-year growth calculation for each country
print("\n--- Year-over-Year Growth (Jamaica) ---")
jamaica_data = df[df["Country"] == "Jamaica"].copy()
jamaica_data["Growth_%"] = jamaica_data["Visitors_Thousands"].pct_change() * 100
print(jamaica_data[["Year", "Visitors_Thousands", "Growth_%"]])


# =============================================================================
# PART 7: PRACTICAL ANALYSIS — Caribbean Tourism Insights
# =============================================================================

print("\n" + "=" * 65)
print("PART 7: PRACTICAL ANALYSIS — Key Insights!")
print("=" * 65)

# COVID impact: compare 2020 to 2023
print("--- COVID Recovery: 2020 vs 2023 ---")
for country in df["Country"].unique():
    v_2020 = df[(df["Country"] == country) & (df["Year"] == 2020)]["Visitors_Thousands"].values[0]
    v_2023 = df[(df["Country"] == country) & (df["Year"] == 2023)]["Visitors_Thousands"].values[0]
    change = ((v_2023 - v_2020) / v_2020) * 100
    emoji = "📈" if change > 0 else "📉"
    print(f"  {emoji} {country}: {change:+.1f}% change")

# Which sector is most popular?
print(f"\n--- Most Popular Tourism Sector (2023) ---")
sector_2023 = data_2023.groupby("Top_Sector")["Visitors_Thousands"].sum()
print(sector_2023.sort_values(ascending=False))

# Total Caribbean tourism revenue
total_revenue_2023 = data_2023["Revenue_Millions_USD"].sum()
print(f"\n--- Total Caribbean Tourism Revenue (2023): ${total_revenue_2023:,}M USD ---")
print(f"That's ${total_revenue_2023/1000:.1f} BILLION US Dollars! Tourism is HUGE fi di Caribbean! 🏨")


print("\n" + "=" * 65)
print("🎉 Nice work! Yuh just did REAL data analysis wid pandas!")
print("Dis is exactly wha data scientists do every day.")
print("Try di quiz questions below, then move to Lesson 3!")
print("=" * 65)


# =============================================================================
# QUIZ TIME! 🧠
# =============================================================================
# Check quiz_answers.md when yuh done!
#
# Q1: What is a DataFrame?
#     a) A picture frame  b) A table of data wid rows and columns
#     c) A type of loop  d) A Python variable
#
# Q2: What does df.head(3) do?
#     a) Shows the first 3 columns  b) Shows the last 3 rows
#     c) Shows the first 3 rows  d) Deletes 3 rows
#
# Q3: How do yuh filter a DataFrame fi only show Jamaica data?
#     a) df.filter("Jamaica")  b) df[df["Country"] == "Jamaica"]
#     c) df.jamaica()  d) df.get("Jamaica")
#
# Q4: What does df.shape return?
#     a) The size of each column  b) A drawing of the data
#     c) A tuple with (rows, columns)  d) The first row
#
# Q5: What does df.groupby("Country")["Visitors"].sum() do?
#     a) Counts the countries  b) Groups data by country and adds up visitors
#     c) Sorts countries  d) Filters visitors
#
# Q6: Which Caribbean country had the MOST tourists in 2023 (from our data)?
#     (Look at di output above fi find di answer!)
#
# Q7: Name TWO Caribbean sectors (besides tourism) where pandas would be
#     useful fi analyzing data. What kind of data would they analyze?
#     (Think: agriculture, fishing, healthcare, sports, energy)
#
# Q8: If yuh wanted to find all countries where tourism DECREASED from
#     2020 to 2023, which pandas operations would yuh use?
#     a) filter and sort  b) groupby and filter
#     c) filter, calculate change, filter negative  d) just use head()
# =============================================================================
