#!/usr/bin/env python3
"""
Caribbean AI Academy - Dataset Generator
=========================================
Generating sweet Caribbean data, like ah market vendor selling fresh data by de pound!

Dis script create synthetic but realistic datasets covering:
  - Tourism across all Caribbean nations
  - Weather patterns (sun, rain, an' hurricane season)
  - Agriculture (sugarcane, cocoa, banana, nutmeg, coffee)
  - Economics (GDP, employment, sector breakdowns)
  - Sports (West Indies cricket & Caribbean track and field glory)

Run it: python generate_datasets.py
An' watch de data flow like a steelpan melody!
"""

import os
import random
import csv
from datetime import datetime

# Seed for reproducibility - like planting ah coconut tree, same seed same tree
random.seed(42)

# ============================================================
# All Caribbean Countries & Territories
# From Anguilla to Trinidad, we reppin' EVERY island!
# ============================================================
CARIBBEAN_COUNTRIES = [
    "Anguilla", "Antigua and Barbuda", "Aruba", "Bahamas", "Barbados",
    "Bermuda", "Bonaire", "British Virgin Islands", "Cayman Islands",
    "Cuba", "Curacao", "Dominica", "Dominican Republic", "Grenada",
    "Guadeloupe", "Haiti", "Jamaica", "Martinique", "Montserrat",
    "Puerto Rico", "Saba", "Saint Barthelemy", "Saint Kitts and Nevis",
    "Saint Lucia", "Saint Martin", "Saint Vincent and the Grenadines",
    "Sint Eustatius", "Sint Maarten", "Suriname", "Trinidad and Tobago",
    "Turks and Caicos Islands", "US Virgin Islands"
]

# Population-based scale factors (relative tourism appeal / size)
# Bigger islands get more visitors, small ones get fewer but still sweet
COUNTRY_SCALE = {
    "Anguilla": 0.05, "Antigua and Barbuda": 0.15, "Aruba": 0.20,
    "Bahamas": 0.50, "Barbados": 0.25, "Bermuda": 0.15,
    "Bonaire": 0.05, "British Virgin Islands": 0.08, "Cayman Islands": 0.18,
    "Cuba": 0.80, "Curacao": 0.12, "Dominica": 0.06,
    "Dominican Republic": 1.00, "Grenada": 0.08, "Guadeloupe": 0.15,
    "Haiti": 0.10, "Jamaica": 0.65, "Martinique": 0.15,
    "Montserrat": 0.02, "Puerto Rico": 0.70, "Saba": 0.01,
    "Saint Barthelemy": 0.06, "Saint Kitts and Nevis": 0.07,
    "Saint Lucia": 0.15, "Saint Martin": 0.12,
    "Saint Vincent and the Grenadines": 0.06, "Sint Eustatius": 0.01,
    "Sint Maarten": 0.15, "Suriname": 0.08,
    "Trinidad and Tobago": 0.20, "Turks and Caicos Islands": 0.12,
    "US Virgin Islands": 0.20,
}

MONTHS = [
    "January", "February", "March", "April", "May", "June",
    "July", "August", "September", "October", "November", "December"
]

YEARS = list(range(2015, 2026))  # 2015-2025 inclusive

# Output directory - right here so
OUTPUT_DIR = os.path.dirname(os.path.abspath(__file__))


def jitter(value, pct=0.15):
    """Add some natural variation - nuttin' in de Caribbean stay exactly de same!"""
    return round(value * random.uniform(1 - pct, 1 + pct), 2)


def covid_factor(year):
    """COVID hit we tourism HARD in 2020, recovering gradual after dat."""
    if year == 2020:
        return 0.25
    elif year == 2021:
        return 0.55
    elif year == 2022:
        return 0.80
    return 1.0


# ============================================================
# 1. TOURISM DATASET
# "Come visit we islands!" - Every Tourism Board
# ============================================================
def generate_tourism():
    """
    Generate tourism data for all Caribbean countries, 2015-2025.
    Columns: country, year, visitor_count, revenue_usd, hotel_occupancy_pct
    """
    print("  Generating Tourism data... pack yuh suitcase!")
    rows = []
    for country in CARIBBEAN_COUNTRIES:
        scale = COUNTRY_SCALE.get(country, 0.10)
        base_visitors = int(scale * 5_000_000)  # base annual visitors
        for year in YEARS:
            # Growth trend ~3% per year from 2015 baseline, COVID dip
            growth = 1.0 + 0.03 * (year - 2015)
            cf = covid_factor(year)
            visitors = int(jitter(base_visitors * growth * cf, 0.12))
            # Revenue: avg $800-1500 per visitor depending on island
            avg_spend = jitter(800 + scale * 700, 0.10)
            revenue = round(visitors * avg_spend, 2)
            # Occupancy: 55-85% normally, drops with COVID
            occupancy = round(min(95, max(20, jitter(65 + scale * 15, 0.08) * cf)), 1)
            rows.append([country, year, visitors, revenue, occupancy])

    path = os.path.join(OUTPUT_DIR, "tourism.csv")
    _write_csv(path, ["country", "year", "visitor_count", "revenue_usd", "hotel_occupancy_pct"], rows)
    return len(rows)


# ============================================================
# 2. WEATHER DATASET
# "If yuh doh like de weather, wait five minutes"
# ============================================================
def generate_weather():
    """
    Monthly weather data by island, 2015-2025.
    Columns: island, year, month, avg_temp_celsius, rainfall_mm, hurricane_category
    """
    print("  Generating Weather data... checking if rain ah fall!")

    # Base temps by month (tropical, 25-32C range)
    base_temps = [26.5, 26.8, 27.2, 27.8, 28.5, 29.0, 29.5, 29.8, 29.5, 28.8, 27.8, 27.0]
    # Base rainfall by month (dry season Jan-Apr, wet season Jun-Nov)
    base_rain = [60, 45, 40, 55, 100, 150, 160, 180, 190, 175, 130, 80]
    # Hurricane season: June-November, peak Aug-Oct
    hurricane_chance = [0, 0, 0, 0, 0, 0.02, 0.05, 0.10, 0.12, 0.08, 0.03, 0]

    rows = []
    for island in CARIBBEAN_COUNTRIES:
        # Slight latitude variation: northern islands slightly cooler
        idx = CARIBBEAN_COUNTRIES.index(island)
        temp_offset = (idx % 7 - 3) * 0.3  # small variation
        rain_offset = (idx % 5 - 2) * 10

        for year in YEARS:
            for m_idx, month in enumerate(MONTHS):
                temp = round(jitter(base_temps[m_idx] + temp_offset, 0.04), 1)
                rain = round(max(5, jitter(base_rain[m_idx] + rain_offset, 0.25)), 1)

                # Hurricane: 0 = none, 1-5 = category
                hurricane = 0
                if random.random() < hurricane_chance[m_idx]:
                    hurricane = random.choices([1, 2, 3, 4, 5],
                                               weights=[40, 25, 20, 10, 5])[0]
                    rain += hurricane * 80  # hurricanes bring PLENTY rain

                rows.append([island, year, month, temp, round(rain, 1), hurricane])

    path = os.path.join(OUTPUT_DIR, "weather.csv")
    _write_csv(path, ["island", "year", "month", "avg_temp_celsius", "rainfall_mm", "hurricane_category"], rows)
    return len(rows)


# ============================================================
# 3. AGRICULTURE DATASET
# "Ground provision an' cash crop keepin' we fed!"
# ============================================================
CROPS = {
    # crop: (base_yield_tonnes, base_price_usd_per_tonne, main_producers)
    "sugarcane": (
        50000, 35,
        ["Cuba", "Dominican Republic", "Jamaica", "Barbados", "Trinidad and Tobago",
         "Guadeloupe", "Saint Kitts and Nevis", "Haiti", "Puerto Rico", "Martinique"]
    ),
    "cocoa": (
        800, 2500,
        ["Trinidad and Tobago", "Grenada", "Dominican Republic", "Haiti", "Jamaica",
         "Cuba", "Saint Lucia", "Dominica", "Suriname", "Saint Vincent and the Grenadines"]
    ),
    "banana": (
        12000, 450,
        ["Saint Lucia", "Dominica", "Saint Vincent and the Grenadines", "Jamaica",
         "Martinique", "Guadeloupe", "Grenada", "Suriname", "Dominican Republic", "Haiti"]
    ),
    "nutmeg": (
        2500, 8000,
        ["Grenada", "Trinidad and Tobago", "Saint Vincent and the Grenadines",
         "Dominica", "Saint Lucia", "Jamaica"]
    ),
    "coffee": (
        1500, 3200,
        ["Jamaica", "Cuba", "Haiti", "Dominican Republic", "Puerto Rico",
         "Trinidad and Tobago", "Guadeloupe", "Martinique"]
    ),
}


def generate_agriculture():
    """
    Agriculture data: crop yields and prices by island, 2015-2025.
    Columns: island, year, crop, yield_tonnes, price_usd_per_tonne
    """
    print("  Generating Agriculture data... time to reap wha' we sow!")
    rows = []
    for crop, (base_yield, base_price, producers) in CROPS.items():
        for island in producers:
            # Major producers get full yield, others get fraction
            producer_rank = producers.index(island)
            yield_scale = max(0.1, 1.0 - producer_rank * 0.1)

            for year in YEARS:
                # Slight annual variation, general growth trend
                year_factor = 1.0 + 0.01 * (year - 2015)
                # Weather can hurt crops
                weather_luck = random.uniform(0.75, 1.15)
                yld = round(jitter(base_yield * yield_scale * year_factor * weather_luck, 0.10), 1)

                # Price fluctuates with global markets
                price_trend = 1.0 + 0.02 * (year - 2015)
                price = round(jitter(base_price * price_trend, 0.12), 2)

                rows.append([island, year, crop, yld, price])

    path = os.path.join(OUTPUT_DIR, "agriculture.csv")
    _write_csv(path, ["island", "year", "crop", "yield_tonnes", "price_usd_per_tonne"], rows)
    return len(rows)


# ============================================================
# 4. ECONOMICS DATASET
# "De economy have to grow, we cyah stay so!"
# ============================================================

# Approximate GDP in millions USD (2020 baseline estimates)
GDP_BASELINE = {
    "Anguilla": 340, "Antigua and Barbuda": 1500, "Aruba": 2900,
    "Bahamas": 12000, "Barbados": 5200, "Bermuda": 7500,
    "Bonaire": 450, "British Virgin Islands": 1100, "Cayman Islands": 5500,
    "Cuba": 100000, "Curacao": 3100, "Dominica": 550,
    "Dominican Republic": 85000, "Grenada": 1200, "Guadeloupe": 9800,
    "Haiti": 14500, "Jamaica": 15500, "Martinique": 9500,
    "Montserrat": 65, "Puerto Rico": 105000, "Saba": 50,
    "Saint Barthelemy": 420, "Saint Kitts and Nevis": 1000,
    "Saint Lucia": 1800, "Saint Martin": 800,
    "Saint Vincent and the Grenadines": 850, "Sint Eustatius": 110,
    "Sint Maarten": 1500, "Suriname": 3800,
    "Trinidad and Tobago": 22000, "Turks and Caicos Islands": 1000,
    "US Virgin Islands": 3850,
}


def generate_economics():
    """
    Economic indicators by country, 2015-2025.
    Columns: country, year, gdp_millions_usd, unemployment_pct,
             tourism_pct_gdp, agriculture_pct_gdp
    """
    print("  Generating Economics data... counting every dollar an' cent!")
    rows = []
    for country in CARIBBEAN_COUNTRIES:
        base_gdp = GDP_BASELINE.get(country, 500)
        scale = COUNTRY_SCALE.get(country, 0.10)

        # Tourism-dependent vs diversified economies
        base_tourism_pct = 20 + scale * 40  # 20-60% of GDP
        base_agri_pct = max(2, 15 - scale * 10)  # smaller islands more agri-dependent

        for year in YEARS:
            # GDP growth ~2-4% per year, COVID shock
            growth = 1.0 + random.uniform(0.02, 0.04) * (year - 2015)
            cf = covid_factor(year)
            # GDP hit proportional to tourism dependence during COVID
            gdp_cf = 1.0 if cf == 1.0 else cf + (1 - cf) * (1 - base_tourism_pct / 100)
            gdp = round(jitter(base_gdp * growth * gdp_cf, 0.05), 2)

            # Unemployment: 5-25% base, spikes during COVID
            base_unemp = 5 + (1 - scale) * 15
            unemp_cf = 1.0 if cf == 1.0 else 1.0 + (1 - cf) * 0.5
            unemployment = round(min(40, max(2, jitter(base_unemp * unemp_cf, 0.10))), 1)

            # Sector shares shift with COVID
            tourism_pct = round(max(5, jitter(base_tourism_pct * cf, 0.08)), 1)
            agri_pct = round(min(35, max(1, jitter(base_agri_pct / max(0.5, cf), 0.08))), 1)

            rows.append([country, year, gdp, unemployment, tourism_pct, agri_pct])

    path = os.path.join(OUTPUT_DIR, "economics.csv")
    _write_csv(path, [
        "country", "year", "gdp_millions_usd", "unemployment_pct",
        "tourism_pct_gdp", "agriculture_pct_gdp"
    ], rows)
    return len(rows)


# ============================================================
# 5. SPORTS DATASET
# "Wi run fast an' wi bat good - is Caribbean ting dat!"
# ============================================================

# West Indies cricket legends and contemporary players
CRICKET_PLAYERS = [
    # (name, country, era, batting_style)
    ("Brian Lara", "Trinidad and Tobago", "legend", "batsman"),
    ("Vivian Richards", "Antigua and Barbuda", "legend", "batsman"),
    ("Curtly Ambrose", "Antigua and Barbuda", "legend", "bowler"),
    ("Courtney Walsh", "Jamaica", "legend", "bowler"),
    ("Malcolm Marshall", "Barbados", "legend", "bowler"),
    ("Clive Lloyd", "Guyana", "legend", "batsman"),
    ("Garfield Sobers", "Barbados", "legend", "allrounder"),
    ("George Headley", "Jamaica", "legend", "batsman"),
    ("Michael Holding", "Jamaica", "legend", "bowler"),
    ("Joel Garner", "Barbados", "legend", "bowler"),
    ("Desmond Haynes", "Barbados", "legend", "batsman"),
    ("Gordon Greenidge", "Barbados", "legend", "batsman"),
    ("Lance Gibbs", "Guyana", "legend", "bowler"),
    ("Rohan Kanhai", "Guyana", "legend", "batsman"),
    ("Andy Roberts", "Antigua and Barbuda", "legend", "bowler"),
    ("Chris Gayle", "Jamaica", "modern", "batsman"),
    ("Shivnarine Chanderpaul", "Guyana", "modern", "batsman"),
    ("Dwayne Bravo", "Trinidad and Tobago", "modern", "allrounder"),
    ("Kieron Pollard", "Trinidad and Tobago", "modern", "allrounder"),
    ("Jason Holder", "Barbados", "modern", "allrounder"),
    ("Shai Hope", "Barbados", "modern", "batsman"),
    ("Nicholas Pooran", "Trinidad and Tobago", "modern", "batsman"),
    ("Shimron Hetmyer", "Guyana", "modern", "batsman"),
    ("Alzarri Joseph", "Antigua and Barbuda", "modern", "bowler"),
    ("Kemar Roach", "Barbados", "modern", "bowler"),
    ("Shannon Gabriel", "Trinidad and Tobago", "modern", "bowler"),
    ("Sunil Narine", "Trinidad and Tobago", "modern", "bowler"),
    ("Andre Russell", "Jamaica", "modern", "allrounder"),
    ("Darren Bravo", "Trinidad and Tobago", "modern", "batsman"),
    ("Kraigg Brathwaite", "Barbados", "modern", "batsman"),
]

# Caribbean track and field athletes
TRACK_ATHLETES = [
    ("Usain Bolt", "Jamaica", "100m", 9.58, 2009),
    ("Usain Bolt", "Jamaica", "200m", 19.19, 2009),
    ("Shelly-Ann Fraser-Pryce", "Jamaica", "100m", 10.60, 2022),
    ("Shelly-Ann Fraser-Pryce", "Jamaica", "200m", 21.79, 2022),
    ("Elaine Thompson-Herah", "Jamaica", "100m", 10.54, 2021),
    ("Elaine Thompson-Herah", "Jamaica", "200m", 21.53, 2021),
    ("Yohan Blake", "Jamaica", "100m", 9.69, 2012),
    ("Yohan Blake", "Jamaica", "200m", 19.26, 2012),
    ("Asafa Powell", "Jamaica", "100m", 9.72, 2008),
    ("Veronica Campbell-Brown", "Jamaica", "200m", 21.74, 2008),
    ("Shericka Jackson", "Jamaica", "200m", 21.45, 2022),
    ("Shericka Jackson", "Jamaica", "100m", 10.65, 2022),
    ("Kirani James", "Grenada", "400m", 43.74, 2012),
    ("Kirani James", "Grenada", "400m", 43.63, 2022),
    ("Machel Cedenio", "Trinidad and Tobago", "400m", 44.01, 2019),
    ("Jereem Richards", "Trinidad and Tobago", "200m", 19.97, 2018),
    ("Deon Lendore", "Trinidad and Tobago", "400m", 44.36, 2019),
    ("Ato Boldon", "Trinidad and Tobago", "100m", 9.86, 1998),
    ("Ato Boldon", "Trinidad and Tobago", "200m", 19.80, 1997),
    ("Hasely Crawford", "Trinidad and Tobago", "100m", 10.07, 1976),
    ("Kim Collins", "Saint Kitts and Nevis", "100m", 9.93, 2003),
    ("Pauline Davis-Thompson", "Bahamas", "200m", 22.27, 2000),
    ("Tonique Williams-Darling", "Bahamas", "400m", 49.07, 2004),
    ("Shaunae Miller-Uibo", "Bahamas", "400m", 48.36, 2022),
    ("Shaunae Miller-Uibo", "Bahamas", "200m", 21.74, 2022),
    ("Steven Gardiner", "Bahamas", "400m", 43.85, 2021),
    ("Obadele Thompson", "Barbados", "100m", 9.87, 1998),
    ("Ryan Brathwaite", "Barbados", "110m hurdles", 13.14, 2009),
    ("Alyssa Chapman", "Haiti", "100m", 11.30, 2021),
    ("Marlena Wesh", "Haiti", "400m", 51.88, 2016),
    ("Felix Sanchez", "Dominican Republic", "400m hurdles", 47.25, 2003),
    ("Luguellin Santos", "Dominican Republic", "400m", 44.11, 2012),
    ("Marileidy Paulino", "Dominican Republic", "400m", 48.76, 2024),
    ("Jasmine Camacho-Quinn", "Puerto Rico", "100m hurdles", 12.26, 2021),
    ("Javier Culson", "Puerto Rico", "400m hurdles", 47.72, 2012),
    ("Aymara Alcelays", "Cuba", "100m hurdles", 12.58, 2016),
    ("Dayron Robles", "Cuba", "110m hurdles", 12.87, 2008),
    ("Javier Sotomayor", "Cuba", "high jump", 2.45, 1993),
    ("Ana Fidelia Quirot", "Cuba", "800m", 115.86, 1997),
    ("Iván Pedroso", "Cuba", "long jump", 8.71, 1995),
]


def generate_sports():
    """
    Sports data: cricket + track and field.
    Cricket columns: player, country, matches, runs, wickets, batting_average
    Track columns: athlete, country, event, time_or_distance, year
    """
    print("  Generating Sports data... HOWZAT and ON YUH MARKS!")

    # --- Cricket ---
    cricket_rows = []
    for name, country, era, style in CRICKET_PLAYERS:
        if era == "legend":
            matches = random.randint(80, 180)
        else:
            matches = random.randint(30, 120)

        if style == "batsman":
            runs = random.randint(3000, 12000)
            wickets = random.randint(0, 30)
            avg = round(runs / max(1, matches * random.uniform(1.5, 2.2)), 2)
        elif style == "bowler":
            runs = random.randint(500, 3500)
            wickets = random.randint(100, 500)
            avg = round(runs / max(1, matches * random.uniform(1.5, 2.2)), 2)
        else:  # allrounder
            runs = random.randint(2000, 8000)
            wickets = random.randint(50, 250)
            avg = round(runs / max(1, matches * random.uniform(1.5, 2.2)), 2)

        cricket_rows.append([name, country, matches, runs, wickets, avg])

    path_cricket = os.path.join(OUTPUT_DIR, "sports_cricket.csv")
    _write_csv(path_cricket, [
        "player", "country", "matches", "runs", "wickets", "batting_average"
    ], cricket_rows)

    # --- Track and Field ---
    track_rows = []
    for athlete, country, event, mark, year in TRACK_ATHLETES:
        track_rows.append([athlete, country, event, mark, year])

    path_track = os.path.join(OUTPUT_DIR, "sports_track_field.csv")
    _write_csv(path_track, [
        "athlete", "country", "event", "time_or_distance", "year"
    ], track_rows)

    return len(cricket_rows), len(track_rows)


# ============================================================
# Utility
# ============================================================
def _write_csv(filepath, headers, rows):
    """Write rows to CSV - clean and simple like coconut water."""
    with open(filepath, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(headers)
        writer.writerows(rows)


# ============================================================
# Main - LET WE GO!
# ============================================================
def main():
    print("=" * 60)
    print("  CARIBBEAN AI ACADEMY - Dataset Generator")
    print("  'Wi bringing de data, fresh like ah morning sea breeze!'")
    print("=" * 60)
    print()

    total = 0

    n = generate_tourism()
    print(f"    -> Tourism: {n:,} rows written to tourism.csv")
    total += n

    n = generate_weather()
    print(f"    -> Weather: {n:,} rows written to weather.csv")
    total += n

    n = generate_agriculture()
    print(f"    -> Agriculture: {n:,} rows written to agriculture.csv")
    total += n

    n = generate_economics()
    print(f"    -> Economics: {n:,} rows written to economics.csv")
    total += n

    n_cricket, n_track = generate_sports()
    print(f"    -> Cricket: {n_cricket:,} rows written to sports_cricket.csv")
    print(f"    -> Track & Field: {n_track:,} rows written to sports_track_field.csv")
    total += n_cricket + n_track

    print()
    print(f"  TOTAL: {total:,} rows of Caribbean data generated!")
    print(f"  All files saved to: {OUTPUT_DIR}")
    print()
    print("  'Data ready! Time to analyze an' learn, yuh hear?'")
    print("=" * 60)


if __name__ == "__main__":
    main()
