"""
=============================================================================
Caribbean AI Academy - Sixth Form Capstone Project
Climate Change Impact Predictor — Data Generator
=============================================================================
Generates synthetic Caribbean climate data for 16 nations.
Patterns are based on real Caribbean meteorological trends.

Run this FIRST before running main.py.

Author: Adrian Dunkley | Caribbean AI Academy
=============================================================================
"""

import numpy as np
import json
import os

# Caribbean nations and their characteristics
CARIBBEAN_NATIONS = {
    'Jamaica':             {'lat': 18.1, 'elevation': 'mixed', 'size': 'medium'},
    'Trinidad & Tobago':   {'lat': 10.5, 'elevation': 'low',   'size': 'medium'},
    'Barbados':            {'lat': 13.2, 'elevation': 'low',   'size': 'small'},
    'Bahamas':             {'lat': 25.0, 'elevation': 'vlow',  'size': 'large'},
    'Guyana':              {'lat': 6.8,  'elevation': 'low',   'size': 'large'},
    'Suriname':            {'lat': 5.8,  'elevation': 'low',   'size': 'large'},
    'Belize':              {'lat': 17.2, 'elevation': 'low',   'size': 'medium'},
    'Haiti':               {'lat': 19.0, 'elevation': 'mixed', 'size': 'medium'},
    'Grenada':             {'lat': 12.1, 'elevation': 'mixed', 'size': 'small'},
    'St Lucia':            {'lat': 13.9, 'elevation': 'mixed', 'size': 'small'},
    'Dominica':            {'lat': 15.4, 'elevation': 'high',  'size': 'small'},
    'Antigua & Barbuda':   {'lat': 17.1, 'elevation': 'low',   'size': 'small'},
    'St Kitts & Nevis':    {'lat': 17.3, 'elevation': 'mixed', 'size': 'small'},
    'St Vincent':          {'lat': 13.2, 'elevation': 'mixed', 'size': 'small'},
    'Cuba':                {'lat': 21.5, 'elevation': 'mixed', 'size': 'large'},
    'Dominican Republic':  {'lat': 18.7, 'elevation': 'mixed', 'size': 'large'},
}

ELEVATION_FACTOR = {'vlow': 1.5, 'low': 1.2, 'mixed': 1.0, 'high': 0.7}


def generate_climate_data(years_range=(1970, 2025)):
    """
    Generate yearly climate data for each Caribbean nation.

    Features per year:
    - avg_temperature (C): Annual average surface temperature
    - sea_level_rise (mm): Cumulative sea level rise from baseline
    - annual_rainfall (mm): Total annual rainfall
    - hurricane_intensity: Average intensity index (1-5 scale)
    - coral_bleaching_pct: Percentage of reefs affected
    - drought_months: Number of drought months in the year

    Target variables (for prediction):
    - flood_risk_index: 0-100 scale
    - agricultural_impact: 0-100 (higher = more damage)
    - tourism_impact: 0-100 (higher = more disruption)
    """
    np.random.seed(42)
    all_data = {}
    start_year, end_year = years_range
    years = list(range(start_year, end_year + 1))

    for country, info in CARIBBEAN_NATIONS.items():
        country_data = []
        elev_f = ELEVATION_FACTOR[info['elevation']]

        for i, year in enumerate(years):
            decade = (year - start_year) / 10.0

            # Temperature: rising trend + variability
            base_temp = 26.5 + (info['lat'] - 15) * 0.1
            temp_trend = 0.018 * (year - start_year)  # ~0.18C per decade
            temp = base_temp + temp_trend + np.random.normal(0, 0.3)

            # Sea level rise: accelerating trend
            slr = 2.0 * decade + 0.15 * decade**2 + np.random.normal(0, 1.5)
            slr *= elev_f
            slr = max(slr, 0)

            # Rainfall: changing patterns — more extremes
            base_rain = 1500 + info['lat'] * 20
            rain_trend = -5 * decade  # Slight drying in some areas
            rain_var = 80 * (1 + 0.05 * decade)  # Increasing variability
            rainfall = base_rain + rain_trend + np.random.normal(0, rain_var)
            rainfall = max(rainfall, 400)

            # Hurricane intensity: increasing trend
            hurr_base = 2.0 + 0.08 * decade
            hurr_intensity = hurr_base + np.random.normal(0, 0.5)
            hurr_intensity = np.clip(hurr_intensity, 0.5, 5.0)

            # Coral bleaching: worsening with temperature
            bleach_base = 5 + 8 * decade + 3 * max(temp - 27.5, 0)
            bleach = bleach_base + np.random.normal(0, 5)
            bleach = np.clip(bleach, 0, 100)

            # Drought months
            drought = int(np.clip(np.random.poisson(1.5 + 0.3 * decade), 0, 12))

            # --- Target variables ---
            flood_risk = (slr * 0.8 + max(rainfall - 1500, 0) * 0.02 +
                          hurr_intensity * 8)
            flood_risk += np.random.normal(0, 5)
            flood_risk = np.clip(flood_risk, 0, 100)

            agri_impact = (max(temp - 27, 0) * 15 + drought * 5 +
                           hurr_intensity * 6)
            agri_impact += np.random.normal(0, 4)
            agri_impact = np.clip(agri_impact, 0, 100)

            tourism_impact = (hurr_intensity * 7 + bleach * 0.3 +
                              max(temp - 28, 0) * 8)
            tourism_impact += np.random.normal(0, 5)
            tourism_impact = np.clip(tourism_impact, 0, 100)

            country_data.append({
                'year': year,
                'avg_temperature': round(temp, 2),
                'sea_level_rise_mm': round(slr, 2),
                'annual_rainfall_mm': round(rainfall, 1),
                'hurricane_intensity': round(hurr_intensity, 2),
                'coral_bleaching_pct': round(bleach, 1),
                'drought_months': drought,
                'flood_risk_index': round(flood_risk, 1),
                'agricultural_impact': round(agri_impact, 1),
                'tourism_impact': round(tourism_impact, 1),
            })

        all_data[country] = country_data

    return all_data, years


def save_data(data, filepath='caribbean_climate_data.json'):
    """Save generated data to JSON file."""
    script_dir = os.path.dirname(os.path.abspath(__file__))
    full_path = os.path.join(script_dir, filepath)
    with open(full_path, 'w') as f:
        json.dump(data, f, indent=2)
    print(f"Data saved to {full_path}")
    return full_path


if __name__ == '__main__':
    print("=" * 60)
    print("  Caribbean Climate Data Generator")
    print("=" * 60)

    data, years = generate_climate_data()

    print(f"\nGenerated data for {len(data)} Caribbean nations")
    print(f"Years covered: {years[0]} to {years[-1]} ({len(years)} years)")
    print(f"Records per nation: {len(years)}")
    print(f"Total records: {len(data) * len(years)}")

    print(f"\nNations included:")
    for nation in data.keys():
        print(f"  - {nation}")

    filepath = save_data(data)

    # Print sample
    print(f"\nSample data (Jamaica, last 3 years):")
    for record in data['Jamaica'][-3:]:
        print(f"  {record['year']}: Temp={record['avg_temperature']}C, "
              f"SLR={record['sea_level_rise_mm']}mm, "
              f"FloodRisk={record['flood_risk_index']}")

    print("\nData generation complete! Now run main.py")
