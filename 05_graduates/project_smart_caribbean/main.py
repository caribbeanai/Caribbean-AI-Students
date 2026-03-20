"""
=============================================================================
 CARIBBEAN AI ACADEMY - GRADUATES CAPSTONE PROJECT
 Smart Caribbean Island AI System
 By Adrian Dunkley
=============================================================================

 Dis project brings together EVERYTHING yuh learned in di Academy.
 Build a smart island management system dat uses AI fi:
 - Energy optimization (solar + grid)
 - Water resource management
 - Tourism demand forecasting
 - Hurricane preparedness
 - Agricultural advisory
 - Transportation optimization

 Each component demonstrates a different AI technique.
 Together, dem show how AI can transform Caribbean island life.

 Usage:
   python main.py
=============================================================================
"""

import numpy as np
from typing import Dict, List, Tuple, Optional
from datetime import datetime

# =============================================================================
# PART 1: SMART ISLAND CONFIGURATION
# =============================================================================

class IslandConfig:
    """Configuration fi a Caribbean island AI system."""

    def __init__(self, name: str = "Caribbean Smart Island",
                 population: int = 50000, area_sq_km: float = 400):
        self.name = name
        self.population = population
        self.area_sq_km = area_sq_km
        self.latitude = 17.5  # Typical Caribbean latitude
        self.longitude = -77.0

        # Infrastructure
        self.solar_capacity_mw = population * 0.002  # 2 kW per capita target
        self.water_reservoir_capacity_ml = population * 0.5  # 500L per capita
        self.hotel_rooms = int(population * 0.05)
        self.hospital_beds = int(population * 0.003)
        self.farm_hectares = area_sq_km * 0.3 * 100  # 30% arable land

    def summary(self):
        print(f"\n{'=' * 50}")
        print(f"  SMART ISLAND: {self.name}")
        print(f"{'=' * 50}")
        print(f"  Population:     {self.population:,}")
        print(f"  Area:           {self.area_sq_km} sq km")
        print(f"  Solar capacity: {self.solar_capacity_mw:.1f} MW")
        print(f"  Water storage:  {self.water_reservoir_capacity_ml:,.0f} ML")
        print(f"  Hotel rooms:    {self.hotel_rooms:,}")
        print(f"  Farm land:      {self.farm_hectares:,.0f} hectares")


# =============================================================================
# PART 2: ENERGY MANAGEMENT AI
# =============================================================================

class EnergyManagementAI:
    """
    AI system fi optimizing island energy (solar + diesel grid).

    TODO: Implement reinforcement learning agent that learns to:
    - Balance solar generation with battery storage
    - Minimize diesel generator usage
    - Handle demand spikes (cruise ship arrivals, events)
    - Prepare for hurricane power outages
    """

    def __init__(self, solar_capacity_mw: float, population: int):
        self.solar_capacity = solar_capacity_mw
        self.battery_capacity_mwh = solar_capacity_mw * 4  # 4 hours storage
        self.battery_level = self.battery_capacity_mwh * 0.5
        self.diesel_capacity_mw = solar_capacity_mw * 0.5
        self.base_demand_mw = population * 0.001  # 1 kW per capita avg

    def predict_solar_generation(self, hour: int, cloud_cover: float) -> float:
        """
        Predict solar generation based on time of day and weather.

        TODO: Replace with trained neural network using historical data.
        """
        # Simple solar curve: peaks at noon
        if 6 <= hour <= 18:
            solar_factor = np.sin(np.pi * (hour - 6) / 12)
        else:
            solar_factor = 0

        # Reduce by cloud cover
        generation = self.solar_capacity * solar_factor * (1 - cloud_cover * 0.8)
        return max(0, generation)

    def predict_demand(self, hour: int, is_cruise_ship_day: bool = False,
                       is_holiday: bool = False) -> float:
        """
        Predict energy demand.

        TODO: Train LSTM/Transformer on historical demand data.
        """
        # Base demand pattern
        demand_curve = {
            0: 0.5, 1: 0.4, 2: 0.4, 3: 0.4, 4: 0.5, 5: 0.6,
            6: 0.8, 7: 0.9, 8: 1.0, 9: 1.0, 10: 1.1, 11: 1.1,
            12: 1.0, 13: 1.0, 14: 1.1, 15: 1.1, 16: 1.0, 17: 0.9,
            18: 1.0, 19: 1.1, 20: 1.0, 21: 0.9, 22: 0.7, 23: 0.6
        }
        demand = self.base_demand_mw * demand_curve.get(hour, 0.7)

        if is_cruise_ship_day:
            demand *= 1.3  # 30% boost from tourists
        if is_holiday:
            demand *= 1.15

        return demand

    def optimize_dispatch(self, hour: int, cloud_cover: float) -> Dict:
        """
        Decide energy dispatch: solar, battery, or diesel.

        TODO: Implement PPO agent for optimal dispatch policy.
        """
        solar_gen = self.predict_solar_generation(hour, cloud_cover)
        demand = self.predict_demand(hour)

        result = {"solar_mw": 0, "battery_mw": 0, "diesel_mw": 0,
                  "battery_charging": 0, "demand_mw": demand}

        # Priority: solar first, then battery, then diesel
        result["solar_mw"] = min(solar_gen, demand)
        remaining = demand - result["solar_mw"]

        # Excess solar charges battery
        excess_solar = solar_gen - result["solar_mw"]
        if excess_solar > 0:
            charge = min(excess_solar, self.battery_capacity_mwh - self.battery_level)
            self.battery_level += charge
            result["battery_charging"] = charge

        # Battery for remaining demand
        if remaining > 0:
            battery_dispatch = min(remaining, self.battery_level)
            result["battery_mw"] = battery_dispatch
            self.battery_level -= battery_dispatch
            remaining -= battery_dispatch

        # Diesel as last resort
        if remaining > 0:
            result["diesel_mw"] = min(remaining, self.diesel_capacity_mw)

        return result


# =============================================================================
# PART 3: WATER RESOURCE AI
# =============================================================================

class WaterResourceAI:
    """
    AI fi managing island water resources.

    TODO: Implement:
    - Rainfall prediction model (time series)
    - Demand forecasting
    - Leak detection (anomaly detection)
    - Optimal distribution scheduling
    """

    def __init__(self, reservoir_capacity_ml: float, population: int):
        self.capacity = reservoir_capacity_ml
        self.current_level = reservoir_capacity_ml * 0.6
        self.daily_per_capita_liters = 200
        self.population = population

    def predict_rainfall(self, month: int) -> float:
        """
        Predict daily rainfall in mm.

        TODO: Train on Caribbean weather station data.
        Caribbean has wet season (Jun-Nov) and dry season (Dec-May).
        """
        # Simplified Caribbean rainfall pattern
        wet_months = {6: 150, 7: 140, 8: 160, 9: 180, 10: 200, 11: 170}
        dry_months = {12: 80, 1: 60, 2: 50, 3: 40, 4: 60, 5: 100}

        monthly_rain = {**wet_months, **dry_months}
        daily_avg = monthly_rain.get(month, 100) / 30

        # Add randomness
        return max(0, daily_avg + np.random.normal(0, daily_avg * 0.5))

    def daily_simulation(self, month: int) -> Dict:
        """Simulate one day of water management."""
        rainfall = self.predict_rainfall(month)
        catchment_area_sqm = 500000  # 0.5 sq km catchment
        inflow_liters = rainfall * catchment_area_sqm / 1000  # mm to m3 to L
        inflow_ml = inflow_liters / 1e6

        demand_ml = (self.population * self.daily_per_capita_liters) / 1e6

        self.current_level = min(self.capacity,
                                  self.current_level + inflow_ml - demand_ml)
        self.current_level = max(0, self.current_level)

        level_pct = self.current_level / self.capacity * 100
        alert = "CRITICAL" if level_pct < 20 else "WARNING" if level_pct < 40 else "OK"

        return {
            "rainfall_mm": rainfall,
            "inflow_ml": inflow_ml,
            "demand_ml": demand_ml,
            "level_pct": level_pct,
            "alert": alert
        }


# =============================================================================
# PART 4: TOURISM DEMAND FORECASTING
# =============================================================================

class TourismForecastAI:
    """
    Predict tourism demand fi capacity planning.

    TODO: Implement:
    - Seasonal demand prediction (ARIMA or Prophet-style)
    - Cruise ship schedule integration
    - Event impact modeling (Carnival, Cricket, Festivals)
    - Dynamic pricing recommendations
    """

    # Caribbean tourism seasonality
    SEASONALITY = {
        1: 1.3, 2: 1.4, 3: 1.3, 4: 1.1,    # High season
        5: 0.8, 6: 0.7, 7: 0.9, 8: 0.8,     # Low season
        9: 0.6, 10: 0.7, 11: 0.9, 12: 1.2    # Shoulder + holiday
    }

    def __init__(self, hotel_rooms: int):
        self.hotel_rooms = hotel_rooms
        self.avg_occupancy = 0.65

    def forecast_arrivals(self, month: int, year: int,
                          special_events: List[str] = None) -> Dict:
        """Forecast monthly tourist arrivals."""
        base = self.hotel_rooms * 30 * self.avg_occupancy  # Room-nights
        seasonal = self.SEASONALITY.get(month, 1.0)

        # Event boost
        event_boost = 1.0
        if special_events:
            if "carnival" in special_events:
                event_boost += 0.4
            if "cricket" in special_events:
                event_boost += 0.2
            if "music_festival" in special_events:
                event_boost += 0.15

        arrivals = base * seasonal * event_boost
        revenue_per_tourist = 150  # USD per night average
        total_revenue = arrivals * revenue_per_tourist

        return {
            "month": month,
            "predicted_arrivals": int(arrivals),
            "occupancy_rate": min(0.95, self.avg_occupancy * seasonal * event_boost),
            "predicted_revenue_usd": total_revenue,
            "recommended_price_adjustment": (seasonal - 1) * 50,
        }


# =============================================================================
# PART 5: HURRICANE PREPAREDNESS AI
# =============================================================================

class HurricanePreparednessAI:
    """
    AI system fi hurricane preparedness and response.

    TODO: Implement:
    - Tropical wave tracking and intensification prediction
    - Resource pre-positioning optimization
    - Evacuation route planning
    - Post-hurricane damage assessment (satellite imagery)
    """

    HURRICANE_CATEGORIES = {
        1: {"wind_mph": 74, "surge_ft": 4, "damage": "Minimal"},
        2: {"wind_mph": 96, "surge_ft": 6, "damage": "Moderate"},
        3: {"wind_mph": 111, "surge_ft": 9, "damage": "Extensive"},
        4: {"wind_mph": 130, "surge_ft": 13, "damage": "Extreme"},
        5: {"wind_mph": 157, "surge_ft": 18, "damage": "Catastrophic"},
    }

    def __init__(self, population: int, hospital_beds: int):
        self.population = population
        self.hospital_beds = hospital_beds

    def assess_threat(self, distance_km: float, category: int,
                      heading_toward: bool) -> Dict:
        """Assess hurricane threat level."""
        if not heading_toward or distance_km > 1000:
            return {"threat": "LOW", "action": "Monitor"}

        cat_info = self.HURRICANE_CATEGORIES.get(category, self.HURRICANE_CATEGORIES[1])

        if distance_km < 200:
            threat = "CRITICAL"
            action = "EVACUATE coastal areas, activate shelters"
        elif distance_km < 500:
            threat = "HIGH"
            action = "Prepare shelters, stage supplies, alert population"
        else:
            threat = "MODERATE"
            action = "Monitor closely, begin preparations"

        # Resource needs estimation
        evacuees = int(self.population * 0.3 * (category / 5))
        water_needed_liters = evacuees * 3 * 7  # 3L/person/day for 7 days
        food_needed_kg = evacuees * 2 * 7
        shelters_needed = evacuees // 200  # 200 people per shelter

        return {
            "threat": threat,
            "category": category,
            "action": action,
            "estimated_evacuees": evacuees,
            "water_needed_liters": water_needed_liters,
            "food_needed_kg": food_needed_kg,
            "shelters_needed": shelters_needed,
            "hospital_surge_capacity": max(0, evacuees * 0.05 - self.hospital_beds),
        }


# =============================================================================
# PART 6: AGRICULTURAL ADVISORY AI
# =============================================================================

class AgriculturalAdvisoryAI:
    """
    AI advisory system fi Caribbean farmers.

    TODO: Implement:
    - Crop recommendation engine (soil + weather + market)
    - Pest and disease prediction
    - Optimal planting calendar
    - Market price forecasting
    """

    CARIBBEAN_CROPS = {
        "banana":    {"water_need": "high", "months": [1,2,3,4,5,6,7,8,9,10,11,12], "revenue_per_ha": 8000},
        "sugarcane": {"water_need": "high", "months": [1,2,3,10,11,12], "revenue_per_ha": 5000},
        "cocoa":     {"water_need": "medium", "months": [3,4,5,9,10,11], "revenue_per_ha": 6000},
        "coffee":    {"water_need": "medium", "months": [9,10,11,12,1,2], "revenue_per_ha": 12000},
        "citrus":    {"water_need": "medium", "months": [11,12,1,2,3,4], "revenue_per_ha": 7000},
        "yam":       {"water_need": "low", "months": [3,4,5], "revenue_per_ha": 4500},
        "cassava":   {"water_need": "low", "months": [4,5,6], "revenue_per_ha": 3500},
        "hot_pepper":{"water_need": "low", "months": [2,3,4,8,9,10], "revenue_per_ha": 9000},
    }

    def recommend_crops(self, month: int, water_available: str,
                        farm_size_ha: float) -> List[Dict]:
        """Recommend crops based on current conditions."""
        recommendations = []
        for crop, info in self.CARIBBEAN_CROPS.items():
            if month in info["months"]:
                water_match = (
                    info["water_need"] == "low" or
                    (info["water_need"] == "medium" and water_available in ["medium", "high"]) or
                    (info["water_need"] == "high" and water_available == "high")
                )
                if water_match:
                    projected_revenue = info["revenue_per_ha"] * farm_size_ha
                    recommendations.append({
                        "crop": crop,
                        "water_need": info["water_need"],
                        "projected_revenue": projected_revenue,
                    })

        recommendations.sort(key=lambda x: x["projected_revenue"], reverse=True)
        return recommendations[:5]


# =============================================================================
# PART 7: SMART ISLAND DASHBOARD (Simulation)
# =============================================================================

class SmartIslandDashboard:
    """
    Central dashboard dat integrates all AI subsystems.
    """

    def __init__(self, config: IslandConfig):
        self.config = config
        self.energy = EnergyManagementAI(config.solar_capacity_mw, config.population)
        self.water = WaterResourceAI(config.water_reservoir_capacity_ml, config.population)
        self.tourism = TourismForecastAI(config.hotel_rooms)
        self.hurricane = HurricanePreparednessAI(config.population, config.hospital_beds)
        self.agriculture = AgriculturalAdvisoryAI(config.farm_hectares)

    def daily_report(self, month: int, day: int, hour: int = 12):
        """Generate comprehensive daily smart island report."""
        print("\n" + "=" * 60)
        print(f"  SMART ISLAND DAILY REPORT - {self.config.name}")
        print(f"  Date: Month {month}, Day {day} | Hour: {hour}:00")
        print("=" * 60)

        # Energy
        cloud_cover = np.random.uniform(0.1, 0.7)
        energy = self.energy.optimize_dispatch(hour, cloud_cover)
        print(f"\n  ENERGY (Cloud: {cloud_cover:.0%})")
        print(f"    Solar:   {energy['solar_mw']:.1f} MW")
        print(f"    Battery: {energy['battery_mw']:.1f} MW "
              f"(Level: {self.energy.battery_level:.1f}/{self.energy.battery_capacity_mwh:.1f} MWh)")
        print(f"    Diesel:  {energy['diesel_mw']:.1f} MW")
        print(f"    Demand:  {energy['demand_mw']:.1f} MW")

        # Water
        water = self.water.daily_simulation(month)
        print(f"\n  WATER")
        print(f"    Rainfall:  {water['rainfall_mm']:.1f} mm")
        print(f"    Level:     {water['level_pct']:.1f}% [{water['alert']}]")

        # Tourism
        tourism = self.tourism.forecast_arrivals(month, 2026)
        print(f"\n  TOURISM")
        print(f"    Predicted arrivals: {tourism['predicted_arrivals']:,}")
        print(f"    Occupancy:          {tourism['occupancy_rate']:.0%}")
        print(f"    Revenue forecast:   ${tourism['predicted_revenue_usd']:,.0f}")

        # Hurricane (during hurricane season Jun-Nov)
        if 6 <= month <= 11:
            distance = np.random.uniform(200, 2000)
            cat = np.random.choice([0, 1, 2, 3], p=[0.6, 0.2, 0.15, 0.05])
            if cat > 0:
                threat = self.hurricane.assess_threat(distance, cat, distance < 800)
                print(f"\n  HURRICANE ALERT")
                print(f"    Category {cat} storm at {distance:.0f} km")
                print(f"    Threat: {threat['threat']}")
                print(f"    Action: {threat['action']}")

        # Agriculture
        water_avail = "high" if water['level_pct'] > 60 else "medium" if water['level_pct'] > 30 else "low"
        crops = self.agriculture.recommend_crops(month, water_avail, 10)
        print(f"\n  AGRICULTURE (Water: {water_avail})")
        for c in crops[:3]:
            print(f"    {c['crop']:>12}: ${c['projected_revenue']:>8,.0f} projected (10 ha)")

        print("\n" + "=" * 60)


# =============================================================================
# MAIN EXECUTION
# =============================================================================

if __name__ == "__main__":
    print("=" * 60)
    print(" CARIBBEAN AI ACADEMY - CAPSTONE PROJECT")
    print(" Smart Caribbean Island AI System")
    print(" By Adrian Dunkley")
    print("=" * 60)

    # Configure island
    island = IslandConfig(
        name="Emerald Bay, Jamaica",
        population=50000,
        area_sq_km=400
    )
    island.summary()

    # Initialize dashboard
    dashboard = SmartIslandDashboard(island)

    # Run daily reports across different months
    print("\n" + "#" * 60)
    print(" ANNUAL SIMULATION")
    print("#" * 60)

    for month in [1, 4, 7, 10]:  # Quarterly reports
        dashboard.daily_report(month=month, day=15, hour=12)

    # Hurricane scenario
    print("\n" + "#" * 60)
    print(" HURRICANE SCENARIO")
    print("#" * 60)
    threat = dashboard.hurricane.assess_threat(
        distance_km=300, category=4, heading_toward=True
    )
    print(f"\n  Category 4 hurricane at 300 km:")
    for k, v in threat.items():
        if isinstance(v, (int, float)):
            print(f"    {k}: {v:,.0f}")
        else:
            print(f"    {k}: {v}")

    # Energy simulation across a day
    print("\n" + "#" * 60)
    print(" 24-HOUR ENERGY SIMULATION")
    print("#" * 60)
    print(f"\n{'Hour':>6} {'Solar':>8} {'Battery':>8} {'Diesel':>8} {'Demand':>8}")
    print("-" * 42)
    for hour in range(24):
        cloud = 0.3 + 0.2 * np.sin(hour / 24 * np.pi)
        result = dashboard.energy.optimize_dispatch(hour, cloud)
        print(f"{hour:>6} {result['solar_mw']:>7.1f} {result['battery_mw']:>7.1f} "
              f"{result['diesel_mw']:>7.1f} {result['demand_mw']:>7.1f}")

    print("\n" + "=" * 60)
    print(" Smart Caribbean - AI fi a better island life!")
    print(" Dis is just di beginning. Extend each component,")
    print(" add real data, and deploy fi real impact!")
    print("=" * 60)
