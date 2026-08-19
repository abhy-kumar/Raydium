"""Multi-Criteria Decision Analysis (MCDA) for Solar Plant Suitability in India."""

from typing import Dict, List, Union
import numpy as np
import pandas as pd

from raydium.models import SUITABILITY_TIERS


def compute_point_suitability(
    ghi_daily: float,
    temp_ambient: float = 25.0,
    latitude: float = 20.0,
    longitude: float = 78.0,
) -> Dict[str, Union[float, str]]:
    """Compute scientific Solar Plant Suitability Index (SPSI) on a 0-100 scale.

    Criteria weights:
    - Solar Resource (GHI & DNI): 45%
    - Thermal Performance Factor: 15%
    - Topography & Terrain Suitability: 20%
    - Climate & Land Aridity Factor: 20%
    """
    # 1. Solar Resource Score (0 - 100)
    # 6.0+ kWh/m²/day = 100, 3.0 kWh/m²/day = 30
    solar_score = float(np.clip((ghi_daily - 2.5) / (6.2 - 2.5) * 100.0, 10.0, 100.0))

    # 2. Thermal Derating Factor (0 - 100)
    # Silicon PV temperature coefficient ~ -0.4%/°C above 25°C
    # Optimal temp <= 25°C, high penalties above 38°C
    if temp_ambient <= 25.0:
        temp_score = 95.0 + max(0.0, 25.0 - temp_ambient) * 0.5
    else:
        temp_loss_percent = (temp_ambient - 25.0) * 0.4
        temp_score = max(40.0, 100.0 - temp_loss_percent * 4.0)

    # 3. Topography & Slope Suitability (0 - 100)
    # Utility solar requires flat or gentle slopes (< 5%).
    # High Himalayas (lat > 31.5 & lon < 77.0) and Western Ghats (lon 73-76, lat 9-18)
    if latitude > 32.5 and longitude < 76.5:
        # Steep rugged western Himalayas
        terrain_score = 35.0
    elif (73.2 <= longitude <= 75.8) and (9.0 <= latitude <= 17.5):
        # Western Ghats escarpments
        terrain_score = 65.0
    elif (69.0 <= longitude <= 76.0) and (24.0 <= latitude <= 29.5):
        # Thar Desert vast flat plains
        terrain_score = 98.0
    elif (68.5 <= longitude <= 72.5) and (22.5 <= latitude <= 24.5):
        # Rann of Kutch flat salt flats
        terrain_score = 96.0
    elif (75.5 <= longitude <= 79.5) and (32.5 <= latitude <= 36.0):
        # Ladakh high altitude plateau plains
        terrain_score = 80.0
    else:
        # General Deccan / Gangetic plains
        terrain_score = 85.0

    # 4. Climate & Arid Land Availability (0 - 100)
    # Vast barren land & 320+ sunny days in Western / Southern India
    if (69.0 <= longitude <= 76.0) and (24.0 <= latitude <= 29.5):
        land_score = 98.0  # Rajasthan Thar
    elif (68.5 <= longitude <= 73.0) and (21.5 <= latitude <= 24.5):
        land_score = 95.0  # Gujarat Kutch / Saurashtra
    elif (74.5 <= longitude <= 79.5) and (13.0 <= latitude <= 18.0):
        land_score = 90.0  # Karnataka / AP Deccan drylands
    elif longitude > 89.0:
        land_score = 50.0  # High rainfall / cloud cover in NE
    else:
        land_score = 75.0

    # Weighted Composite Score (SPSI)
    composite_score = (
        0.45 * solar_score +
        0.15 * temp_score +
        0.20 * terrain_score +
        0.20 * land_score
    )
    composite_score = round(float(np.clip(composite_score, 0.0, 100.0)), 1)

    # Classify Tier
    tier_name = "Tier 4 - Constrained / Low"
    for tier, info in SUITABILITY_TIERS.items():
        if composite_score >= info["min_score"]:
            tier_name = tier
            break

    return {
        "suitability_score": composite_score,
        "suitability_tier": tier_name,
        "solar_resource_score": round(solar_score, 1),
        "thermal_score": round(temp_score, 1),
        "terrain_score": round(terrain_score, 1),
        "land_score": round(land_score, 1),
    }


def calculate_suitability(df: pd.DataFrame) -> pd.DataFrame:
    """Calculate suitability score and tier for an entire DataFrame of solar points."""
    df = df.copy()

    scores = []
    tiers = []
    solar_scores = []
    thermal_scores = []
    terrain_scores = []
    land_scores = []

    for _, row in df.iterrows():
        lat = float(row.get("latitude", 20.0))
        lon = float(row.get("longitude", 78.0))
        ghi = float(row.get("ghi_daily", row.get("potential", 5.0)))
        temp = float(row.get("temp_ambient", 25.0))

        res = compute_point_suitability(ghi, temp, lat, lon)
        scores.append(res["suitability_score"])
        tiers.append(res["suitability_tier"])
        solar_scores.append(res["solar_resource_score"])
        thermal_scores.append(res["thermal_score"])
        terrain_scores.append(res["terrain_score"])
        land_scores.append(res["land_score"])

    df["suitability_score"] = scores
    df["suitability_tier"] = tiers
    df["solar_resource_score"] = solar_scores
    df["thermal_score"] = thermal_scores
    df["terrain_score"] = terrain_scores
    df["land_score"] = land_scores

    return df
