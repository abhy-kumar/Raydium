"""Data models, constants, solar park database, and regional definitions for Raydium."""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple


@dataclass
class SolarPoint:
    """Represents solar irradiance and site parameters at a geographical point."""
    latitude: float
    longitude: float
    ghi_daily: float              # Global Horizontal Irradiance (kWh/m²/day)
    dni_daily: float = 0.0        # Direct Normal Irradiance (kWh/m²/day)
    ghi_annual: float = 0.0       # kWh/m²/year
    temp_ambient: float = 25.0    # Mean ambient temperature (°C)
    suitability_score: float = 0.0  # 0 to 100
    suitability_tier: str = "Unclassified"
    region: str = "All India"
    extra: Dict[str, float] = field(default_factory=dict)

    def to_dict(self) -> Dict:
        return {
            "latitude": round(self.latitude, 4),
            "longitude": round(self.longitude, 4),
            "potential": round(self.ghi_daily, 2),  # Backward compatibility
            "ghi_daily": round(self.ghi_daily, 3),
            "ghi_annual": round(self.ghi_annual if self.ghi_annual > 0 else self.ghi_daily * 365.0, 1),
            "dni_daily": round(self.dni_daily, 3),
            "temp_ambient": round(self.temp_ambient, 1),
            "suitability_score": round(self.suitability_score, 1),
            "suitability_tier": self.suitability_tier,
            "region": self.region,
        }


@dataclass
class SolarPark:
    """Information about an existing or planned mega solar park in India."""
    name: str
    state: str
    capacity_mw: float
    latitude: float
    longitude: float
    status: str
    area_acres: float = 0.0
    commissioned_year: Optional[int] = None
    developer: str = ""
    description: str = ""


# Major operational and under-development mega solar parks across India
MEGA_SOLAR_PARKS: List[SolarPark] = [
    SolarPark(
        name="Bhadla Solar Park",
        state="Rajasthan",
        capacity_mw=2245.0,
        latitude=27.5386,
        longitude=71.9167,
        status="Operational",
        area_acres=14000.0,
        commissioned_year=2020,
        developer="RSDCL / NTPC / SoftBank / Adani",
        description="One of the largest solar parks in the world located in the Thar Desert with over 320 sunny days/year."
    ),
    SolarPark(
        name="Pavagada Solar Park (Shakti Sthala)",
        state="Karnataka",
        capacity_mw=2050.0,
        latitude=14.1689,
        longitude=77.2725,
        status="Operational",
        area_acres=13000.0,
        commissioned_year=2019,
        developer="KREDL / SECI / Fortum / Tata Power",
        description="Located in drought-prone Tumkur district, built on barren leased agricultural land."
    ),
    SolarPark(
        name="Kurnool Ultra Mega Solar Park",
        state="Andhra Pradesh",
        capacity_mw=1000.0,
        latitude=15.6811,
        longitude=78.2831,
        status="Operational",
        area_acres=5932.0,
        commissioned_year=2017,
        developer="APSPCL / NTPC / SunEdison / Greenko",
        description="Pioneering mega park in Andhra Pradesh generating ~2,600 GWh annually."
    ),
    SolarPark(
        name="Rewa Ultra Mega Solar",
        state="Madhya Pradesh",
        capacity_mw=750.0,
        latitude=24.4783,
        longitude=81.5744,
        status="Operational",
        area_acres=3928.0,
        commissioned_year=2018,
        developer="RUMSL / Mahindra / ACME / Solenergi",
        description="Supplies clean energy to Delhi Metro (DMRC) meeting ~60% of its daytime power demand."
    ),
    SolarPark(
        name="Khavda Hybrid Renewable Energy Park",
        state="Gujarat",
        capacity_mw=30000.0,
        latitude=23.8500,
        longitude=69.7500,
        status="Under Construction",
        area_acres=72600.0,
        commissioned_year=2026,
        developer="Adani Green / NTPC / GIPCL",
        description="World's largest hybrid renewable energy park in the Rann of Kutch with 20 GW solar and 10 GW wind."
    ),
    SolarPark(
        name="Dholera Solar Park",
        state="Gujarat",
        capacity_mw=5000.0,
        latitude=22.2500,
        longitude=72.2000,
        status="Under Construction",
        area_acres=27000.0,
        commissioned_year=2025,
        developer="GPCL / SECI / Tata Power / Torrent",
        description="Mega park inside the Dholera Special Investment Region (SIR) along the Gulf of Khambhat."
    ),
    SolarPark(
        name="Charanka Solar Park",
        state="Gujarat",
        capacity_mw=690.0,
        latitude=23.9056,
        longitude=71.2056,
        status="Operational",
        area_acres=5384.0,
        commissioned_year=2012,
        developer="GPCL",
        description="India's first major solar park landmark in Patan district."
    ),
    SolarPark(
        name="Ananthapuramu Ultra Mega Solar",
        state="Andhra Pradesh",
        capacity_mw=1500.0,
        latitude=14.9667,
        longitude=77.4500,
        status="Operational",
        area_acres=7726.0,
        commissioned_year=2019,
        developer="APSPCL / NTPC / Tata Power",
        description="Spans NP Kunta in Anantapur district with excellent southern irradiance."
    ),
    SolarPark(
        name="Kamuthi Solar Power Project",
        state="Tamil Nadu",
        capacity_mw=648.0,
        latitude=9.3556,
        longitude=78.3889,
        status="Operational",
        area_acres=2500.0,
        commissioned_year=2016,
        developer="Adani Power",
        description="Single-location solar plant in Ramanathapuram district built in a record 8 months."
    ),
    SolarPark(
        name="Leh & Kargil Ultra Mega Solar",
        state="Ladakh",
        capacity_mw=10000.0,
        latitude=34.1526,
        longitude=77.5771,
        status="Planned",
        area_acres=50000.0,
        commissioned_year=2028,
        developer="SECI / Ministry of Power",
        description="High-altitude cold desert mega project with ultra-high clear-sky solar irradiance."
    ),
    SolarPark(
        name="Mandsaur Solar Park",
        state="Madhya Pradesh",
        capacity_mw=250.0,
        latitude=24.0700,
        longitude=75.0700,
        status="Operational",
        area_acres=1300.0,
        commissioned_year=2017,
        developer="NTPC",
        description="Utility scale installation in Western Madhya Pradesh."
    ),
    SolarPark(
        name="Kadapa Ultra Mega Solar Park",
        state="Andhra Pradesh",
        capacity_mw=1000.0,
        latitude=14.8833,
        longitude=78.3333,
        status="Operational",
        area_acres=5927.0,
        commissioned_year=2020,
        developer="APSPCL / SECI",
        description="Part of Andhra Pradesh's mega solar initiative in Galiveedu mandal."
    )
]


# Regional bounding boxes for filtering & analysis
REGIONAL_BOUNDS: Dict[str, Tuple[float, float, float, float]] = {
    "all": (68.0, 6.5, 97.5, 37.5),
    "north": (73.0, 26.0, 81.0, 37.5),
    "south": (74.0, 8.0, 80.5, 20.0),
    "west": (68.0, 18.0, 77.5, 28.5),
    "east": (81.0, 18.0, 89.5, 28.0),
    "central": (74.0, 20.0, 84.0, 27.0),
    "northeast": (88.0, 21.5, 97.5, 29.5),
    "rajasthan_thar": (69.5, 24.5, 76.0, 29.5),
    "gujarat_kutch": (68.5, 21.0, 73.5, 24.8),
    "ladakh": (75.5, 32.0, 80.0, 36.0),
}


# Solar plant suitability tiers
SUITABILITY_TIERS = {
    "Tier 1 - Prime Location": {
        "min_score": 85.0,
        "color": "#10b981",  # Emerald Green
        "description": "World-class solar irradiance (>5.8 kWh/m²/day), flat topography, and optimal thermal rating. Ideal for GW-scale utility parks."
    },
    "Tier 2 - Highly Suitable": {
        "min_score": 70.0,
        "color": "#3b82f6",  # Blue
        "description": "High solar resource (5.0 - 5.8 kWh/m²/day) and favorable terrain. Excellent for utility and commercial solar farms."
    },
    "Tier 3 - Moderately Suitable": {
        "min_score": 50.0,
        "color": "#f59e0b",  # Amber
        "description": "Moderate solar resource (4.2 - 5.0 kWh/m²/day). Highly viable for rooftop, agrivoltaics, and distributed solar."
    },
    "Tier 4 - Constrained / Low": {
        "min_score": 0.0,
        "color": "#ef4444",  # Red
        "description": "Sub-optimal solar resource (<4.2 kWh/m²/day), high cloudiness, or steep terrain slope (Himalayas/Ghats)."
    }
}
