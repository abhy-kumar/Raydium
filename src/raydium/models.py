"""Data models, constants, and database for Indian solar plant siting analysis."""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple


@dataclass
class SolarPoint:
    """Represents meteorological and site parameters for a single geographic coordinate."""
    latitude: float
    longitude: float
    ghi_daily: float              # Global Horizontal Irradiance (kWh/m2/day)
    dni_daily: float = 0.0        # Direct Normal Irradiance (kWh/m2/day)
    ghi_annual: float = 0.0       # kWh/m2/year
    temp_ambient: float = 25.0    # Annual mean ambient temperature (°C)
    suitability_score: float = 0.0  # Normalized 0 to 100
    suitability_tier: str = "Unclassified"
    region: str = "All India"
    extra: Dict[str, float] = field(default_factory=dict)

    def to_dict(self) -> Dict:
        return {
            "latitude": round(self.latitude, 4),
            "longitude": round(self.longitude, 4),
            "potential": round(self.ghi_daily, 2),  # Kept for backward compatibility
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
    """Operational or under-construction mega solar project in India."""
    name: str
    state: str
    capacity_mw: float
    latitude: float
    longitude: float
    status: str
    area_acres: float = 0.0
    commissioned_year: Optional[int] = None
    developer: str = ""
    substation: str = ""
    description: str = ""


# Major operational and under-development mega solar parks across India
# Data compiled from MNRE, SECI, and State Nodal Agencies
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
        substation="765/400 kV Bhadla-II PGCIL Pooling Station",
        description="Spans over 14,000 acres in Phalodi/Jodhpur district. Experiences over 325 clear sunny days annually with GHI exceeding 6.2 kWh/m2/day."
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
        substation="400/220 kV Pavagada Pooling Substation",
        description="Constructed across 5 drought-prone villages in Tumkur district using an innovative land-lease model with local farmers."
    ),
    SolarPark(
        name="Khavda Renewable Energy Park",
        state="Gujarat",
        capacity_mw=30000.0,
        latitude=23.8500,
        longitude=69.7500,
        status="Under Construction",
        area_acres=72600.0,
        commissioned_year=2026,
        developer="Adani Green / NTPC / GIPCL",
        substation="765 kV Khavda Pooling Station (ISTS)",
        description="Hybrid solar-wind park in the salt desert of Rann of Kutch. Slated to become the largest single power generation facility on Earth."
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
        substation="400/220 kV Ghani Substation",
        description="Built in Sakunala village of Kurnool district on arid rocky terrain, generating ~2,600 GWh per year."
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
        developer="RUMSL / Mahindra Susten / ACME / Solenergi",
        substation="400/220 kV PGCIL Substation Rewa",
        description="First project in India to supply institutional open-access power directly to Delhi Metro Rail Corporation (DMRC)."
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
        substation="400 kV Dholera SIR Substation",
        description="Situated within the Dholera Special Investment Region along the Gulf of Khambhat with flat coastal topography."
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
        substation="400/220 kV NP Kunta Substation",
        description="Located in NP Kunta, Anantapur district, taking advantage of Southern India's high annual global horizontal insolation."
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
        substation="400/220 kV Charanka Substation",
        description="India's pioneering utility-scale solar park built on wasteland in Patan district under the Gujarat Solar Policy 2009."
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
        substation="400 kV Kamuthi Substation",
        description="Constructed in Ramanathapuram district in just 8 months, equipped with automated robotic panel cleaning systems."
    ),
    SolarPark(
        name="Leh & Kargil Solar Initiative",
        state="Ladakh",
        capacity_mw=10000.0,
        latitude=34.1526,
        longitude=77.5771,
        status="Planned",
        area_acres=50000.0,
        commissioned_year=2028,
        developer="SECI / Ministry of Power",
        substation="Proposed Pang-Kaithal 765 kV HVDC Corridor",
        description="High-altitude cold desert project benefiting from thin atmosphere, high DNI (>7 kWh/m2), and lower ambient temperature cell efficiency gains."
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
        substation="220 kV Suwasra Substation",
        description="Utility-scale park in western Madhya Pradesh connected directly to the Western Regional Grid."
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
        substation="400/220 kV Galiveedu Substation",
        description="Developed on barren government lands in Galiveedu mandal of Rayalaseema region."
    )
]


# Regional bounding boxes for geospatial querying
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


# Siting Suitability Tiers and engineering classifications
SUITABILITY_TIERS = {
    "Tier 1 - Prime Location": {
        "min_score": 85.0,
        "color": "#10b981",
        "description": "Outstanding irradiance (>5.8 kWh/m2/day), flat arid terrain, and optimal grid connectivity. Primary choice for GW-scale utility parks."
    },
    "Tier 2 - Highly Suitable": {
        "min_score": 70.0,
        "color": "#3b82f6",
        "description": "Strong solar resource (5.0 - 5.8 kWh/m2/day) and low slope. Well-suited for 50-500 MW commercial and utility solar farms."
    },
    "Tier 3 - Moderately Suitable": {
        "min_score": 50.0,
        "color": "#f59e0b",
        "description": "Moderate insolation (4.2 - 5.0 kWh/m2/day). Suitable for rooftop solar, microgrids, canal-top solar, and agrivoltaics."
    },
    "Tier 4 - Constrained / Low": {
        "min_score": 0.0,
        "color": "#ef4444",
        "description": "Sub-optimal resource (<4.2 kWh/m2/day), persistent monsoon cloudiness, or steep topography (Himalayas, Western Ghats)."
    }
}
