"""Data models, constants, solar park database, and ranked candidate siting zones for India."""

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


@dataclass
class CandidateSiteZone:
    """High-precision prospective candidate zone for developing NEW mega solar parks."""
    rank: int
    name: str
    district: str
    state: str
    latitude: float
    longitude: float
    potential_capacity_mw: float
    estimated_area_acres: float
    ghi_daily: float
    dni_daily: float
    suitability_score: float
    terrain_slope_pct: float
    land_type: str
    nearest_substation: str
    substation_distance_km: float
    water_cleaning_index: str
    lcoe_estimate_inr: float
    key_advantages: str


# Ranked Ideal Candidate Zones for NEW Mega Solar Parks in India
CANDIDATE_SOLAR_ZONES: List[CandidateSiteZone] = [
    CandidateSiteZone(
        rank=1,
        name="Jaisalmer-Phalodi Hyper-Arid Corridor",
        district="Jaisalmer / Phalodi",
        state="Rajasthan",
        latitude=26.9500,
        longitude=70.9200,
        potential_capacity_mw=8000.0,
        estimated_area_acres=35000.0,
        ghi_daily=6.35,
        dni_daily=6.90,
        suitability_score=98.5,
        terrain_slope_pct=0.8,
        land_type="Barren Sandy & Rocky Desert Wasteland",
        nearest_substation="765/400 kV Fatehgarh-II / Bikaner-II ISTS",
        substation_distance_km=18.0,
        water_cleaning_index="Dry Robotic Cleaning Required (Hyper-Arid)",
        lcoe_estimate_inr=2.35,
        key_advantages="Highest GHI in South Asia (>325 sunny days), virtually zero agricultural or forest conflict, direct access to Green Energy Corridor transmission lines."
    ),
    CandidateSiteZone(
        rank=2,
        name="Bikaner-Puggal Northern Desert Belt",
        district="Bikaner",
        state="Rajasthan",
        latitude=28.2500,
        longitude=72.8000,
        potential_capacity_mw=5500.0,
        estimated_area_acres=24500.0,
        ghi_daily=6.28,
        dni_daily=6.75,
        suitability_score=97.2,
        terrain_slope_pct=0.9,
        land_type="Flat Unirrigated Sand Scrub",
        nearest_substation="765 kV Bikaner Pooling Station",
        substation_distance_km=22.0,
        water_cleaning_index="Dry Robotic Cleaning Recommended",
        lcoe_estimate_inr=2.38,
        key_advantages="Expansive contiguous government revenue wastelands with high ground albedo and direct connection to Northern Regional Grid."
    ),
    CandidateSiteZone(
        rank=3,
        name="Great Rann of Kutch Northern Flats",
        district="Kutch",
        state="Gujarat",
        latitude=24.1200,
        longitude=70.1500,
        potential_capacity_mw=10000.0,
        estimated_area_acres=44000.0,
        ghi_daily=6.18,
        dni_daily=6.60,
        suitability_score=96.0,
        terrain_slope_pct=0.3,
        land_type="Saline Mudflat & Salt Desert",
        nearest_substation="765 kV Khavda Pooling Station (ISTS)",
        substation_distance_km=25.0,
        water_cleaning_index="High Salinity Resistance Coated Modules Required",
        lcoe_estimate_inr=2.40,
        key_advantages="Ultra-flat natural salt terrain with zero human habitation or agricultural displacement, ideal for single-axis tracker arrays."
    ),
    CandidateSiteZone(
        rank=4,
        name="Nyoma-Hanle High-Altitude Cold Desert",
        district="Leh",
        state="Ladakh",
        latitude=33.2000,
        longitude=78.5000,
        potential_capacity_mw=7500.0,
        estimated_area_acres=32000.0,
        ghi_daily=6.45,
        dni_daily=7.20,
        suitability_score=94.5,
        terrain_slope_pct=2.1,
        land_type="High-Altitude Flat Gravel Plateau (4,200m ASL)",
        nearest_substation="Proposed Pang-Kaithal 765 kV HVDC Terminal",
        substation_distance_km=35.0,
        water_cleaning_index="Low Humidity / Dry Air",
        lcoe_estimate_inr=2.52,
        key_advantages="Exceptional clear-sky DNI due to thin atmosphere; ambient temperatures under 15°C eliminate thermal cell degradation and boost module efficiency by up to +12%."
    ),
    CandidateSiteZone(
        rank=5,
        name="Kurnool-Kadapa Dry Escarpment Ridge",
        district="Kurnool / Kadapa",
        state="Andhra Pradesh",
        latitude=15.2500,
        longitude=78.1000,
        potential_capacity_mw=2500.0,
        estimated_area_acres=11500.0,
        ghi_daily=5.65,
        dni_daily=5.85,
        suitability_score=89.0,
        terrain_slope_pct=1.8,
        land_type="Non-cultivable Rocky Wasteland",
        nearest_substation="400/220 kV Ghani / NP Kunta Substation",
        substation_distance_km=15.0,
        water_cleaning_index="Moderate Water Availability (Groundwater/Treated)",
        lcoe_estimate_inr=2.55,
        key_advantages="Consistent year-round Southern grid insolation with high capacity factor (>23% CUF) and proximity to major industrial load centers in Bangalore and Chennai."
    ),
    CandidateSiteZone(
        rank=6,
        name="Pavagada-Bellary Southern Drylands",
        district="Tumkur / Bellary",
        state="Karnataka",
        latitude=14.6000,
        longitude=76.9500,
        potential_capacity_mw=2200.0,
        estimated_area_acres=10000.0,
        ghi_daily=5.58,
        dni_daily=5.70,
        suitability_score=87.5,
        terrain_slope_pct=1.6,
        land_type="Semi-arid Rocky Scrubland",
        nearest_substation="400 kV Pavagada / Hiriyur Substation",
        substation_distance_km=20.0,
        water_cleaning_index="Dry Cleaning Recommended",
        lcoe_estimate_inr=2.58,
        key_advantages="Proven regional solar track record, strong state policy framework for farmer land leasing, and robust 400 kV transmission evacuation."
    ),
    CandidateSiteZone(
        rank=7,
        name="Solapur-Vijayapura Basalt Plateau",
        district="Solapur / Vijayapura",
        state="Maharashtra / Karnataka",
        latitude=17.3000,
        longitude=75.8000,
        potential_capacity_mw=1800.0,
        estimated_area_acres=8200.0,
        ghi_daily=5.52,
        dni_daily=5.60,
        suitability_score=86.0,
        terrain_slope_pct=2.0,
        land_type="Barren Black-Rock Deccan Plateau",
        nearest_substation="765/400 kV Solapur PGCIL Substation",
        substation_distance_km=14.0,
        water_cleaning_index="Semi-Dry Robotic Cleaning",
        lcoe_estimate_inr=2.60,
        key_advantages="Direct access to 765 kV Western-Southern inter-regional power corridor, high local industrial demand, and low cloud cover."
    ),
    CandidateSiteZone(
        rank=8,
        name="Rewa-Sidhi Wasteland Ridge",
        district="Rewa / Sidhi",
        state="Madhya Pradesh",
        latitude=24.2000,
        longitude=82.1000,
        potential_capacity_mw=1600.0,
        estimated_area_acres=7200.0,
        ghi_daily=5.42,
        dni_daily=5.50,
        suitability_score=84.5,
        terrain_slope_pct=2.3,
        land_type="Reclaimed Barren Stony Plateau",
        nearest_substation="400 kV Vindhyachal / Rewa PGCIL",
        substation_distance_km=19.0,
        water_cleaning_index="Standard Cleaning with Pipeline Access",
        lcoe_estimate_inr=2.65,
        key_advantages="Central location with dual connectivity to Northern and Western regional grids, ideal for C&I open access PPA projects."
    ),
    CandidateSiteZone(
        rank=9,
        name="Surendranagar Semi-Arid Basin",
        district="Surendranagar",
        state="Gujarat",
        latitude=23.3500,
        longitude=71.6000,
        potential_capacity_mw=2000.0,
        estimated_area_acres=9000.0,
        ghi_daily=5.95,
        dni_daily=6.20,
        suitability_score=91.0,
        terrain_slope_pct=0.9,
        land_type="Saline Wasteland & Non-Arable Soil",
        nearest_substation="400 kV Surendranagar Substation",
        substation_distance_km=16.0,
        water_cleaning_index="Dry Robotic System",
        lcoe_estimate_inr=2.45,
        key_advantages="Flat topography, proximity to Dholera Industrial Hub, and high winter solar insolation."
    ),
    CandidateSiteZone(
        rank=10,
        name="Ramanathapuram Coastal Solar Belt",
        district="Ramanathapuram",
        state="Tamil Nadu",
        latitude=9.1500,
        longitude=78.1000,
        potential_capacity_mw=1300.0,
        estimated_area_acres=6000.0,
        ghi_daily=5.48,
        dni_daily=5.55,
        suitability_score=83.0,
        terrain_slope_pct=0.8,
        land_type="Flat Coastal Arid Plain",
        nearest_substation="400 kV Kamuthi Substation",
        substation_distance_km=12.0,
        water_cleaning_index="Automated Robotic Washing (High Salt Resistance)",
        lcoe_estimate_inr=2.68,
        key_advantages="Southern grid stabilization, year-round maritime airflow cooling modules, and existing high-voltage transmission infrastructure."
    )
]


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
