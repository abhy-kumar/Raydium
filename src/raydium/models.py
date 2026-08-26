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
    polygon_bounds: List[List[float]] = field(default_factory=list)
    substation_coords: List[float] = field(default_factory=list)


# 15 Ranked Ideal Candidate Zones for NEW Mega Solar Parks in India
CANDIDATE_SOLAR_ZONES: List[CandidateSiteZone] = [
    CandidateSiteZone(
        rank=1,
        name="Jaisalmer-Fatehgarh West Corridor",
        district="Jaisalmer",
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
        nearest_substation="765/400 kV Fatehgarh-II ISTS Pooling Station",
        substation_distance_km=18.0,
        water_cleaning_index="Dry Robotic Cleaning Required (Hyper-Arid)",
        lcoe_estimate_inr=2.35,
        key_advantages="Highest GHI in South Asia (>325 sunny days), zero agricultural conflict, direct access to Green Energy Corridor transmission infrastructure.",
        polygon_bounds=[
            [26.85, 70.80], [27.05, 70.80], [27.05, 71.04], [26.85, 71.04]
        ],
        substation_coords=[26.78, 71.12],
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
        land_type="Flat Unirrigated Sand Scrub Wasteland",
        nearest_substation="765/400 kV Bikaner-II Pooling Station",
        substation_distance_km=22.0,
        water_cleaning_index="Dry Robotic Cleaning Recommended",
        lcoe_estimate_inr=2.38,
        key_advantages="Expansive contiguous government revenue wastelands with high ground albedo and direct evacuation into Northern Regional Grid.",
        polygon_bounds=[
            [28.16, 72.68], [28.34, 72.68], [28.34, 72.92], [28.16, 72.92]
        ],
        substation_coords=[28.05, 73.15],
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
        water_cleaning_index="High Salinity Resistance Anti-Soiling Coated Modules",
        lcoe_estimate_inr=2.40,
        key_advantages="Ultra-flat natural salt terrain with zero human habitation or agricultural displacement, ideal for single-axis tracker arrays.",
        polygon_bounds=[
            [24.00, 69.98], [24.24, 69.98], [24.24, 70.32], [24.00, 70.32]
        ],
        substation_coords=[23.85, 69.75],
    ),
    CandidateSiteZone(
        rank=4,
        name="Phalodi-Bap Arid Siting Corridor",
        district="Phalodi / Jodhpur",
        state="Rajasthan",
        latitude=27.4200,
        longitude=72.2800,
        potential_capacity_mw=4000.0,
        estimated_area_acres=18000.0,
        ghi_daily=6.30,
        dni_daily=6.82,
        suitability_score=96.8,
        terrain_slope_pct=0.7,
        land_type="Hyper-Arid Stony Gravel Plain",
        nearest_substation="765/400 kV Bhadla-II Pooling Station",
        substation_distance_km=16.0,
        water_cleaning_index="Dry Robotic Cleaning",
        lcoe_estimate_inr=2.36,
        key_advantages="Direct corridor adjacent to Bhadla complex with proven grid evacuation infrastructure and high clear-sky index.",
        polygon_bounds=[
            [27.32, 72.15], [27.52, 72.15], [27.52, 72.41], [27.32, 72.41]
        ],
        substation_coords=[27.54, 71.92],
    ),
    CandidateSiteZone(
        rank=5,
        name="Nyoma-Hanle High-Altitude Plateau",
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
        water_cleaning_index="Low Humidity / Dry Air System",
        lcoe_estimate_inr=2.52,
        key_advantages="Exceptional clear-sky DNI due to thin atmosphere; ambient temperatures under 15°C eliminate thermal cell degradation and boost module efficiency by up to +12%.",
        polygon_bounds=[
            [33.08, 78.32], [33.32, 78.32], [33.32, 78.68], [33.08, 78.68]
        ],
        substation_coords=[32.90, 77.80],
    ),
    CandidateSiteZone(
        rank=6,
        name="Banaskantha-Radhanpur Flatlands",
        district="Banaskantha / Patan",
        state="Gujarat",
        latitude=23.8200,
        longitude=71.5800,
        potential_capacity_mw=3500.0,
        estimated_area_acres=15000.0,
        ghi_daily=5.98,
        dni_daily=6.25,
        suitability_score=92.5,
        terrain_slope_pct=0.6,
        land_type="Saline Wasteland & Non-Arable Soil",
        nearest_substation="765/400 kV Banaskantha Substation",
        substation_distance_km=19.0,
        water_cleaning_index="Dry Robotic System",
        lcoe_estimate_inr=2.44,
        key_advantages="Extremely flat topography, proximity to Western Grid load centers, and low dust deposition rates.",
        polygon_bounds=[
            [23.72, 71.45], [23.92, 71.45], [23.92, 71.71], [23.72, 71.71]
        ],
        substation_coords=[23.68, 71.85],
    ),
    CandidateSiteZone(
        rank=7,
        name="Barmer-Sheo Hyper-Arid Basin",
        district="Barmer",
        state="Rajasthan",
        latitude=26.1500,
        longitude=71.2500,
        potential_capacity_mw=3200.0,
        estimated_area_acres=14000.0,
        ghi_daily=6.22,
        dni_daily=6.65,
        suitability_score=95.0,
        terrain_slope_pct=1.1,
        land_type="Desert Sandy Wasteland",
        nearest_substation="400 kV Barmer / Jaisalmer ISTS Line",
        substation_distance_km=21.0,
        water_cleaning_index="Dry Robotic Cleaning Required",
        lcoe_estimate_inr=2.39,
        key_advantages="High annual sun hours (>320 days), low humidity, zero forest or prime agricultural impact.",
        polygon_bounds=[
            [26.05, 71.12], [26.25, 71.12], [26.25, 71.38], [26.05, 71.38]
        ],
        substation_coords=[25.75, 71.40],
    ),
    CandidateSiteZone(
        rank=8,
        name="Kurnool-Orvakal Wasteland Plateau",
        district="Kurnool",
        state="Andhra Pradesh",
        latitude=15.6500,
        longitude=78.2000,
        potential_capacity_mw=3000.0,
        estimated_area_acres=13500.0,
        ghi_daily=5.80,
        dni_daily=6.05,
        suitability_score=90.5,
        terrain_slope_pct=1.5,
        land_type="Non-cultivable Rocky Deccan Wasteland",
        nearest_substation="400/220 kV Kurnool / Orvakal Pooling Substation",
        substation_distance_km=14.0,
        water_cleaning_index="Semi-Dry Robotic / Treated Water",
        lcoe_estimate_inr=2.52,
        key_advantages="High capacity utilization factor (>24% CUF), robust transmission connectivity into Southern Grid, and zero forest diversion.",
        polygon_bounds=[
            [15.55, 78.08], [15.75, 78.08], [15.75, 78.32], [15.55, 78.32]
        ],
        substation_coords=[15.80, 78.05],
    ),
    CandidateSiteZone(
        rank=9,
        name="Kadapa-Mylavaram Solar Basin",
        district="Kadapa",
        state="Andhra Pradesh",
        latitude=14.8500,
        longitude=78.3500,
        potential_capacity_mw=2200.0,
        estimated_area_acres=10000.0,
        ghi_daily=5.72,
        dni_daily=5.90,
        suitability_score=89.0,
        terrain_slope_pct=1.4,
        land_type="Barren Stony Red Soil Plateau",
        nearest_substation="400 kV Kadapa / NP Kunta Substation",
        substation_distance_km=18.0,
        water_cleaning_index="Semi-Dry Robotic Cleaning",
        lcoe_estimate_inr=2.54,
        key_advantages="Consistent year-round Southern insolation with excellent grid evacuation into Bangalore/Chennai power corridors.",
        polygon_bounds=[
            [14.75, 78.22], [14.95, 78.22], [14.95, 78.48], [14.75, 78.48]
        ],
        substation_coords=[14.50, 78.80],
    ),
    CandidateSiteZone(
        rank=10,
        name="Pavagada East Expansion Corridor",
        district="Tumkur",
        state="Karnataka",
        latitude=14.2800,
        longitude=77.3500,
        potential_capacity_mw=2000.0,
        estimated_area_acres=9000.0,
        ghi_daily=5.68,
        dni_daily=5.82,
        suitability_score=88.2,
        terrain_slope_pct=1.3,
        land_type="Semi-Arid Dry Scrubland",
        nearest_substation="400 kV Pavagada Pooling Substation",
        substation_distance_km=12.0,
        water_cleaning_index="Dry Cleaning Recommended",
        lcoe_estimate_inr=2.56,
        key_advantages="Adjacent to established 2,050 MW Shakti Sthala complex with shared transmission corridors and active farmer lease models.",
        polygon_bounds=[
            [14.20, 77.25], [14.36, 77.25], [14.36, 77.45], [14.20, 77.45]
        ],
        substation_coords=[14.17, 77.27],
    ),
    CandidateSiteZone(
        rank=11,
        name="Koppal-Kuknur Sun Corridor",
        district="Koppal / Gadag",
        state="Karnataka",
        latitude=15.4500,
        longitude=76.0500,
        potential_capacity_mw=2500.0,
        estimated_area_acres=11000.0,
        ghi_daily=5.65,
        dni_daily=5.78,
        suitability_score=87.8,
        terrain_slope_pct=1.2,
        land_type="Flat Dry Black Cotton / Scrubland",
        nearest_substation="765/400 kV Koppal-II Pooling Station",
        substation_distance_km=15.0,
        water_cleaning_index="Semi-Dry Robotic Cleaning",
        lcoe_estimate_inr=2.58,
        key_advantages="Hub of Karnataka Green Energy Corridor Phase-II with high transmission margins and flat expanse.",
        polygon_bounds=[
            [15.35, 75.92], [15.55, 75.92], [15.55, 76.18], [15.35, 76.18]
        ],
        substation_coords=[15.35, 76.15],
    ),
    CandidateSiteZone(
        rank=12,
        name="Dholera SIR Coastal Solar Plains",
        district="Ahmedabad",
        state="Gujarat",
        latitude=22.2500,
        longitude=72.2000,
        potential_capacity_mw=2500.0,
        estimated_area_acres=11000.0,
        ghi_daily=5.88,
        dni_daily=6.10,
        suitability_score=90.0,
        terrain_slope_pct=0.4,
        land_type="Coastal Saline Wasteland (Gulf of Khambhat)",
        nearest_substation="400 kV Dholera SIR Substation",
        substation_distance_km=10.0,
        water_cleaning_index="Anti-Saline Coated Robotic Wash",
        lcoe_estimate_inr=2.48,
        key_advantages="Dedicated ultra-mega solar zone inside Dholera Special Investment Region with zero land acquisition resistance.",
        polygon_bounds=[
            [22.15, 72.08], [22.35, 72.08], [22.35, 72.32], [22.15, 72.32]
        ],
        substation_coords=[22.24, 72.19],
    ),
    CandidateSiteZone(
        rank=13,
        name="Neemuch-Mandsaur Sun Corridor",
        district="Neemuch / Mandsaur",
        state="Madhya Pradesh",
        latitude=24.3500,
        longitude=75.1000,
        potential_capacity_mw=2000.0,
        estimated_area_acres=9000.0,
        ghi_daily=5.62,
        dni_daily=5.75,
        suitability_score=87.0,
        terrain_slope_pct=1.7,
        land_type="Undulating Barren Rocky Wasteland",
        nearest_substation="400 kV Neemuch / Mandsaur Substation",
        substation_distance_km=17.0,
        water_cleaning_index="Standard Cleaning",
        lcoe_estimate_inr=2.62,
        key_advantages="Strategic central grid location with direct interconnections to Rajasthan and Western industrial grids.",
        polygon_bounds=[
            [24.25, 74.98], [24.45, 74.98], [24.45, 75.22], [24.25, 75.22]
        ],
        substation_coords=[24.46, 74.87],
    ),
    CandidateSiteZone(
        rank=14,
        name="Dhule-Dondaicha Arid Basin",
        district="Dhule",
        state="Maharashtra",
        latitude=21.2000,
        longitude=74.7500,
        potential_capacity_mw=1500.0,
        estimated_area_acres=7000.0,
        ghi_daily=5.58,
        dni_daily=5.70,
        suitability_score=86.5,
        terrain_slope_pct=1.8,
        land_type="Barren Deccan Basalt Ridge",
        nearest_substation="400 kV Dhule / Dondaicha Substation",
        substation_distance_km=15.0,
        water_cleaning_index="Semi-Dry Robotic Cleaning",
        lcoe_estimate_inr=2.64,
        key_advantages="Direct access to 765 kV Western Region power lines feeding Mumbai-Pune industrial corridor.",
        polygon_bounds=[
            [21.10, 74.62], [21.30, 74.62], [21.30, 74.88], [21.10, 74.88]
        ],
        substation_coords=[20.90, 74.78],
    ),
    CandidateSiteZone(
        rank=15,
        name="Ramanathapuram-Kamuthi Coastal Belt",
        district="Ramanathapuram",
        state="Tamil Nadu",
        latitude=9.2500,
        longitude=78.2500,
        potential_capacity_mw=1800.0,
        estimated_area_acres=8000.0,
        ghi_daily=5.52,
        dni_daily=5.60,
        suitability_score=84.5,
        terrain_slope_pct=0.7,
        land_type="Flat Coastal Arid Plain",
        nearest_substation="400 kV Kamuthi Substation",
        substation_distance_km=14.0,
        water_cleaning_index="Automated Robotic Washing (High Salt Resistance)",
        lcoe_estimate_inr=2.66,
        key_advantages="Southern grid stabilization, year-round maritime airflow cooling modules, and proven evacuation infrastructure.",
        polygon_bounds=[
            [9.18, 78.12], [9.35, 78.12], [9.35, 78.38], [9.18, 78.38]
        ],
        substation_coords=[9.35, 78.40],
    ),
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
