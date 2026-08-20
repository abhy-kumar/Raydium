# Raydium: Solar Plant Siting & Resource Intelligence Platform for India

[![Raydium CI](https://github.com/abhy-kumar/Raydium/actions/workflows/ci.yml/badge.svg)](https://github.com/abhy-kumar/Raydium/actions/workflows/ci.yml)
[![Daily Pipeline](https://github.com/abhy-kumar/Raydium/actions/workflows/run_raydium.yml/badge.svg)](https://github.com/abhy-kumar/Raydium/actions/workflows/run_raydium.yml)
[![Python 3.9+](https://img.shields.io/badge/python-3.9%20%7C%203.10%20%7C%203.11%20%7C%203.12-blue.svg)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

**Raydium** is an open-source geospatial analysis platform designed to identify, evaluate, and rank **ideal utility-scale solar power plant locations across India**.

By combining open satellite meteorology from the **NASA POWER API**, temperature derating physics, topographic slope analysis, land classification, and an interactive GIS dashboard, Raydium provides an end-to-end toolchain for renewable energy researchers, policy planners, and developers.

---

## Technical Overview

Selecting optimal sites for multi-megawatt or gigawatt-scale solar installations requires more than raw solar radiation figures. Raydium implements a **Multi-Criteria Solar Plant Suitability Index (SPSI)** across a fine-resolution spatial grid conforming to India's official Survey of India boundary.

### Siting Evaluation Criteria

| Factor | Weight | Parameter & Source | Physical Rationale |
| :--- | :---: | :--- | :--- |
| **Solar Resource** | **45%** | Global Horizontal Irradiance (GHI) & DNI (NASA POWER) | Primary driver of annual electricity generation ($kWh/m^2/\text{day}$). |
| **Thermal Derating** | **15%** | Ambient Temperature ($T_{2M}$) | Crystalline silicon PV modules experience a power loss of approximately $-0.35\%$ to $-0.45\% / ^\circ\text{C}$ above $25^\circ\text{C}$ STC. High-altitude cold deserts (e.g. Ladakh) receive efficiency bonuses, while extreme summer temperatures in central plains are penalized. |
| **Topography & Slope** | **20%** | Terrain elevation & gradient | Utility-scale solar arrays with single-axis tracking require flat or low-slope terrain ($<5\%$). Steep mountainous terrain (Himalayas, Western Ghats) is heavily constrained. |
| **Land & Climate** | **20%** | Aridity & land classification | Prioritizes arid and semi-arid non-agricultural scrublands (Thar Desert, Rann of Kutch, Deccan drylands) with $>300$ clear-sky days annually. |

---

## Key Features

- **Vectorized Spatial Processing**: Built on Shapely 2.0 C-extensions (`contains_xy`), evaluating nationwide boundary containment across tens of thousands of points in sub-second time.
- **NASA POWER Client**: Asynchronous, rate-limited HTTP client with token-bucket flow control, exponential backoff, and local DiskCache persistence for zero-cost insolation data extraction.
- **Continuous 2D Interpolation**: SciPy-powered continuous 2D surface interpolation (`griddata` cubic/linear with nearest-neighbor fill) masked to the official Survey of India polygon using Rasterio.
- **Mega Solar Parks Database**: Geo-referenced catalog of India's world-record solar parks (*Bhadla (2,245 MW)*, *Pavagada (2,050 MW)*, *Khavda Hybrid (30,000 MW)*, *Kurnool (1,000 MW)*, *Rewa (750 MW)*, *Dholera (5,000 MW)*).
- **Interactive GIS Dashboard**: Lightweight (<450 KB), responsive HTML5/Leaflet application with real-time capacity sizing, annual GWh yield projections, LCOE estimates, and carbon offset calculations.
- **Automated Daily Pipeline**: GitHub Actions workflow scheduled daily at **00:00 UTC (05:30 AM IST)** that re-fetches insolation data, recomputes the suitability index, and automatically deploys the live dashboard to GitHub Pages.

---

## Installation

### Prerequisites
- Python 3.9, 3.10, 3.11, or 3.12
- Standard virtual environment (`venv` or `uv`)

### Setup

```bash
# Clone the repository
git clone https://github.com/abhy-kumar/Raydium.git
cd Raydium

# Create and activate virtual environment
python -m venv .venv
# On Windows:
.venv\Scripts\activate
# On Linux / macOS:
source .venv/bin/activate

# Install in editable mode
pip install -e .
```

---

## Command Line Interface

Raydium provides a CLI interface through the `raydium` command:

```bash
# Display system info and major tracked solar parks
raydium info

# Run the complete end-to-end pipeline (Grid -> Collect -> Suitability -> Interpolate -> Visualize)
raydium pipeline --resolution 0.25

# Rapid offline testing / simulation mode
raydium pipeline --resolution 0.25 --simulate

# Collect solar data for a specific region
raydium collect --region rajasthan_thar --resolution 0.25 --output thar_solar_data.csv

# Generate static 300 DPI publication map and interactive HTML dashboard
raydium visualize --data india_solar_data.csv

# Print summary energy statistics and export report to JSON
raydium analyze --data india_solar_data.csv --json-out solar_report.json

# Launch local dashboard server
raydium serve --port 8000
```

---

## Siting Suitability Tiers

```
  Tier 1: Prime Location (Score >= 85)
  ├── Thar Desert (Jodhpur, Jaisalmer, Bikaner, Phalodi)
  ├── Rann of Kutch & Saurashtra
  └── Ladakh High-Altitude Plateau

  Tier 2: Highly Suitable (Score 70 - 84.9)
  ├── Interior Maharashtra (Deccan Plateau)
  ├── Karnataka (Tumkur, Bellary, Chitradurga)
  ├── Andhra Pradesh & Telangana (Rayalaseema)
  └── Western Madhya Pradesh (Rewa, Mandsaur)

  Tier 3: Moderately Suitable (Score 50 - 69.9)
  ├── Gangetic Plains (UP, Bihar, West Bengal)
  └── Coastal Tamil Nadu & Odisha

  Tier 4: Constrained / Low (Score < 50)
  ├── High Himalayas & steep mountainous zones
  └── Dense Western Ghats & heavy monsoon rainforest corridors
```

---

## Python API Usage

```python
import asyncio
import pandas as pd
from raydium.grid import generate_india_grid
from raydium.collector import NASADataCollector
from raydium.suitability import calculate_suitability
from raydium.interpolator import SpatialInterpolator
from raydium.visualizer import MapVisualizer
from raydium.analyzer import SolarAnalyzer

async def main():
    # 1. Generate boundary-clipped spatial grid
    grid = generate_india_grid(resolution_deg=0.25)
    coords = list(zip(grid["latitude"], grid["longitude"]))

    # 2. Collect NASA POWER solar insolation
    collector = NASADataCollector()
    records = await collector.collect(coords, simulate=True)
    df = pd.DataFrame(records)

    # 3. Calculate multi-criteria suitability scores
    df = calculate_suitability(df)

    # 4. Interpolate continuous surface
    interpolator = SpatialInterpolator()
    raster = interpolator.interpolate_surface(df, value_column="suitability_score")

    # 5. Render publication PNG and interactive HTML dashboard
    visualizer = MapVisualizer()
    visualizer.render_static_map(raster, output_image="solar_potential_high_res.png")
    visualizer.render_interactive_dashboard(df, output_html="index.html")

    # 6. Analyze statistics
    report = SolarAnalyzer.generate_summary_report(df)
    print(f"Mean Daily GHI: {report['solar_resource']['mean_daily_ghi']} kWh/m2/day")

if __name__ == "__main__":
    asyncio.run(main())
```

---

## Test Suite

Run unit tests with pytest:

```bash
pytest tests/ -v
```

---

## Project Structure

```
.
├── src/
│   └── raydium/
│       ├── __init__.py          # Package initialization
│       ├── __main__.py          # python -m raydium entry point
│       ├── models.py            # Data models, solar park DB & bounds
│       ├── grid.py              # Vectorized grid generation
│       ├── collector.py         # Async NASA POWER client & rate limiter
│       ├── suitability.py       # Multi-Criteria Siting Suitability (SPSI)
│       ├── interpolator.py      # SciPy 2D continuous spatial interpolation
│       ├── analyzer.py          # PV sizing, CUF & carbon analytics
│       ├── visualizer.py        # Matplotlib cartography & Leaflet dashboard
│       └── cli.py               # Typer & Rich CLI
├── tests/
│   ├── test_grid.py
│   ├── test_collector.py
│   ├── test_suitability.py
│   ├── test_interpolator.py
│   ├── test_analyzer.py
│   └── test_cli.py
├── .github/
│   └── workflows/
│       ├── ci.yml               # Automated CI matrix
│       └── run_raydium.yml      # Daily automated solar pipeline & Pages deploy
├── india-soi.geojson            # Official Survey of India boundary GeoJSON
├── collect_solar_data.py        # Legacy compatibility wrapper
├── visualize_solar_data.py      # Legacy compatibility wrapper
├── main.py                      # Main entry runner
├── requirements.txt             # Dependency requirements
├── pyproject.toml               # PEP 517 / 621 packaging metadata
├── .gitignore                   # Git ignore specifications
└── README.md                    # Project documentation
```

---

## Data Sources & Citations

- **NASA POWER Project**: [NASA Prediction of Worldwide Energy Resources](https://power.larc.nasa.gov/) (Solar Irradiance & Surface Meteorology).
- **Survey of India / DataMeet**: [Country Boundary GeoJSON](https://github.com/datameet/maps).
- **Ministry of New and Renewable Energy (MNRE)**: [National Solar Mission Reports](https://mnre.gov.in/).
- **Central Electricity Authority (CEA)**: [CO2 Baseline Database for the Indian Power Sector](https://cea.nic.in/).

---

## License

MIT License © 2026 [abhy-kumar](https://github.com/abhy-kumar) (abhiks177@gmail.com)
