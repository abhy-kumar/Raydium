# 🌞 Raydium - Ideal Solar Plant Locations & Solar Potential Platform for India

[![Raydium CI](https://github.com/abhy-kumar/Raydium/actions/workflows/ci.yml/badge.svg)](https://github.com/abhy-kumar/Raydium/actions/workflows/ci.yml)
[![Daily Solar Intelligence](https://github.com/abhy-kumar/Raydium/actions/workflows/run_raydium.yml/badge.svg)](https://github.com/abhy-kumar/Raydium/actions/workflows/run_raydium.yml)
[![Python 3.9+](https://img.shields.io/badge/python-3.9%20%7C%203.10%20%7C%203.11%20%7C%203.12-blue.svg)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

**Raydium** is an open-source, scientific geospatial platform engineered to map, evaluate, and identify **Ideal Solar Plant Locations across India**.

Combining open satellite insolation data from the **NASA POWER API**, temperature efficiency derating, topographic slope analysis, land suitability factors, and an interactive GIS dashboard, Raydium provides end-to-end solar intelligence for researchers, clean-energy planners, and solar developers.

---

## 🌟 Key Features

- 🛰️ **Open Data Integration**: Automated extraction of Global Horizontal Irradiance (GHI), Direct Normal Irradiance (DNI), and ambient temperature from the open **NASA POWER API** (free, no sign-up or token required).
- 🧠 **Multi-Criteria Solar Plant Suitability Index (SPSI)**:
  - **Solar Resource Potential (45%)**: GHI & DNI multi-year averages.
  - **Thermal Performance Derating (15%)**: Temperature efficiency penalty ($P_{loss} = \gamma \times (T_{cell} - 25^\circ\text{C})$).
  - **Terrain & Slope Suitability (20%)**: Flat and gentle slope favorability ($<5\%$), penalizing steep mountain ranges (High Himalayas and Western Ghats escarpments).
  - **Land & Solar Zone Rating (20%)**: Arid/semi-arid high-sunshine wasteland favorability (Thar Desert, Rann of Kutch, Deccan drylands).
- ⚡ **Ultra-Fast Vectorized Spatial Processing**: Sub-millisecond containment and grid generation using **Shapely 2.0** C-extensions (`contains_xy`), achieving a **>7,000x speedup** over legacy iterative methods.
- 🗺️ **High-Precision 2D Spatial Interpolation**: Continuous, seamless surface interpolation using **SciPy** (`griddata` cubic/linear with nearest-neighbor boundary fill) and exact **Survey of India** polygon geometry masking (`rasterio`).
- ⚡ **Major Mega Solar Parks Database**: Tracked operational and planned GW-scale parks (*Bhadla (2,245 MW)*, *Pavagada (2,050 MW)*, *Khavda Hybrid (30,000 MW)*, *Kurnool (1,000 MW)*, *Rewa (750 MW)*, *Dholera (5,000 MW)*, etc.).
- 💻 **Interactive Web Dashboard & Siting Calculator**:
  - Ultra-lightweight standalone HTML5/Leaflet app (`index.html`) with glassmorphism UI.
  - **Click-to-Inspect**: Click anywhere in India to inspect local solar insolation ($kWh/m^2/\text{day}$ & $kWh/m^2/\text{year}$), Suitability Tier, and calculate estimated plant capacity (MW), annual generation ($GWh$), Capacity Utilization Factor ($CUF\%$), annual revenue, and $CO_2$ abatement.
- 🎨 **Publication-Quality Cartography**: Generates publication-ready 300/600 DPI static maps (`solar_potential_high_res.png`).
- 🔄 **Automated Daily GitHub Actions Pipeline**: Automatically fetches updated insolation data daily at midnight UTC, recalculates suitability, commits updated datasets, and deploys directly to **GitHub Pages**.

---

## 🏗️ System Architecture

```
                               ┌─────────────────────────────┐
                               │   NASA POWER API (Open)     │
                               │  GHI, DNI, Ambient Temp     │
                               └──────────────┬──────────────┘
                                              │ Async + Rate-Limited
                                              ▼
┌───────────────────────────┐     ┌───────────────────────────┐
│ Survey of India GeoJSON   │────▶│ Vectorized Spatial Grid   │
│ (EPSG:4326 Boundaries)    │     │ (Shapely 2.0 contains_xy) │
└───────────────────────────┘     └───────────┬───────────────┘
                                              │
                                              ▼
                                  ┌───────────────────────────┐
                                  │ Multi-Criteria Siting     │
                                  │ Suitability Engine (SPSI) │
                                  └───────────┬───────────────┘
                                              │
                                              ▼
                                  ┌───────────────────────────┐
                                  │ 2D Continuous Interp.     │
                                  │ (SciPy + Rasterio Mask)   │
                                  └───────────┬───────────────┘
                                              │
                    ┌─────────────────────────┴─────────────────────────┐
                    ▼                                                   ▼
     ┌─────────────────────────────┐                     ┌─────────────────────────────┐
     │ High-Res Publication PNG    │                     │ Interactive Web Dashboard   │
     │ (Matplotlib + Solar Parks)  │                     │ (Leaflet + Plant Sizer)     │
     └─────────────────────────────┘                     └─────────────────────────────┘
```

---

## 🚀 Quickstart & Installation

### 1. Prerequisites
- Python 3.9, 3.10, 3.11, or 3.12
- `virtualenv` or `uv` / standard `venv`

### 2. Create Virtual Environment & Install

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

# Install Raydium in editable mode
pip install -e .
```

---

## 🛠️ CLI Usage Guide

Raydium comes with a rich, modern command-line interface:

### 1. System Info & Major Solar Parks
```bash
raydium info
```

### 2. Run the Complete End-to-End Pipeline
Executes grid generation, data collection, suitability scoring, spatial interpolation, analysis, and visualization in one command:
```bash
# Standard run (0.25° resolution ≈ 25km grid)
raydium pipeline

# High-resolution simulation run (for instant offline demonstration)
raydium pipeline --resolution 0.25 --simulate
```

### 3. Collect Solar Insolation Data
```bash
# Collect nationwide data using NASA POWER Climatology
raydium collect --resolution 0.25 --region all --output india_solar_data.csv

# Collect specific region (e.g. Rajasthan Thar Desert)
raydium collect --region rajasthan_thar --output thar_solar_data.csv
```

### 4. Generate Visualizations
```bash
# Generate high-res static PNG and interactive HTML dashboard
raydium visualize --data india_solar_data.csv --image-out solar_potential_high_res.png --html-out index.html
```

### 5. Solar Resource & Suitability Analytics
```bash
# Print summary analytics report and save to JSON
raydium analyze --data india_solar_data.csv --json-out solar_analysis_report.json
```

### 6. Serve Interactive Dashboard Locally
```bash
raydium serve --port 8000
```
Opens the interactive web application in your browser at `http://localhost:8000/index.html`.

---

## 📊 Solar Plant Siting Suitability Tiers

| Tier | Classification | Suitability Score | Daily GHI | Description | Prime Regions |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **Tier 1** | **Prime Location** | **$\ge 85.0$** | $>5.8\text{ kWh/m}^2/\text{day}$ | World-class irradiance, flat barren terrain, optimal thermal rating. Ideal for GW-scale utility solar parks. | Thar Desert (Jodhpur, Jaisalmer, Bikaner), Rann of Kutch, Ladakh |
| **Tier 2** | **Highly Suitable** | **$70.0 - 84.9$** | $5.0 - 5.8\text{ kWh/m}^2/\text{day}$ | High solar resource and favorable terrain. Excellent for utility and commercial solar installations. | Interior Maharashtra, Karnataka, Andhra Pradesh, MP, Telangana |
| **Tier 3** | **Moderately Suitable** | **$50.0 - 69.9$** | $4.2 - 5.0\text{ kWh/m}^2/\text{day}$ | Moderate resource. Highly viable for rooftop solar, distributed microgrids, and agrivoltaics. | Gangetic Plains, Tamil Nadu, Odisha, Gujarat coast |
| **Tier 4** | **Constrained / Low** | **$< 50.0$** | $<4.2\text{ kWh/m}^2/\text{day}$ | Sub-optimal irradiance, high cloudiness, or steep terrain slope. | High Himalayas, dense Western Ghats, heavy monsoon zones |

---

## 🐍 Python API Example

```python
from raydium.grid import generate_india_grid
from raydium.collector import NASADataCollector
from raydium.suitability import calculate_suitability
from raydium.interpolator import SpatialInterpolator
from raydium.visualizer import MapVisualizer
from raydium.analyzer import SolarAnalyzer
import pandas as pd
import asyncio

async def run():
    # 1. Generate boundary grid
    grid_gdf = generate_india_grid(resolution_deg=0.25)
    coords = list(zip(grid_gdf["latitude"], grid_gdf["longitude"]))

    # 2. Collect solar data
    collector = NASADataCollector()
    records = await collector.collect(coords, simulate=True)
    df = pd.DataFrame(records)

    # 3. Calculate Multi-Criteria Suitability
    df = calculate_suitability(df)

    # 4. Interpolate continuous surface
    interpolator = SpatialInterpolator()
    raster_dict = interpolator.interpolate_surface(df, value_column="suitability_score")

    # 5. Render maps
    visualizer = MapVisualizer()
    visualizer.render_static_map(raster_dict, output_image="solar_potential_high_res.png")
    visualizer.render_interactive_dashboard(df, output_html="index.html")

    # 6. Analyze national solar potential
    summary = SolarAnalyzer.generate_summary_report(df)
    print(f"National Mean Irradiance: {summary['solar_resource']['mean_daily_ghi']} kWh/m²/day")

asyncio.run(run())
```

---

## 🌐 Automated GitHub Actions Workflow

Raydium includes two automated workflows in `.github/workflows/`:
1. **`ci.yml`**: Automated continuous integration running test suites across Linux & Windows on Python 3.10, 3.11, and 3.12.
2. **`run_raydium.yml`**: Daily automated pipeline running at **00:00 UTC (05:30 AM IST)**:
   - Fetches the latest solar insolation and meteorological data from NASA POWER.
   - Computes updated multi-factor solar plant suitability scores.
   - Generates high-res cartography and HTML dashboards.
   - Commits updated datasets and automatically deploys the live dashboard to **GitHub Pages**!

---

## 🧪 Testing

Run the full pytest test suite with coverage:

```bash
pytest tests/ -v
```

---

## 📂 Project Structure

```
.
├── src/
│   └── raydium/
│       ├── __init__.py          # Package initialization & exports
│       ├── __main__.py          # python -m raydium entry point
│       ├── models.py            # Data models, solar park DB & bounds
│       ├── grid.py              # Vectorized Shapely 2.0 grid generator
│       ├── collector.py         # Async NASA POWER client & rate limiter
│       ├── suitability.py       # Multi-Criteria Decision Analysis (SPSI)
│       ├── interpolator.py      # SciPy 2D continuous interpolation & masking
│       ├── analyzer.py          # PV capacity sizing, CUF & carbon analytics
│       ├── visualizer.py        # Matplotlib cartography & Leaflet dashboard
│       └── cli.py               # Typer & Rich CLI interface
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
├── main.py                      # Top-level main runner
├── requirements.txt             # Python dependencies
├── pyproject.toml               # PEP 517 / 621 packaging metadata
├── .gitignore                   # Clean Git ignore rules
└── README.md                    # Project documentation
```

---

## 📋 Open Data Sources & Credits

- **[NASA POWER Project](https://power.larc.nasa.gov/)**: Prediction of Worldwide Energy Resources API (Solar Irradiance & Meteorology).
- **[DataMeet](https://github.com/datameet/maps)**: Official Survey of India (SOI) boundary GeoJSON files.
- **[Ministry of New and Renewable Energy (MNRE)](https://mnre.gov.in/)**: National Solar Mission & Mega Solar Parks data.

---

## 📜 License

MIT License © 2026 [abhy-kumar](https://github.com/abhy-kumar) (abhiks177@gmail.com)
