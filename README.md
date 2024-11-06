# 🌞 Raydium - Solar Potential Analysis for India

Raydium aims to map India's solar potential using high-resolution data and advanced processing techniques. This project is actively being developed, and contributions are highly appreciated! 🚀

## 🌟 Features

- **Solar Potential Mapping**: Fetches and calculates solar potential using NASA POWER API data 🌞
- **Interactive Visualization**: Displays solar potential over India with interactive maps 🖼️
- **High-Resolution Interpolation**: Generates detailed solar potential maps using grid sampling and interpolation 🗺️
- **Optimized Data Fetching**:
  - **Rate Limiting**: Ensures respectful API usage by limiting request rates ⏱️
  - **Caching**: Stores fetched data locally to minimize redundant API calls 💾
  - **Concurrency**: Utilizes asynchronous processing for efficient data handling ⚡
- **Automated with GitHub Actions**: Automatically runs daily or on-demand using GitHub Actions 🕒
- **Comprehensive Logging**: Tracks the process flow and errors with detailed logs 📜

## 🛠️ Setup

### Prerequisites

- **Python**: Version 3.8 or higher
- **Required Packages**: Listed in `requirements.txt`
- **GeoJSON File**: Ensure you have the `india-soi.geojson` file in the root directory. You can obtain it from [DataMeet](https://github.com/datameet/maps/blob/master/Country/india-soi.geojson).

### 1️⃣ Clone the Repository

```bash
git clone https://github.com/your-username/raydium.git
cd raydium
```

### 2️⃣ Install Dependencies

```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
pip install -r requirements.txt
```

Or manually install dependencies with:

```bash
pip install numpy pandas geopandas folium pvlib requests beautifulsoup4 shapely scipy branca matplotlib tqdm```
```
### 3️⃣ Run the Scripts Locally
The project is now split into two separate scripts for better modularity and efficiency:

Data Collection: Fetches and processes solar data.
Visualization: Generates visualizations based on the collected data.

## 🌐 GitHub Actions Workflow

To automate the solar analysis daily or on-demand, Raydium uses GitHub Actions! 🎉

1. **Enable GitHub Actions**:
   - The workflow is defined in `.github/workflows/run_raydium.yml`.
2. **Triggers**:
   - Run daily at midnight UTC (modifiable by editing the `cron`).
   - Or trigger manually from the **Actions** tab.

#### 📝 Workflow Overview

```yaml
name: Generate Solar Potential Data

on:
  schedule:
    - cron: '0 0 * * *'  # Run daily at midnight UTC
  workflow_dispatch:      # Allow manual trigger

jobs:
  collect-data:
    name: Data Collection - ${{ matrix.region }}
    runs-on: ubuntu-latest
    strategy:
      matrix:
        region: [north, south, east, west]  # Define regions based on your GeoJSON files

    steps:
      - name: Checkout Repository
        uses: actions/checkout@v3
        with:
          fetch-depth: 0

      - name: Set Up Python Environment
        uses: actions/setup-python@v4
        with:
          python-version: '3.10'

      - name: Install System Dependencies
        run: |
          sudo apt-get update
          sudo apt-get install -y libgeos-dev libproj-dev

      - name: Install Python Dependencies
        run: |
          python -m pip install --upgrade pip
          pip install -r requirements.txt

      - name: Run Data Collection for ${{ matrix.region }}
        run: |
          python data_collection.py --region ${{ matrix.region }}
        env:
          PYTHONUNBUFFERED: 1

      - name: Upload Solar Data Artifact
        uses: actions/upload-artifact@v3
        with:
          name: solar-data-${{ matrix.region }}
          path: india_solar_data_${{ matrix.region }}.csv

  generate-visualization:
    name: Generate Visualization
    runs-on: ubuntu-latest
    needs: collect-data
    steps:
      - name: Checkout Repository
        uses: actions/checkout@v3
        with:
          fetch-depth: 0

      - name: Download All Solar Data Artifacts
        uses: actions/download-artifact@v3
        with:
          name: solar-data-north
          path: ./data/north
      - uses: actions/download-artifact@v3
        with:
          name: solar-data-south
          path: ./data/south
      - uses: actions/download-artifact@v3
        with:
          name: solar-data-east
          path: ./data/east
      - uses: actions/download-artifact@v3
        with:
          name: solar-data-west
          path: ./data/west

      - name: Combine Solar Data
        run: |
          cat ./data/north/india_solar_data_north.csv ./data/south/india_solar_data_south.csv ./data/east/india_solar_data_east.csv ./data/west/india_solar_data_west.csv > india_solar_data.csv

      - name: Set Up Python Environment
        uses: actions/setup-python@v4
        with:
          python-version: '3.10'

      - name: Install System Dependencies
        run: |
          sudo apt-get update
          sudo apt-get install -y libgeos-dev libproj-dev

      - name: Install Python Dependencies
        run: |
          python -m pip install --upgrade pip
          pip install -r requirements.txt

      - name: Run Visualization
        run: |
          python visualization.py
        env:
          PYTHONUNBUFFERED: 1

      - name: Upload Visualization Artifacts
        uses: actions/upload-artifact@v3
        with:
          name: visualization
          path: |
            india_solar_potential.html
            solar_potential_high_res.png

  commit-data:
    name: Commit and Push Generated Files
    runs-on: ubuntu-latest
    needs: generate-visualization
    steps:
      - name: Checkout Repository
        uses: actions/checkout@v3
        with:
          fetch-depth: 0

      - name: Download Solar Data Artifacts
        uses: actions/download-artifact@v3
        with:
          name: solar-data-north
          path: ./data/north
      - uses: actions/download-artifact@v3
        with:
          name: solar-data-south
          path: ./data/south
      - uses: actions/download-artifact@v3
        with:
          name: solar-data-east
          path: ./data/east
      - uses: actions/download-artifact@v3
        with:
          name: solar-data-west
          path: ./data/west

      - name: Download Visualization Artifacts
        uses: actions/download-artifact@v3
        with:
          name: visualization
          path: ./visualization

      - name: Configure Git Credentials
        run: |
          git config user.name "github-actions[bot]"
          git config user.email "github-actions[bot]@users.noreply.github.com"

      - name: Combine Solar Data
        run: |
          cat ./data/north/india_solar_data_north.csv ./data/south/india_solar_data_south.csv ./data/east/india_solar_data_east.csv ./data/west/india_solar_data_west.csv > india_solar_data.csv

      - name: Copy Generated Files to Repository
        run: |
          cp india_solar_data.csv india_solar_potential.html solar_potential_high_res.png .

      - name: Commit and Push Changes
        run: |
          git add india_solar_data.csv india_solar_potential.html solar_potential_high_res.png
          git diff --cached --quiet || (
            git commit -m "Update solar potential data [skip ci]" &&
            git push
          )
        env:
          GITHUB_TOKEN: ${{ secrets.GITHUB_TOKEN }}
```

## 📊 Output

- **india_solar_potential.html**: Interactive map showcasing solar potential across India.
- **solar_potential_high_res.png**: High-resolution image of the solar potential map.
- **india_solar_data.csv**: Raw solar potential data for each grid point sampled.

## 📂 File Structure

```plaintext
.
├── data_collection.py                   # Script for fetching and processing solar data
├── visualization.py                    # Script for generating visualizations
├── requirements.txt                    # Python dependencies
├── india-soi.geojson                   # GeoJSON file of India's boundaries
├── .github
│   └── workflows
│       └── run_raydium.yml              # GitHub Actions workflow
├── README.md                           # Project documentation
├── solar_data.log                      # Log file for the data collection process
├── visualization.log                   # Log file for the visualization process
├── india_solar_data.csv                # Aggregated solar potential data
├── india_solar_potential.html          # Interactive solar potential map
└── solar_potential_high_res.png        # High-resolution solar potential image
```

## 📋 Credits

Special thanks to [DataMeet](https://github.com/datameet/maps/blob/master/Country/india-soi.geojson) for the GeoJSON boundary file of India and to [NASA POWER API](https://power.larc.nasa.gov/docs/services/api/) for the data required to calculate the solar potential.
