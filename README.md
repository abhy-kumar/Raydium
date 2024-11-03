# 🌞 Raydium - Solar Potential Analysis for India

Raydium aims to map India's solar potential. This program is still heavily WIP and contributions would be really appreciated. 

## 🌟 Features

- **Solar Potential Mapping**: Fetches and calculates solar potential from NASA POWER API data 🌞
- **Interactive Visualization**: Displays solar potential over India with interactive maps 🖼️
- **High-Resolution Interpolation**: Uses grid sampling for a detailed solar potential map 🗺️
- **Automated with GitHub Actions**: Automatically runs daily or on-demand in GitHub Actions 🕒

## 🛠️ Setup

### Prerequisites
- Python 3.8+
- Required packages in `raydium.py` include: `geopandas`, `folium`, `pvlib`, `requests`, `beautifulsoup4`

### 1️⃣ Clone the Repository

```bash
git clone https://github.com/your-username/raydium.git
cd raydium
```

### 2️⃣ Install Dependencies

```bash
pip install -r requirements.txt
```

Or manually install dependencies with:

```bash
pip install geopandas folium pvlib requests beautifulsoup4
```

### 3️⃣ Run the Script Locally

1. **Upload GeoJSON File**: Ensure you have the `india-soi.geojson` file in the root directory.
2. **Run the Python Script**:

   ```bash
   python raydium.py
   ```

   - This will generate `india_solar_potential.html` (an interactive map) and `india_solar_data.csv` (raw data) in your project directory.

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
    - cron: '0 0 * * *'  # Run daily at midnight (00:00)
  workflow_dispatch:  # Allow manual trigger

jobs:
  generate-solar-data:
    runs-on: ubuntu-latest
    
    steps:
    - name: Checkout repository
      uses: actions/checkout@v3
      
    - name: Set up Python
      uses: actions/setup-python@v4
      with:
        python-version: '3.10'
        
    - name: Install dependencies
      run: |
        python -m pip install --upgrade pip
        pip install numpy pandas geopandas folium requests scipy branca
        
    - name: Run solar potential analysis
      run: |
        python raydium.py
      env:
        PYTHONUNBUFFERED: 1
        
    - name: Check if files were generated
      run: |
        if [ ! -f india_solar_potential.html ] || [ ! -f india_solar_data.csv ]; then
          echo "Error: Required files were not generated"
          exit 1
        fi
        
    - name: Configure Git
      run: |
        git config --local user.email "github-actions[bot]@users.noreply.github.com"
        git config --local user.name "github-actions[bot]"
        
    - name: Commit and push if there are changes
      run: |
        git add india_solar_potential.html india_solar_data.csv
        git diff --staged --quiet || (git commit -m "Update solar potential data [skip ci]" && git push)
      env:
        GITHUB_TOKEN: ${{ secrets.GITHUB_TOKEN }}

```

## 📊 Output

- **india_solar_potential.html**: Interactive map showcasing solar potential across India.
- **india_solar_data.csv**: Raw solar potential data for each grid point sampled.

Both files are accessible as artifacts in GitHub Actions and are downloadable for further analysis.

## 📂 File Structure

```plaintext
.
├── raydium.py                  # Main script for fetching and visualizing solar potential
├── india-soi.geojson           # GeoJSON file of India's boundaries
├── .github
│   └── workflows
│       └── run_raydium.yml     # GitHub Actions workflow
├── README.md                   # Project documentation
└── requirements.txt            # Dependencies for local setup
```

## 📋 Credits

Special thanks to [DataMeet](https://github.com/datameet/maps/blob/master/Country/india-soi.geojson) for the GeoJSON boundary file of India and to [NASA POWER API](https://power.larc.nasa.gov/docs/services/api/) for the data required to calculate the solar potential.
