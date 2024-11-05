import numpy as np
import pandas as pd
import geopandas as gpd
import folium
import requests
from shapely.geometry import Point
from scipy.interpolate import griddata
import time
import branca.colormap as cm
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import os
from concurrent.futures import ThreadPoolExecutor, as_completed
import shelve
from requests.adapters import HTTPAdapter
from requests.packages.urllib3.util.retry import Retry
import logging
import threading
from tqdm import tqdm
import json

# Logging setup
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    handlers=[
        logging.FileHandler("solar_map.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class RateLimiter:
    def __init__(self, max_calls, period):
        self.max_calls = max_calls
        self.period = period
        self.lock = threading.Lock()
        self.calls = []
    
    def acquire(self):
        with self.lock:
            current = time.time()
            while self.calls and self.calls[0] <= current - self.period:
                self.calls.pop(0)
            if len(self.calls) >= self.max_calls:
                sleep_time = self.period - (current - self.calls[0]) + 0.1
                logger.info(f"Rate limit reached. Sleeping for {sleep_time:.2f} seconds.")
                time.sleep(sleep_time)
            self.calls.append(time.time())

rate_limiter = RateLimiter(max_calls=1, period=2)

def create_session_with_retry():
    session = requests.Session()
    retries = Retry(
        total=10,
        backoff_factor=2,
        status_forcelist=[429, 500, 502, 503, 504],
        allowed_methods=["HEAD", "GET", "OPTIONS"],
        respect_retry_after_header=True
    )
    adapter = HTTPAdapter(max_retries=retries)
    session.mount('https://', adapter)
    session.mount('http://', adapter)
    return session

def fetch_nasa_power_data(lat, lon, cache, session):
    key = f"{lat}_{lon}"
    if key in cache:
        return cache[key]
    
    base_url = "https://power.larc.nasa.gov/api/temporal/daily/point"
    params = {
        'parameters': 'ALLSKY_SFC_SW_DWN',
        'community': 'RE',
        'longitude': str(lon),
        'latitude': str(lat),
        'start': '20230101',
        'end': '20231231',
        'format': 'JSON'
    }
    
    while True:
        try:
            rate_limiter.acquire()
            response = session.get(base_url, params=params)
            if response.status_code == 429:
                retry_after = response.headers.get('Retry-After')
                if retry_after:
                    sleep_time = int(retry_after) + 1
                else:
                    sleep_time = 60
                logger.warning(f"Received 429 for lat={lat}, lon={lon}. Retrying after {sleep_time} seconds.")
                time.sleep(sleep_time)
                continue
            response.raise_for_status()
            data = response.json()
            cache[key] = data
            return data
        except requests.exceptions.RequestException as e:
            logger.error(f"RequestException for lat={lat}, lon={lon}: {e}. Retrying in 10 seconds.")
            time.sleep(10)
        except json.JSONDecodeError as e:
            logger.error(f"JSONDecodeError for lat={lat}, lon={lon}: {e}. Skipping this point.")
            cache[key] = None
            return None

def calculate_solar_potential(power_data, panel_efficiency=0.2):
    if not power_data:
        return None
    try:
        daily_data = power_data['properties']['parameter']['ALLSKY_SFC_SW_DWN']
        radiation_values = [v * 0.277778 for v in daily_data.values() 
                            if isinstance(v, (int, float)) and v != -999]
        if not radiation_values:
            return None
        return np.mean(radiation_values) * 365 * panel_efficiency
    except KeyError as e:
        logger.error(f"KeyError while calculating solar potential: {e}")
        return None
    except Exception as e:
        logger.error(f"Unexpected error while calculating solar potential: {e}")
        return None

def fetch_and_calculate(lat, lon, cache, session):
    power_data = fetch_nasa_power_data(lat, lon, cache, session)
    potential = calculate_solar_potential(power_data)
    if potential is not None:
        return {'latitude': lat, 'longitude': lon, 'potential': potential}
    return None

def create_india_solar_map(geojson_path='india-soi.geojson'):
    logger.info(f"Reading GeoJSON file from: {geojson_path}")
    india = gpd.read_file(geojson_path)
    
    # Use total_bounds instead of simplifying geometry
    bounds = india.total_bounds
    logger.info(f"Map bounds: {bounds}")
    
    # Center the map on India
    center_lat = (bounds[1] + bounds[3]) / 2
    center_lon = (bounds[0] + bounds[2]) / 2
    m = folium.Map(location=[center_lat, center_lon], zoom_start=5, tiles='CartoDB positron')
    
    # Define grid parameters
    lat_step = 0.33
    lon_step = 0.33
    lat_range = np.arange(bounds[1], bounds[3] + lat_step, lat_step)
    lon_range = np.arange(bounds[0], bounds[2] + lon_step, lon_step)
    
    # Data collection
    cache_file = 'nasa_power_cache.db'
    cache = shelve.open(cache_file)
    session = create_session_with_retry()
    max_workers = 3
    
    logger.info(f"Starting data collection with {max_workers} workers...")
    solar_data = []
    
    # Create grid points within India's boundary
    india_union = india.unary_union
    grid_points = [(lat, lon) for lat in lat_range for lon in lon_range 
                   if india_union.contains(Point(lon, lat)) or india_union.touches(Point(lon, lat))]
    
    total_grid_points = len(grid_points)
    logger.info(f"Total grid points to process: {total_grid_points}")
    
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {executor.submit(fetch_and_calculate, lat, lon, cache, session): (lat, lon) 
                  for lat, lon in grid_points}
        
        for future in tqdm(as_completed(futures), total=total_grid_points, desc="Processing Points"):
            result = future.result()
            if result:
                solar_data.append(result)
    
    cache.close()
    
    # Create DataFrame from collected data
    solar_df = pd.DataFrame(solar_data)
    if solar_df.empty:
        raise ValueError("No solar data collected")
    
    logger.info("Creating visualization...")
    
    # Increase grid size for smoother interpolation
    grid_size = 750
    
    # Create regular grid
    grid_lat = np.linspace(bounds[1], bounds[3], grid_size)
    grid_lon = np.linspace(bounds[0], bounds[2], grid_size)
    grid_lon, grid_lat = np.meshgrid(grid_lon, grid_lat)
    
    # Interpolate data
    points = solar_df[['longitude', 'latitude']].values
    values = solar_df['potential'].values
    grid_z = griddata(points, values, (grid_lon, grid_lat), method='linear')
    
    # Create mask using precise boundary
    grid_points_mesh = [Point(lon, lat) for lon, lat in zip(grid_lon.flatten(), grid_lat.flatten())]
    india_gdf = gpd.GeoSeries(grid_points_mesh)
    mask = india_gdf.within(india_union).values.reshape(grid_z.shape)
    grid_z = np.ma.masked_array(grid_z, ~mask)
    
    # Create colormap
    colormap = cm.LinearColormap(
        ['#440154', '#482878', '#3E4989', '#31688E', '#26828E',
         '#1F9E89', '#35B779', '#6DCD59', '#B4DE2C', '#FDE725'],
        vmin=np.nanmin(values),
        vmax=np.nanmax(values),
        caption='Solar Potential (kWh/m²/year)'
    )
    
    # Save high-resolution image
    fig, ax = plt.subplots(figsize=(20, 24))
    ax.set_axis_off()
    
    # Use exact bounds for imshow
    img = ax.imshow(
        grid_z,
        cmap=ListedColormap(colormap.colors),
        extent=[bounds[0], bounds[2], bounds[1], bounds[3]],
        origin='lower',
        aspect='auto'
    )
    
    # Add India boundary to matplotlib plot
    india.boundary.plot(ax=ax, color='black', linewidth=1)
    
    image_path = 'solar_potential_high_res.png'
    plt.savefig(image_path, bbox_inches='tight', pad_inches=0, dpi=600)
    plt.close(fig)
    
    # Add image overlay with precise bounds
    folium.raster_layers.ImageOverlay(
        name='Solar Potential',
        image=image_path,
        bounds=[[bounds[1], bounds[0]], [bounds[3], bounds[2]]],
        opacity=0.7,
        interactive=True,
        cross_origin=False,
        zindex=1,
    ).add_to(m)
    
    # Add India boundary
    folium.GeoJson(
        india,
        style_function=lambda x: {
            'fillColor': 'none',
            'color': 'black',
            'weight': 1.5,
            'fillOpacity': 0
        },
        name='India Boundary'
    ).add_to(m)
    
    colormap.add_to(m)
    folium.LayerControl().add_to(m)
    
    # Save outputs
    output_csv = 'india_solar_data.csv'
    output_html = 'india_solar_potential.html'
    logger.info(f"Saving data to {output_csv}")
    solar_df.to_csv(output_csv, index=False)
    logger.info(f"Saving map to {output_html}")
    m.save(output_html)
    
    return m, solar_df

def main():
    try:
        logger.info("Starting solar potential map generation with improved alignment...")
        solar_map, solar_data = create_india_solar_map()
        logger.info("Process completed successfully!")
        logger.info("Map has been saved as 'india_solar_potential.html'")
        logger.info("Raw data has been saved as 'india_solar_data.csv'")
    except Exception as e:
        logger.error(f"An error occurred: {e}")

if __name__ == "__main__":
    main()
