import asyncio
import aiohttp
import numpy as np
import pandas as pd
import geopandas as gpd
import folium
from shapely.geometry import Point, box
import time
import branca.colormap as cm
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import logging
import diskcache as dc
from tqdm.asyncio import tqdm_asyncio
import json
import rasterio
from rasterio.transform import from_bounds
from rasterio.features import geometry_mask
from rasterio.warp import calculate_default_transform, reproject, Resampling
import pyproj
from affine import Affine

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
        self.calls = []
        self.lock = asyncio.Lock()

    async def acquire(self):
        async with self.lock:
            current = time.time()
            while self.calls and self.calls[0] <= current - self.period:
                self.calls.pop(0)
            if len(self.calls) >= self.max_calls:
                sleep_time = self.period - (current - self.calls[0]) + 0.1
                logger.info(f"Rate limit reached. Sleeping for {sleep_time:.2f} seconds.")
                await asyncio.sleep(sleep_time)
            self.calls.append(time.time())

rate_limiter = RateLimiter(max_calls=1, period=2)  # 1 call every 2 seconds

async def fetch_and_calculate(session, latitude, longitude, cache):
    """
    Fetch solar data from NASA POWER API for the given latitude and longitude.
    Calculate solar potential and return the result.

    Args:
        session (aiohttp.ClientSession): The HTTP session for making requests.
        latitude (float): Latitude of the point.
        longitude (float): Longitude of the point.
        cache (diskcache.Cache): Cache for storing and retrieving fetched data.

    Returns:
        dict: Dictionary containing solar potential and coordinates, or None if failed.
    """
    # Define the cache key based on coordinates
    cache_key = f"solar_data_{latitude}_{longitude}"

    # Check if data is already cached
    if cache_key in cache:
        logger.info(f"Cache hit for coordinates: ({latitude}, {longitude})")
        solar_potential = cache[cache_key]
    else:
        logger.info(f"Fetching data for coordinates: ({latitude}, {longitude})")
        # NASA POWER API endpoint for solar radiation
        api_url = "https://power.larc.nasa.gov/api/temporal/daily/point"

        # Define API parameters
        params = {
            "parameters": "ALLSKY_SFC_SW_DWN",  # All-sky surface shortwave downwelling radiation
            "community": "SB",
            "longitude": longitude,
            "latitude": latitude,
            "start": "20200101",
            "end": "20201231",
            "format": "JSON"
        }

        try:
            await rate_limiter.acquire()  # Ensure rate limiting
            async with session.get(api_url, params=params) as response:
                if response.status != 200:
                    logger.error(f"Failed to fetch data for ({latitude}, {longitude}). Status: {response.status}")
                    return None
                data = await response.json()

                # Extract the solar radiation data
                solar_radiation = data.get("properties", {}).get("parameter", {}).get("ALLSKY_SFC_SW_DWN", {})
                if not solar_radiation:
                    logger.error(f"No solar radiation data found for ({latitude}, {longitude})")
                    return None

                # Calculate the average solar potential (in kWh/m²/year)
                total_radiation = sum(solar_radiation.values())
                average_radiation = total_radiation / len(solar_radiation) * 0.0036  # Convert from MJ/m²/day to kWh/m²/year
                solar_potential = round(average_radiation, 2)

                # Store in cache
                cache[cache_key] = solar_potential
        except Exception as e:
            logger.error(f"Exception while fetching data for ({latitude}, {longitude}): {e}")
            return None

    return {
        "latitude": latitude,
        "longitude": longitude,
        "potential": solar_potential
    }

async def create_india_solar_map_async(geojson_path='india-soi.geojson'):
    logger.info(f"Reading GeoJSON file from: {geojson_path}")
    
    # Read and reproject the GeoJSON to a suitable projection for India
    india = gpd.read_file(geojson_path)
    
    # Convert to Asia South Albers Equal Area Conic projection
    india_proj = india.to_crs('EPSG:24370')  # This projection is suitable for India
    bounds = india_proj.total_bounds
    
    # Calculate grid parameters
    resolution = 5000  # 5km resolution
    width = int((bounds[2] - bounds[0]) / resolution)
    height = int((bounds[3] - bounds[1]) / resolution)
    
    # Create the transform matrix for the raster
    transform = from_bounds(bounds[0], bounds[1], bounds[2], bounds[3], width, height)
    
    # Create grid points in the projected CRS
    x_coords = np.linspace(bounds[0], bounds[2], width)
    y_coords = np.linspace(bounds[1], bounds[3], height)
    xx, yy = np.meshgrid(x_coords, y_coords)
    
    # Create point geometries for each grid cell
    points = [Point(x, y) for x, y in zip(xx.flatten(), yy.flatten())]
    grid_gdf = gpd.GeoDataFrame(geometry=points, crs=india_proj.crs)
    
    # Create mask for points within India
    mask = grid_gdf.within(india_proj.unary_union)
    valid_points = grid_gdf[mask]
    
    # Convert points back to WGS84 for API calls
    valid_points_wgs84 = valid_points.to_crs('EPSG:4326')
    
    # Setup cache and session
    cache = dc.Cache('nasa_power_cache')
    connector = aiohttp.TCPConnector(limit=10)
    timeout = aiohttp.ClientTimeout(total=60)
    
    logger.info("Starting data collection with asynchronous requests...")
    solar_data = []
    
    async with aiohttp.ClientSession(connector=connector, timeout=timeout) as session:
        tasks = [
            fetch_and_calculate(session, point.y, point.x, cache)
            for point in valid_points_wgs84.geometry
        ]
        
        for future in tqdm_asyncio.as_completed(tasks, total=len(tasks), desc="Processing Points"):
            result = await future
            if result:
                solar_data.append(result)
    
    cache.close()
    
    # Create DataFrame and convert to GeoDataFrame
    solar_df = pd.DataFrame(solar_data)
    solar_gdf = gpd.GeoDataFrame(
        solar_df,
        geometry=[Point(xy) for xy in zip(solar_df.longitude, solar_df.latitude)],
        crs='EPSG:4326'
    )
    
    # Project to the same CRS as the boundary
    solar_gdf_proj = solar_gdf.to_crs(india_proj.crs)
    
    # Create empty raster
    raster_data = np.zeros((height, width), dtype=np.float32)
    raster_data[:] = np.nan
    
    # Fill raster with solar potential values
    for idx, row in solar_gdf_proj.iterrows():
        # Get raster indices for point
        col_float, row_float = ~transform * (row.geometry.x, row.geometry.y)
        col, row_idx = int(col_float), int(row_float)
        if 0 <= row_idx < height and 0 <= col < width:
            raster_data[row_idx, col] = row['potential']
    
    # Create mask from India boundary
    geometry = [feature['geometry'] for feature in india_proj.__geo_interface__['features']]
    mask_raster = geometry_mask(geometry, out_shape=(height, width), transform=transform, invert=True)
    
    # Apply mask
    raster_data = np.ma.masked_array(raster_data, ~mask_raster)
    
    # Create visualization
    fig, ax = plt.subplots(figsize=(20, 24))
    ax.set_axis_off()
    
    # Create colormap
    colormap = cm.LinearColormap(
        ['#440154', '#482878', '#3E4989', '#31688E', '#26828E',
         '#1F9E89', '#35B779', '#6DCD59', '#B4DE2C', '#FDE725'],
        vmin=np.nanmin(raster_data),
        vmax=np.nanmax(raster_data),
        caption='Solar Potential (kWh/m²/year)'
    )
    
    # Plot the raster
    img = ax.imshow(
        raster_data,
        extent=[bounds[0], bounds[2], bounds[1], bounds[3]],
        cmap=ListedColormap(colormap.colors),
        origin='lower'
    )
    
    # Plot boundary
    india_proj.boundary.plot(ax=ax, color='black', linewidth=1)
    
    # Save high-resolution image
    image_path = 'solar_potential_high_res.png'
    plt.savefig(image_path, bbox_inches='tight', pad_inches=0, dpi=300)
    plt.close(fig)
    
    # Create Folium map
    center_lat = (india.total_bounds[1] + india.total_bounds[3]) / 2
    center_lon = (india.total_bounds[0] + india.total_bounds[2]) / 2
    m = folium.Map(location=[center_lat, center_lon], zoom_start=5, tiles='CartoDB positron')
    
    # Add the raster overlay
    img_bounds = [[india.total_bounds[1], india.total_bounds[0]], 
                  [india.total_bounds[3], india.total_bounds[2]]]
    
    folium.raster_layers.ImageOverlay(
        name='Solar Potential',
        image=image_path,
        bounds=img_bounds,
        opacity=0.7,
        interactive=True,
        cross_origin=False,
        zindex=1,
    ).add_to(m)
    
    # Add boundary overlay
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
    
    # Add colormap and layer control
    colormap.add_to(m)
    folium.LayerControl().add_to(m)
    
    # Save outputs
    output_csv = 'india_solar_data.csv'
    output_html = 'india_solar_potential.html'
    solar_df.to_csv(output_csv, index=False)
    m.save(output_html)
    
    return m, solar_df

async def main_async():
    try:
        logger.info("Starting solar potential map generation with improved alignment...")
        solar_map, solar_data = await create_india_solar_map_async()
        logger.info("Process completed successfully!")
        logger.info("Map has been saved as 'india_solar_potential.html'")
        logger.info("Raw data has been saved as 'india_solar_data.csv'")
    except Exception as e:
        logger.error(f"An error occurred: {e}")

def main():
    asyncio.run(main_async())

if __name__ == "__main__":
    main()