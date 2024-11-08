import asyncio
import aiohttp
import numpy as np
import pandas as pd
import geopandas as gpd
import folium
from shapely.geometry import Point
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

# Adjust rate limiter based on API rate limits
rate_limiter = RateLimiter(max_calls=5, period=1)  # 5 calls per second

async def fetch_and_calculate(session, latitude, longitude, cache):
    cache_key = f"solar_data_{latitude}_{longitude}"
    if cache_key in cache:
        logger.info(f"Cache hit for coordinates: ({latitude}, {longitude})")
        solar_potential = cache[cache_key]
    else:
        logger.info(f"Fetching data for coordinates: ({latitude}, {longitude})")
        api_url = "https://power.larc.nasa.gov/api/temporal/daily/point"
        params = {
            "parameters": "ALLSKY_SFC_SW_DWN",
            "community": "SB",
            "longitude": longitude,
            "latitude": latitude,
            "start": "20200101",
            "end": "20201231",
            "format": "JSON"
        }
        try:
            await rate_limiter.acquire()
            async with session.get(api_url, params=params, timeout=10) as response:
                if response.status != 200:
                    logger.error(f"Failed to fetch data for ({latitude}, {longitude}). Status: {response.status}")
                    return None
                data = await response.json()
                solar_radiation = data.get("properties", {}).get("parameter", {}).get("ALLSKY_SFC_SW_DWN", {})
                if not solar_radiation:
                    logger.error(f"No solar radiation data found for ({latitude}, {longitude})")
                    return None
                total_radiation = sum(solar_radiation.values())
                average_radiation = total_radiation / len(solar_radiation) * 0.0036  # Convert from MJ/m²/day to kWh/m²/year
                solar_potential = round(average_radiation, 2)
                cache[cache_key] = solar_potential
        except asyncio.TimeoutError:
            logger.error(f"Timeout fetching data for ({latitude}, {longitude})")
            return None
        except Exception as e:
            logger.error(f"Exception while fetching data for ({latitude}, {longitude}): {e}")
            return None

    return {
        "latitude": latitude,
        "longitude": longitude,
        "potential": solar_potential
    }

async def process_batch(session, batch, cache, solar_data):
    tasks = [
        fetch_and_calculate(session, point.y, point.x, cache)
        for point in batch.geometry
    ]
    results = await asyncio.gather(*tasks)
    for result in results:
        if result:
            solar_data.append(result)

async def create_india_solar_map_async(geojson_path='india-soi.geojson'):
    logger.info(f"Reading GeoJSON file from: {geojson_path}")
    india = gpd.read_file(geojson_path)
    india_proj = india.to_crs('EPSG:32644')  # India Albers Equal Area Conic
    bounds = india_proj.total_bounds
    logger.info(f"Projected bounds: {bounds}")

    resolution = 5000  # 5 km resolution
    width = int((bounds[2] - bounds[0]) / resolution)
    height = int((bounds[3] - bounds[1]) / resolution)
    logger.info(f"Grid resolution set to {resolution} meters.")
    logger.info(f"Grid dimensions: width={width}, height={height}")

    try:
        transform = from_bounds(bounds[0], bounds[1], bounds[2], bounds[3], width, height)
        logger.info("Transform matrix created.")
    except Exception as e:
        logger.error(f"Error creating transform matrix: {e}")
        raise

    try:
        x_coords = np.linspace(bounds[0], bounds[2], width)
        y_coords = np.linspace(bounds[1], bounds[3], height)
        xx, yy = np.meshgrid(x_coords, y_coords)
        logger.info("Meshgrid created.")
    except Exception as e:
        logger.error(f"Error creating meshgrid: {e}")
        raise

    try:
        points = [Point(x, y) for x, y in zip(xx.flatten(), yy.flatten())]
        grid_gdf = gpd.GeoDataFrame(geometry=points, crs=india_proj.crs)
        logger.info("Grid points created.")
    except Exception as e:
        logger.error(f"Error creating grid GeoDataFrame: {e}")
        raise

    try:
        logger.info("Creating spatial index for India geometry.")
        india_union = india_proj.unary_union
        sindex = gpd.GeoSeries([india_union]).sindex
        logger.info("Spatial index created.")
        
        logger.info("Performing spatial 'within' operation.")
        mask = grid_gdf.within(india_union)
        logger.info("Mask created.")
        valid_points = grid_gdf[mask]
        logger.info(f"Valid points within India: {len(valid_points)}")
    except Exception as e:
        logger.error(f"Error during spatial 'within' operation: {e}")
        raise

    cache = dc.Cache('nasa_power_cache')
    connector = aiohttp.TCPConnector(limit=100)  # Adjusted for higher concurrency
    timeout = aiohttp.ClientTimeout(total=60)
    logger.info("Starting data collection with asynchronous requests...")
    solar_data = []

    BATCH_SIZE = 1000  # Define an appropriate batch size
    total_points = len(valid_points)
    logger.info(f"Total valid points to process: {total_points}")

    async with aiohttp.ClientSession(connector=connector, timeout=timeout) as session:
        for i in range(0, total_points, BATCH_SIZE):
            batch = valid_points.iloc[i:i+BATCH_SIZE]
            logger.info(f"Processing batch {i//BATCH_SIZE + 1}/{(total_points + BATCH_SIZE - 1)//BATCH_SIZE}")
            await process_batch(session, batch, cache, solar_data)
            logger.info(f"Completed batch {i//BATCH_SIZE + 1}/{(total_points + BATCH_SIZE - 1)//BATCH_SIZE}")

    cache.close()
    logger.info("Data collection completed.")

    # Create DataFrame and convert to GeoDataFrame
    solar_df = pd.DataFrame(solar_data)
    solar_gdf = gpd.GeoDataFrame(
        solar_df,
        geometry=[Point(xy) for xy in zip(solar_df.longitude, solar_df.latitude)],
        crs='EPSG:4326'
    )
    logger.info("Solar data converted to GeoDataFrame.")

    # Project to the same CRS as the boundary
    solar_gdf_proj = solar_gdf.to_crs(india_proj.crs)
    logger.info("Solar GeoDataFrame projected to CRS.")

    # Create empty raster
    raster_data = np.zeros((height, width), dtype=np.float32)
    raster_data[:] = np.nan

    # Fill raster with solar potential values
    for idx, row in solar_gdf_proj.iterrows():
        col_float, row_float = ~transform * (row.geometry.x, row.geometry.y)
        col, row_idx = int(col_float), int(row_float)
        if 0 <= row_idx < height and 0 <= col < width:
            raster_data[row_idx, col] = row['potential']
        if idx % 100000 == 0 and idx > 0:
            logger.info(f"Filled raster data up to index {idx}.")

    logger.info("Raster data filled with solar potentials.")

    # Create mask from India boundary
    geometry = [feature['geometry'] for feature in india_proj.__geo_interface__['features']]
    mask_raster = geometry_mask(geometry, out_shape=(height, width), transform=transform, invert=True)
    raster_data = np.ma.masked_array(raster_data, ~mask_raster)
    logger.info("Applied boundary mask to raster data.")

    # Create visualization
    fig, ax = plt.subplots(figsize=(20, 20))
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
        extent=india_proj.total_bounds,
        cmap=ListedColormap(colormap.colors),
        origin='lower'
    )

    # Plot boundary
    india_proj.boundary.plot(ax=ax, color='black', linewidth=1)
    logger.info("Plotted raster and boundary.")

    # Save high-resolution image
    image_path = 'solar_potential_high_res.png'
    plt.savefig(image_path, bbox_inches='tight', pad_inches=0, dpi=300)
    plt.close(fig)
    logger.info(f"Saved high-resolution image to {image_path}.")

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
    logger.info(f"Saved solar data to {output_csv} and map to {output_html}.")

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
