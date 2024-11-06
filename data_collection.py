import asyncio
import aiohttp
import numpy as np
import pandas as pd
import os
import logging
import diskcache as dc
from tqdm.asyncio import tqdm_asyncio
from shapely.geometry import Point

# Import necessary modules and classes from raydium.py
from raydium import RateLimiter, fetch_nasa_power_data, calculate_solar_potential, fetch_and_calculate

# Logging setup (same as in raydium.py)
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    handlers=[
        logging.FileHandler("solar_data.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

async def collect_data(geojson_path='india-soi.geojson', output_csv='india_solar_data.csv'):
    # Read GeoJSON and prepare grid points (same as in raydium.py)
    import geopandas as gpd
    import numpy as np

    logger.info(f"Reading GeoJSON file from: {geojson_path}")
    india = gpd.read_file(geojson_path)
    bounds = india.total_bounds
    logger.info(f"Map bounds: {bounds}")
    
    # Adjust grid step size if necessary
    lat_step = 0.5
    lon_step = 0.5
    lat_range = np.arange(bounds[1], bounds[3] + lat_step, lat_step)
    lon_range = np.arange(bounds[0], bounds[2] + lon_step, lon_step)
    
    cache = dc.Cache('nasa_power_cache')
    connector = aiohttp.TCPConnector(limit=10)
    timeout = aiohttp.ClientTimeout(total=60)
    
    solar_data = []
    india_union = india.unary_union
    grid_points = [(lat, lon) for lat in lat_range for lon in lon_range 
                   if india_union.contains(Point(lon, lat)) or india_union.touches(Point(lon, lat))]
    
    total_grid_points = len(grid_points)
    logger.info(f"Total grid points to process: {total_grid_points}")
    
    async with aiohttp.ClientSession(connector=connector, timeout=timeout) as session:
        tasks = [
            fetch_and_calculate(session, lat, lon, cache)
            for lat, lon in grid_points
        ]
        for future in tqdm_asyncio.as_completed(tasks, total=total_grid_points, desc="Processing Points"):
            result = await future
            if result:
                solar_data.append(result)
    
    cache.close()
    
    solar_df = pd.DataFrame(solar_data)
    if solar_df.empty:
        raise ValueError("No solar data collected")
    
    logger.info(f"Saving solar data to {output_csv}")
    solar_df.to_csv(output_csv, index=False)
    
    return solar_df

async def main():
    await collect_data()

if __name__ == "__main__":
    asyncio.run(main())