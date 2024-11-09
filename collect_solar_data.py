import asyncio
import aiohttp
import geopandas as gpd
import pandas as pd
from shapely.geometry import Point
import time
import logging
import diskcache as dc
from rasterio.transform import from_bounds
import numpy as np

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    handlers=[
        logging.FileHandler("solar_data_collection.log"),
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
            self.calls = [t for t in self.calls if t > current - self.period]
            if len(self.calls) >= self.max_calls:
                sleep_time = self.period - (current - min(self.calls)) + 0.1
                await asyncio.sleep(sleep_time)
            self.calls.append(current)

rate_limiter = RateLimiter(max_calls=2, period=1)

async def fetch_solar_data(session, latitude, longitude, cache):
    cache_key = f"solar_data_{latitude}_{longitude}"
    if cache_key in cache:
        return cache[cache_key]

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
        async with session.get(api_url, params=params, timeout=30) as response:
            if response.status == 429:
                logger.warning("Rate limit exceeded, sleeping for 60 seconds")
                await asyncio.sleep(60)
                return None
            elif response.status != 200:
                logger.error(f"Error {response.status} for ({latitude}, {longitude})")
                return None

            data = await response.json()
            solar_data = data.get("properties", {}).get("parameter", {}).get("ALLSKY_SFC_SW_DWN", {})
            if not solar_data:
                return None

            avg_radiation = sum(solar_data.values()) / len(solar_data) * 0.0036
            result = {
                "latitude": latitude,
                "longitude": longitude,
                "potential": round(avg_radiation, 2)
            }
            cache[cache_key] = result
            return result

    except Exception as e:
        logger.error(f"Error fetching data for ({latitude}, {longitude}): {e}")
        return None

async def process_grid_points(grid_points, batch_size=50):
    cache = dc.Cache('nasa_power_cache')
    solar_data = []
    
    async with aiohttp.ClientSession() as session:
        for i in range(0, len(grid_points), batch_size):
            batch = grid_points.iloc[i:i+batch_size]
            tasks = [fetch_solar_data(session, point.y, point.x, cache) 
                    for point in batch.geometry]
            
            results = await asyncio.gather(*tasks)
            valid_results = [r for r in results if r is not None]
            solar_data.extend(valid_results)
            
            logger.info(f"Processed batch {i//batch_size + 1}/{(len(grid_points) + batch_size - 1)//batch_size}")
            
            if i % 1000 == 0:
                df = pd.DataFrame(solar_data)
                df.to_csv('solar_data_partial.csv', index=False)
    
    cache.close()
    return solar_data

def create_grid_points(geojson_path='india-soi.geojson', resolution=5000):
    india = gpd.read_file(geojson_path)
    india_proj = india.to_crs('EPSG:32644')
    bounds = india_proj.total_bounds

    width = int((bounds[2] - bounds[0]) / resolution)
    height = int((bounds[3] - bounds[1]) / resolution)

    x_coords = np.linspace(bounds[0], bounds[2], width)
    y_coords = np.linspace(bounds[1], bounds[3], height)
    xx, yy = np.meshgrid(x_coords, y_coords)

    points = [Point(x, y) for x, y in zip(xx.flatten(), yy.flatten())]
    grid_gdf = gpd.GeoDataFrame(geometry=points, crs=india_proj.crs)
    
    india_union = india_proj.unary_union
    valid_points = grid_gdf[grid_gdf.within(india_union)]
    
    return valid_points.to_crs('EPSG:4326')

async def main():
    try:
        grid_points = create_grid_points()
        logger.info(f"Created {len(grid_points)} grid points")
        
        solar_data = await process_grid_points(grid_points)
        df = pd.DataFrame(solar_data)
        df.to_csv('india_solar_data.csv', index=False)
        logger.info("Data collection completed")
        
    except Exception as e:
        logger.error(f"Error in main: {e}")

if __name__ == "__main__":
    asyncio.run(main())