import os
import fiona
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

def create_grid_points(geojson_path='india-soi.geojson', resolution=10000, chunk_size=100):
    try:
        # Load and verify the GeoJSON
        if not os.path.exists(geojson_path):
            raise FileNotFoundError(f"GeoJSON file not found: {geojson_path}")
            
        india = gpd.read_file(geojson_path)
        
        # Ensure input geometry is in WGS84
        if india.crs != 'EPSG:4326':
            india = india.to_crs('EPSG:4326')
            
        # Project to an appropriate UTM zone for India
        # Using UTM zone 44N which is suitable for most of India
        india_proj = india.to_crs('EPSG:32644')
        bounds = india_proj.total_bounds
        
        # Calculate grid dimensions
        width = int((bounds[2] - bounds[0]) / resolution)
        height = int((bounds[3] - bounds[1]) / resolution)
        
        # Pre-compute the boundary geometry with buffer
        # Adding a small negative buffer to ensure points are well within boundaries
        india_union = india_proj.unary_union.buffer(-resolution/10)
        
        valid_points = []
        total_processed = 0
        total_valid = 0

        # Generate points in chunks
        for i in range(0, width, chunk_size):
            chunk_start_x = bounds[0] + i * resolution
            chunk_end_x = min(bounds[0] + (i + chunk_size) * resolution, bounds[2])
            
            for j in range(0, height, chunk_size):
                chunk_start_y = bounds[1] + j * resolution
                chunk_end_y = min(bounds[1] + (j + chunk_size) * resolution, bounds[3])
                
                # Create points for this chunk
                x_coords = np.linspace(chunk_start_x, chunk_end_x, 
                                     min(chunk_size, int((chunk_end_x-chunk_start_x)/resolution)))
                y_coords = np.linspace(chunk_start_y, chunk_end_y, 
                                     min(chunk_size, int((chunk_end_y-chunk_start_y)/resolution)))
                
                # Create mesh grid
                xx, yy = np.meshgrid(x_coords, y_coords)
                points = [Point(x, y) for x, y in zip(xx.ravel(), yy.ravel())]
                
                total_processed += len(points)
                
                # Create GeoDataFrame for the chunk
                chunk_gdf = gpd.GeoDataFrame(geometry=points, crs=india_proj.crs)
                
                # Use contains instead of intersects for stricter filtering
                valid_chunk_points = chunk_gdf[chunk_gdf.geometry.within(india_union)]
                
                if len(valid_chunk_points) > 0:
                    valid_points.append(valid_chunk_points)
                    total_valid += len(valid_chunk_points)
                    
                    # Save progress periodically
                    if total_valid % 10000 == 0:
                        temp_result = pd.concat(valid_points)
                        temp_result_wgs84 = temp_result.to_crs('EPSG:4326')
                        temp_result_wgs84.to_csv(f'grid_points_temp_{total_valid}.csv', index=False)
                        
        if not valid_points:
            raise ValueError("No valid points found within India's boundary")
            
        # Combine all valid points
        result = pd.concat(valid_points)
        
        # Convert back to WGS84 for final output
        result_wgs84 = result.to_crs('EPSG:4326')
        
        # Add final validation step
        india_wgs84 = india.to_crs('EPSG:4326')
        final_points = result_wgs84[result_wgs84.geometry.within(india_wgs84.unary_union)]
        
        return final_points
        
    except Exception as e:
        logger.error(f"Error in create_grid_points: {str(e)}")
        raise
    
async def main():
    try:
        # Using a larger resolution (20km) for initial testing
        grid_points = create_grid_points(resolution=20000, chunk_size=50)
        logger.info(f"Created {len(grid_points)} grid points")
        
        # Save the grid points before processing
        grid_points.to_csv('grid_points.csv', index=False)
        logger.info("Saved grid points to CSV")
        
        solar_data = await process_grid_points(grid_points)
        df = pd.DataFrame(solar_data)
        df.to_csv('india_solar_data.csv', index=False)
        logger.info("Data collection completed")
        
    except Exception as e:
        logger.error(f"Error in main: {e}")
        
if __name__ == "__main__":
    asyncio.run(main())