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
from datetime import datetime

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
    def __init__(self, max_calls, period, backoff_factor=1.5):
        self.max_calls = max_calls
        self.period = period
        self.calls = []
        self.lock = asyncio.Lock()
        self.backoff_factor = backoff_factor
        self.current_backoff = 0
        self.max_backoff = 60  # Maximum backoff time in seconds

    async def acquire(self):
        async with self.lock:
            current = time.time()
            self.calls = [t for t in self.calls if t > current - self.period]
            
            if len(self.calls) >= self.max_calls:
                # Calculate base sleep time
                sleep_time = self.period - (current - min(self.calls)) + 0.1
                
                # Add exponential backoff if we're hitting limits frequently
                if self.current_backoff > 0:
                    sleep_time += self.current_backoff
                    self.current_backoff = min(
                        self.current_backoff * self.backoff_factor,
                        self.max_backoff
                    )
                
                await asyncio.sleep(sleep_time)
                self.calls = []  # Reset after backing off
            else:
                # Gradually reduce backoff when successful
                self.current_backoff = max(0, self.current_backoff / self.backoff_factor)
            
            self.calls.append(current)

# Increased rate limit with proper backoff
rate_limiter = RateLimiter(max_calls=3, period=1)

class SolarDataValidator:
    # NASA POWER radiation data typical ranges (kWh/m²/day)
    MIN_RADIATION = 0.5
    MAX_RADIATION = 8.5
    
    @staticmethod
    def validate_coordinates(lat, lon):
        return -90 <= lat <= 90 and -180 <= lon <= 180
    
    @staticmethod
    def validate_radiation(value):
        return SolarDataValidator.MIN_RADIATION <= value <= SolarDataValidator.MAX_RADIATION
    
    @staticmethod
    def convert_radiation(value):
        # NASA POWER data is in MJ/m²/day, convert to kWh/m²/day
        # 1 MJ = 0.277778 kWh (verified conversion factor)
        return value * 0.277778

async def fetch_solar_data(session, latitude, longitude, cache):
    if not SolarDataValidator.validate_coordinates(latitude, longitude):
        logger.error(f"Invalid coordinates: ({latitude}, {longitude})")
        return None

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
                logger.warning("Rate limit exceeded, backing off...")
                rate_limiter.current_backoff = max(
                    rate_limiter.current_backoff * rate_limiter.backoff_factor,
                    5  # Minimum backoff of 5 seconds
                )
                return None
            elif response.status != 200:
                logger.error(f"Error {response.status} for ({latitude}, {longitude})")
                return None

            data = await response.json()
            solar_data = data.get("properties", {}).get("parameter", {}).get("ALLSKY_SFC_SW_DWN", {})
            if not solar_data:
                return None

            # Calculate daily averages and validate
            daily_values = []
            for value in solar_data.values():
                converted_value = SolarDataValidator.convert_radiation(value)
                if SolarDataValidator.validate_radiation(converted_value):
                    daily_values.append(converted_value)
                else:
                    logger.warning(f"Invalid radiation value {converted_value} at ({latitude}, {longitude})")

            if not daily_values:
                return None

            avg_radiation = round(sum(daily_values) / len(daily_values), 2)
            result = {
                "latitude": latitude,
                "longitude": longitude,
                "potential": avg_radiation,
                "min_radiation": round(min(daily_values), 2),
                "max_radiation": round(max(daily_values), 2)
            }
            cache[cache_key] = result
            return result

    except Exception as e:
        logger.error(f"Error fetching data for ({latitude}, {longitude}): {e}")
        return None

class ProgressTracker:
    def __init__(self, total_points):
        self.total_points = total_points
        self.processed_points = 0
        self.start_time = time.time()
        self.checkpoint_interval = 100
        self.last_checkpoint = 0

    def update(self, batch_size):
        self.processed_points += batch_size
        if self.processed_points - self.last_checkpoint >= self.checkpoint_interval:
            self.save_checkpoint()
            self.last_checkpoint = self.processed_points
        
        # Calculate progress and ETA
        progress = (self.processed_points / self.total_points) * 100
        elapsed_time = time.time() - self.start_time
        points_per_second = self.processed_points / elapsed_time if elapsed_time > 0 else 0
        remaining_points = self.total_points - self.processed_points
        eta_seconds = remaining_points / points_per_second if points_per_second > 0 else 0
        
        logger.info(
            f"Progress: {progress:.1f}% ({self.processed_points}/{self.total_points}) "
            f"Points/sec: {points_per_second:.2f} "
            f"ETA: {datetime.fromtimestamp(time.time() + eta_seconds).strftime('%Y-%m-%d %H:%M:%S')}"
        )

    def save_checkpoint(self):
        checkpoint_file = f'solar_data_checkpoint_{self.processed_points}.json'
        logger.info(f"Saving checkpoint: {checkpoint_file}")
        # Actual saving logic would go here

async def process_grid_points(grid_points, batch_size=50):
    cache = dc.Cache('nasa_power_cache')
    solar_data = []
    progress_tracker = ProgressTracker(len(grid_points))
    
    async with aiohttp.ClientSession() as session:
        for i in range(0, len(grid_points), batch_size):
            batch = grid_points.iloc[i:i+batch_size]
            tasks = [fetch_solar_data(session, point.y, point.x, cache) 
                    for point in batch.geometry]
            
            results = await asyncio.gather(*tasks)
            valid_results = [r for r in results if r is not None]
            solar_data.extend(valid_results)
            
            progress_tracker.update(batch_size)
            
            # Save partial results more frequently
            if len(solar_data) % 500 == 0:
                df = pd.DataFrame(solar_data)
                timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
                df.to_csv(f'solar_data_partial_{timestamp}.csv', index=False)
    
    cache.close()
    return solar_data

def create_grid_points(geojson_path='india-soi.geojson', resolution=10000, chunk_size=100):
    try:
        if not os.path.exists(geojson_path):
            raise FileNotFoundError(f"GeoJSON file not found: {geojson_path}")
            
        india = gpd.read_file(geojson_path)
        
        # Ensure input geometry is in WGS84
        if india.crs != 'EPSG:4326':
            india = india.to_crs('EPSG:4326')
        
        # Use a more appropriate projection for India
        # Using Indian Grid (EPSG:24378) for better accuracy
        india_proj = india.to_crs('EPSG:24378')
        bounds = india_proj.total_bounds
        
        # Calculate grid dimensions
        width = int((bounds[2] - bounds[0]) / resolution)
        height = int((bounds[3] - bounds[1]) / resolution)
        
        # Create hexagonal grid instead of rectangular
        # This ensures better coverage with fewer points
        hex_size = resolution * 0.866  # Height of a hexagon
        hex_width = resolution
        
        valid_points = []
        total_processed = 0
        total_valid = 0

        for i in range(width):
            x = bounds[0] + i * hex_width
            # Offset every other row
            offset = (hex_size / 2) if (i % 2) == 0 else 0
            
            for j in range(height):
                y = bounds[1] + j * hex_size + offset
                point = Point(x, y)
                
                if india_proj.geometry.contains(point).any():
                    valid_points.append(point)
                    total_valid += 1
                
                total_processed += 1
                
                if total_valid % 1000 == 0:
                    logger.info(f"Generated {total_valid} valid points out of {total_processed} total")
        
        if not valid_points:
            raise ValueError("No valid points found within India's boundary")
        
        # Create GeoDataFrame with all valid points
        result = gpd.GeoDataFrame(geometry=valid_points, crs=india_proj.crs)
        
        # Convert back to WGS84
        result_wgs84 = result.to_crs('EPSG:4326')
        
        # Validate final points
        india_wgs84 = india.to_crs('EPSG:4326')
        final_points = result_wgs84[result_wgs84.geometry.within(india_wgs84.unary_union)]
        
        logger.info(f"Final grid contains {len(final_points)} points")
        return final_points
        
    except Exception as e:
        logger.error(f"Error in create_grid_points: {str(e)}")
        raise

async def main():
    try:
        # Create output directory if it doesn't exist
        os.makedirs('output', exist_ok=True)
        
        # Using 15km resolution for better coverage
        grid_points = create_grid_points(resolution=15000, chunk_size=50)
        logger.info(f"Created {len(grid_points)} grid points")
        
        # Save the grid points before processing
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        grid_points.to_csv(f'output/grid_points_{timestamp}.csv', index=False)
        logger.info("Saved grid points to CSV")
        
        solar_data = await process_grid_points(grid_points)
        df = pd.DataFrame(solar_data)
        
        # Save final results with timestamp
        df.to_csv(f'output/india_solar_data_{timestamp}.csv', index=False)
        
        # Generate summary statistics
        summary = {
            'total_points': len(df),
            'avg_potential': df['potential'].mean(),
            'min_potential': df['potential'].min(),
            'max_potential': df['potential'].max(),
            'completion_time': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        }
        
        pd.DataFrame([summary]).to_csv(f'output/summary_{timestamp}.csv', index=False)
        logger.info("Data collection completed successfully")
        
    except Exception as e:
        logger.error(f"Error in main: {e}")
        
if __name__ == "__main__":
    asyncio.run(main())