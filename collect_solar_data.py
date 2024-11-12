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
    MIN_RADIATION = 2.0  # Updated for more realistic minimum
    MAX_RADIATION = 8.5
    
    @staticmethod
    def validate_coordinates(lat, lon):
        return -90 <= lat <= 90 and -180 <= lon <= 180
    
    @staticmethod
    def validate_radiation(value):
        return SolarDataValidator.MIN_RADIATION <= value <= SolarDataValidator.MAX_RADIATION
    
    @staticmethod
    def convert_radiation(value):
        # NASA POWER ALLSKY_SFC_SW_DWN data is already in kWh/m²/day
        # No conversion needed
        return value

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

def create_grid_points(geojson_path='india-soi.geojson', resolution=10000):
    """
    Create a grid of points within India's boundary with improved geometry handling.
    """
    try:
        # Load and verify the input geometry
        india = gpd.read_file(geojson_path)
        logger.info(f"Initial CRS: {india.crs}")
        
        # Force to WGS84
        if india.crs != 'EPSG:4326':
            india = india.to_crs('EPSG:4326')
            logger.info("Converted to WGS84")
        
        # Buffer and simplify the geometry slightly to handle any topology issues
        india_geom = india.geometry.buffer(0).unary_union
        logger.info(f"Geometry valid: {india_geom.is_valid}")
        logger.info(f"Geometry type: {india_geom.geom_type}")
        
        # Get bounds
        bounds = india_geom.bounds
        logger.info(f"Bounds: {bounds}")
        
        # Calculate grid spacing in degrees
        # Convert resolution from meters to degrees (approximately)
        mid_lat = (bounds[1] + bounds[3]) / 2
        deg_resolution = resolution / 111000  # 1 degree ≈ 111km at the equator
        
        # Adjust for latitude
        lon_resolution = deg_resolution / np.cos(np.radians(mid_lat))
        
        logger.info(f"Grid resolution - Lat: {deg_resolution}, Lon: {lon_resolution}")
        
        # Generate grid coordinates
        lons = np.arange(bounds[0], bounds[2], lon_resolution)
        lats = np.arange(bounds[1], bounds[3], deg_resolution)
        
        logger.info(f"Grid size: {len(lons)}x{len(lats)} points")
        
        # Create points with progress logging
        valid_points = []
        total_points = len(lons) * len(lats)
        processed = 0
        
        for lon in lons:
            for lat in lats:
                point = Point(lon, lat)
                processed += 1
                
                # Use contains() instead of within() for better performance
                if india_geom.contains(point):
                    valid_points.append(point)
                
                if processed % 1000 == 0:
                    logger.info(
                        f"Processed {processed}/{total_points} points "
                        f"({(processed/total_points)*100:.1f}%) - "
                        f"Valid points so far: {len(valid_points)}"
                    )
        
        if not valid_points:
            logger.error("No valid points found!")
            logger.error(f"Total processed: {processed}")
            logger.error(f"Grid bounds: {bounds}")
            # Test a known point within India
            test_point = Point(77.2, 28.6)  # New Delhi
            logger.error(f"Test point (Delhi) contained: {india_geom.contains(test_point)}")
            raise ValueError("No valid points found within boundary")
        
        # Create GeoDataFrame
        points_gdf = gpd.GeoDataFrame(
            geometry=valid_points,
            crs='EPSG:4326'
        )
        
        # Add coordinates as columns
        points_gdf['latitude'] = points_gdf.geometry.y
        points_gdf['longitude'] = points_gdf.geometry.x
        
        logger.info(f"Successfully created {len(points_gdf)} points")
        
        # Save debug information
        try:
            import matplotlib.pyplot as plt
            fig, ax = plt.subplots(figsize=(15, 15))
            india.plot(ax=ax, alpha=0.5)
            points_gdf.plot(ax=ax, color='red', markersize=1)
            plt.savefig('debug_grid.png')
            logger.info("Saved debug plot")
            
            # Save a few sample points for verification
            sample_points = points_gdf.head(5)
            logger.info("Sample points:")
            for idx, point in sample_points.iterrows():
                logger.info(f"Point {idx}: ({point.longitude}, {point.latitude})")
        except Exception as e:
            logger.warning(f"Could not create debug output: {e}")
        
        return points_gdf
        
    except Exception as e:
        logger.error(f"Error in create_grid_points: {e}")
        logger.exception("Detailed traceback:")
        raise

def main():
    try:
        os.makedirs('output', exist_ok=True)
        grid_points = create_grid_points(resolution=15000)
        
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        output_file = f'output/grid_points_{timestamp}.csv'
        grid_points.to_csv(output_file, index=False)
        
        logger.info(f"Saved {len(grid_points)} points to {output_file}")
        
    except Exception as e:
        logger.error(f"Error in main: {e}")
        logger.exception("Detailed traceback:")

if __name__ == "__main__":
    main()
