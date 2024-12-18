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
import numpy as np
from datetime import datetime
import json

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    handlers=[logging.FileHandler("solar_data_collection.log"), logging.StreamHandler()]
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
        self.max_backoff = 60

    async def acquire(self):
        async with self.lock:
            current = time.time()
            self.calls = [t for t in self.calls if t > current - self.period]
            if len(self.calls) >= self.max_calls:
                sleep_time = self.period - (current - min(self.calls)) + 0.1
                sleep_time += self.current_backoff if self.current_backoff > 0 else 0
                self.current_backoff = min(self.current_backoff * self.backoff_factor, self.max_backoff)
                await asyncio.sleep(sleep_time)
                self.calls = []
            else:
                self.current_backoff = max(0, self.current_backoff / self.backoff_factor)
            self.calls.append(current)

rate_limiter = RateLimiter(max_calls=3, period=1)

class SolarDataValidator:
    MIN_RADIATION = 2.0
    MAX_RADIATION = 8.5
    
    @staticmethod
    def validate_coordinates(lat, lon):
        return -90 <= lat <= 90 and -180 <= lon <= 180
    
    @staticmethod
    def validate_radiation(value):
        return SolarDataValidator.MIN_RADIATION <= value <= SolarDataValidator.MAX_RADIATION
    
    @staticmethod
    def convert_radiation(value):
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
                rate_limiter.current_backoff = max(rate_limiter.current_backoff * rate_limiter.backoff_factor, 5)
                return None
            elif response.status != 200:
                logger.error(f"Error {response.status} for ({latitude}, {longitude})")
                return None

            data = await response.json()
            solar_data = data.get("properties", {}).get("parameter", {}).get("ALLSKY_SFC_SW_DWN", {})
            if not solar_data:
                return None

            daily_values = [
                SolarDataValidator.convert_radiation(value)
                for value in solar_data.values()
                if SolarDataValidator.validate_radiation(value)
            ]
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

class CheckpointManager:
    def __init__(self, checkpoint_dir='checkpoints'):
        self.checkpoint_dir = checkpoint_dir
        os.makedirs(checkpoint_dir, exist_ok=True)
        self.grid_points_file = os.path.join(checkpoint_dir, 'grid_points.geojson')
        self.progress_file = os.path.join(checkpoint_dir, 'progress.json')
        self.data_file = os.path.join(checkpoint_dir, 'collected_data.csv')

    def save_grid_points(self, grid_points):
        if not os.path.exists(self.grid_points_file):
            grid_points.to_file(self.grid_points_file, driver='GeoJSON')

    def save_progress(self, processed_indices):
        with open(self.progress_file, 'w') as f:
            json.dump({'processed_indices': list(processed_indices)}, f)

    def save_data(self, solar_data):
        df = pd.DataFrame(solar_data)
        df.to_csv(self.data_file, index=False, mode='a', header=not os.path.exists(self.data_file))

    def load_state(self):
        processed_indices = set()
        solar_data = []

        if os.path.exists(self.progress_file):
            with open(self.progress_file, 'r') as f:
                data = json.load(f)
                processed_indices = set(data['processed_indices'])

        if os.path.exists(self.data_file):
            solar_data = pd.read_csv(self.data_file).to_dict('records')

        return processed_indices, solar_data

    def load_grid_points(self):
        if os.path.exists(self.grid_points_file):
            return gpd.read_file(self.grid_points_file)
        return None

class ProgressTracker:
    def __init__(self, total_points, checkpoint_manager, processed_indices=None):
        self.total_points = total_points
        self.processed_points = len(processed_indices) if processed_indices else 0
        self.start_time = time.time()
        self.checkpoint_interval = 500
        self.last_checkpoint = self.processed_points
        self.checkpoint_manager = checkpoint_manager
        self.processed_indices = processed_indices if processed_indices else set()

    def update(self, batch_indices, batch_data):
        self.processed_indices.update(batch_indices)
        self.processed_points = len(self.processed_indices)
        
        if self.processed_points - self.last_checkpoint >= self.checkpoint_interval:
            self.save_checkpoint(batch_data)
            self.last_checkpoint = self.processed_points
        
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

    def save_checkpoint(self, batch_data):
        self.checkpoint_manager.save_progress(self.processed_indices)
        self.checkpoint_manager.save_data(batch_data)
        logger.info(f"Checkpoint saved at {self.processed_points} points")

async def process_grid_points(grid_points, checkpoint_manager, batch_size=50):
    processed_indices, solar_data = checkpoint_manager.load_state()
    progress_tracker = ProgressTracker(len(grid_points), checkpoint_manager, processed_indices)
    cache = dc.Cache('nasa_power_cache')
    
    async with aiohttp.ClientSession() as session:
        remaining_indices = set(range(len(grid_points))) - processed_indices
        
        for i in range(0, len(remaining_indices), batch_size):
            batch_indices = list(remaining_indices)[i:i+batch_size]
            batch = grid_points.iloc[batch_indices]
            tasks = [fetch_solar_data(session, point.y, point.x, cache) for point in batch.geometry]
            results = await asyncio.gather(*tasks)
            valid_results = [r for r in results if r is not None]
            
            if valid_results:
                progress_tracker.update(batch_indices, valid_results)
                solar_data.extend(valid_results)
    
    cache.close()
    return solar_data

def create_grid_points(geojson_path='india-soi.geojson', resolution=15000, checkpoint_manager=None):
    if checkpoint_manager:
        existing_points = checkpoint_manager.load_grid_points()
        if existing_points is not None:
            logger.info("Loading existing grid points from checkpoint")
            return existing_points
    
    try:
        india = gpd.read_file(geojson_path)
        if india.crs != 'EPSG:4326':
            india = india.to_crs('EPSG:4326')
        
        india_geom = india.geometry.buffer(0).unary_union
        bounds = india_geom.bounds
        
        mid_lat = (bounds[1] + bounds[3]) / 2
        deg_resolution = resolution / 111000
        lon_resolution = deg_resolution / np.cos(np.radians(mid_lat))
        
        lons = np.arange(bounds[0], bounds[2], lon_resolution)
        lats = np.arange(bounds[1], bounds[3], deg_resolution)
        
        valid_points = []
        total_points = len(lons) * len(lats)
        processed = 0
        sindex = india.sindex  # Spatial index for faster contains checks
        
        for lon in lons:
            for lat in lats:
                point = Point(lon, lat)
                processed += 1
                possible_matches_index = list(sindex.intersection(point.bounds))
                possible_matches = india.iloc[possible_matches_index]
                if any(possible_matches.contains(point)):
                    valid_points.append(point)
                
                if processed % 5000 == 0:
                    logger.info(
                        f"Processed {processed}/{total_points} points "
                        f"({(processed/total_points)*100:.1f}%) - "
                        f"Valid points so far: {len(valid_points)}"
                    )
        
        if not valid_points:
            raise ValueError("No valid points found within boundary")
        
        points_gdf = gpd.GeoDataFrame(geometry=valid_points, crs='EPSG:4326')
        points_gdf['latitude'] = points_gdf.geometry.y
        points_gdf['longitude'] = points_gdf.geometry.x
        
        if checkpoint_manager:
            checkpoint_manager.save_grid_points(points_gdf)
        
        return points_gdf
        
    except Exception as e:
        logger.error(f"Error in create_grid_points: {e}")
        raise

async def main():
    try:
        checkpoint_manager = CheckpointManager()
        grid_points = create_grid_points(resolution=10000, checkpoint_manager=checkpoint_manager)
        solar_data = await process_grid_points(grid_points, checkpoint_manager)
        
        # Combine checkpoint data with final results
        result_df = pd.DataFrame(solar_data)
        output_file = 'india_solar_data.csv'
        result_df.to_csv(output_file, index=False)
        logger.info(f"Saved complete solar data to {output_file}")
        
    except Exception as e:
        logger.error(f"Error in main: {e}")

if __name__ == "__main__":
    asyncio.run(main())