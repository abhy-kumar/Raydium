"""Production asynchronous NASA POWER API data collector with rate-limiting and caching."""

import asyncio
import json
import logging
import os
import random
import time
from typing import Any, Dict, List, Optional, Set, Tuple

import aiohttp
import diskcache as dc
import numpy as np
import pandas as pd

from raydium.models import SolarPoint

logger = logging.getLogger(__name__)


class TokenBucketRateLimiter:
    """Async Token Bucket rate limiter allowing smooth bursts and sustained rate compliance."""

    def __init__(self, rate: float = 10.0, capacity: float = 20.0):
        """Args:
            rate: Sustained tokens added per second.
            capacity: Maximum token burst bucket size.
        """
        self.rate = rate
        self.capacity = capacity
        self.tokens = capacity
        self.last_update = time.time()
        self.lock = asyncio.Lock()

    async def acquire(self, tokens: float = 1.0) -> None:
        """Wait until enough tokens are available."""
        async with self.lock:
            while True:
                now = time.time()
                elapsed = now - self.last_update
                self.last_update = now
                self.tokens = min(self.capacity, self.tokens + elapsed * self.rate)

                if self.tokens >= tokens:
                    self.tokens -= tokens
                    return

                needed = tokens - self.tokens
                wait_time = needed / self.rate
                await asyncio.sleep(wait_time)


class NASADataCollector:
    """High-throughput asynchronous client for NASA POWER Solar API."""

    CLIMATOLOGY_URL = "https://power.larc.nasa.gov/api/temporal/climatology/point"
    DAILY_URL = "https://power.larc.nasa.gov/api/temporal/daily/point"

    def __init__(
        self,
        cache_dir: str = "nasa_power_cache",
        checkpoint_dir: str = "checkpoints",
        max_rate: float = 8.0,
        max_concurrency: int = 15,
        max_retries: int = 4,
        timeout_seconds: float = 25.0,
    ):
        self.cache_dir = cache_dir
        self.checkpoint_dir = checkpoint_dir
        os.makedirs(cache_dir, exist_ok=True)
        os.makedirs(checkpoint_dir, exist_ok=True)

        self.cache = dc.Cache(cache_dir)
        self.rate_limiter = TokenBucketRateLimiter(rate=max_rate, capacity=max_rate * 2)
        self.semaphore = asyncio.Semaphore(max_concurrency)
        self.max_retries = max_retries
        self.timeout = aiohttp.ClientTimeout(total=timeout_seconds)

        self.progress_file = os.path.join(checkpoint_dir, "progress.json")
        self.checkpoint_csv = os.path.join(checkpoint_dir, "checkpoint_solar_data.csv")

    def _get_cache_key(self, lat: float, lon: float, mode: str) -> str:
        return f"nasa_{mode}_{lat:.3f}_{lon:.3f}"

    def simulate_solar_data(self, lat: float, lon: float) -> SolarPoint:
        """Simulate realistic solar resource measurements based on India's climate zones.
        
        Used for offline mode, testing, and rapid prototyping without hitting NASA servers.
        """
        # Physics-based baseline approximation for India:
        # High in NW (Thar Desert 68-75E, 24-29N), High in Deccan Plateau, Moderate in Gangetic plains, Low in NE/High Himalayas
        base_ghi = 5.2

        # Latitude effect (higher irradiance in 12-28N zone)
        lat_factor = -0.04 * abs(lat - 22.0)

        # Desert bonus for Western India (Rajasthan / Gujarat)
        if 69.0 <= lon <= 76.0 and 23.0 <= lat <= 29.5:
            desert_bonus = 0.9 + 0.3 * (1.0 - (abs(lon - 71.5) / 5.0))
        else:
            desert_bonus = 0.0

        # Southern plateau bonus (high solar days)
        if 74.0 <= lon <= 79.0 and 11.0 <= lat <= 18.0:
            plateau_bonus = 0.45
        else:
            plateau_bonus = 0.0

        # Cloudiness penalty for North-East & High Himalayas
        ne_penalty = 0.0
        himalaya_cloud_penalty = 0.0
        if lon > 88.0 and lat < 28.0:
            ne_penalty = -0.7
        elif lat > 32.0 and lon < 77.0:
            himalaya_cloud_penalty = -0.5
        elif lat > 32.0 and lon >= 77.0:
            # Ladakh high-altitude clear sky bonus
            himalaya_cloud_penalty = 0.6

        # Ambient temperature model
        ambient_temp = 28.0 - 0.5 * (lat - 15.0) - (4.0 if lat > 32.0 else 0.0)
        ambient_temp += random.uniform(-1.0, 1.0)

        ghi_daily = base_ghi + lat_factor + desert_bonus + plateau_bonus + ne_penalty + himalaya_cloud_penalty
        ghi_daily += random.uniform(-0.1, 0.1)
        ghi_daily = float(np.clip(ghi_daily, 2.5, 6.8))

        dni_daily = float(np.clip(ghi_daily * 1.08 + random.uniform(-0.15, 0.15), 2.2, 7.5))
        ghi_annual = round(ghi_daily * 365.0, 1)

        return SolarPoint(
            latitude=lat,
            longitude=lon,
            ghi_daily=ghi_daily,
            dni_daily=dni_daily,
            ghi_annual=ghi_annual,
            temp_ambient=ambient_temp,
        )

    async def fetch_point_data(
        self,
        session: aiohttp.ClientSession,
        lat: float,
        lon: float,
        mode: str = "climatology",
    ) -> Optional[SolarPoint]:
        """Fetch solar irradiance for a single coordinate point with caching and retry."""
        cache_key = self._get_cache_key(lat, lon, mode)
        if cache_key in self.cache:
            cached_data = self.cache[cache_key]
            return SolarPoint(**cached_data)

        params = {
            "parameters": "ALLSKY_SFC_SW_DWN,T2M",
            "community": "RE",
            "longitude": f"{lon:.4f}",
            "latitude": f"{lat:.4f}",
            "format": "JSON",
        }

        url = self.CLIMATOLOGY_URL if mode == "climatology" else self.DAILY_URL

        for attempt in range(1, self.max_retries + 1):
            try:
                await self.rate_limiter.acquire()
                async with self.semaphore:
                    async with session.get(url, params=params, timeout=self.timeout) as resp:
                        if resp.status == 200:
                            data = await resp.json()
                            params_data = data.get("properties", {}).get("parameter", {})

                            ghi_dict = params_data.get("ALLSKY_SFC_SW_DWN", {})
                            temp_dict = params_data.get("T2M", {})

                            if mode == "climatology":
                                ghi_daily = float(ghi_dict.get("ANN", 0.0))
                                temp_ambient = float(temp_dict.get("ANN", 25.0))
                            else:
                                vals = [v for v in ghi_dict.values() if v > 0]
                                ghi_daily = float(sum(vals) / len(vals)) if vals else 0.0
                                t_vals = [v for v in temp_dict.values() if v is not None]
                                temp_ambient = float(sum(t_vals) / len(t_vals)) if t_vals else 25.0

                            if ghi_daily <= 0:
                                return None

                            point = SolarPoint(
                                latitude=lat,
                                longitude=lon,
                                ghi_daily=ghi_daily,
                                dni_daily=ghi_daily * 1.05,
                                ghi_annual=round(ghi_daily * 365.0, 1),
                                temp_ambient=temp_ambient,
                            )

                            # Save to disk cache
                            self.cache[cache_key] = point.__dict__
                            return point

                        elif resp.status == 429:
                            # Rate limit encountered: exponential backoff with jitter
                            backoff = (2 ** attempt) + random.uniform(0.5, 2.0)
                            logger.warning(f"Rate limited (429) at ({lat}, {lon}). Backing off for {backoff:.1f}s...")
                            await asyncio.sleep(backoff)

                        elif resp.status in (500, 502, 503, 504):
                            backoff = (1.5 ** attempt) + random.uniform(0.2, 1.0)
                            logger.warning(f"Server error {resp.status} at ({lat}, {lon}). Retrying in {backoff:.1f}s...")
                            await asyncio.sleep(backoff)

                        else:
                            logger.error(f"Failed ({resp.status}) at ({lat}, {lon})")
                            return None

            except (aiohttp.ClientError, asyncio.TimeoutError) as err:
                if attempt == self.max_retries:
                    logger.error(f"Network error at ({lat}, {lon}) after {attempt} attempts: {err}")
                    return None
                await asyncio.sleep(1.0 * attempt + random.uniform(0.1, 0.5))

        return None

    def load_checkpoint(self) -> Tuple[Set[Tuple[float, float]], List[Dict]]:
        """Load already processed points and collected records."""
        processed_coords: Set[Tuple[float, float]] = set()
        solar_records: List[Dict] = []

        if os.path.exists(self.progress_file):
            try:
                with open(self.progress_file, "r") as f:
                    data = json.load(f)
                    processed_coords = {tuple(c) for c in data.get("processed_coords", [])}
            except Exception as e:
                logger.warning(f"Could not read progress file: {e}")

        if os.path.exists(self.checkpoint_csv):
            try:
                df = pd.read_csv(self.checkpoint_csv)
                solar_records = df.to_dict("records")
            except Exception as e:
                logger.warning(f"Could not read checkpoint CSV: {e}")

        return processed_coords, solar_records

    def save_checkpoint(self, processed_coords: Set[Tuple[float, float]], new_records: List[Dict]) -> None:
        """Atomically persist progress and newly collected data."""
        # Save JSON progress
        tmp_json = self.progress_file + ".tmp"
        with open(tmp_json, "w") as f:
            json.dump({"processed_coords": list(processed_coords)}, f)
        if os.path.exists(tmp_json):
            os.replace(tmp_json, self.progress_file)

        # Append CSV data
        if new_records:
            df = pd.DataFrame(new_records)
            header = not os.path.exists(self.checkpoint_csv)
            df.to_csv(self.checkpoint_csv, mode="a", index=False, header=header)

    async def collect(
        self,
        coordinates: List[Tuple[float, float]],
        simulate: bool = False,
        batch_size: int = 100,
        mode: str = "climatology",
    ) -> List[Dict]:
        """Collect solar irradiance for all given coordinate points.

        Args:
            coordinates: List of (latitude, longitude) tuples.
            simulate: If True, uses offline climate physics simulation.
            batch_size: Number of points to process before checkpointing.
            mode: 'climatology' or 'daily'.

        Returns:
            List of dictionary records with solar metrics.
        """
        if simulate:
            logger.info(f"Simulating solar potential for {len(coordinates):,} points...")
            results = [self.simulate_solar_data(lat, lon).to_dict() for lat, lon in coordinates]
            return results

        processed_coords, solar_records = self.load_checkpoint()
        logger.info(f"Checkpoint loaded: {len(processed_coords):,} already collected points.")

        remaining = [(lat, lon) for lat, lon in coordinates if (round(lat, 4), round(lon, 4)) not in processed_coords]
        logger.info(f"Total points: {len(coordinates):,}. Remaining to fetch: {len(remaining):,}.")

        if not remaining:
            return solar_records

        connector = aiohttp.TCPConnector(limit=50, keepalive_timeout=30)
        async with aiohttp.ClientSession(connector=connector) as session:
            for i in range(0, len(remaining), batch_size):
                batch = remaining[i : i + batch_size]
                tasks = [self.fetch_point_data(session, lat, lon, mode=mode) for lat, lon in batch]
                batch_results = await asyncio.gather(*tasks)

                new_records = []
                for (lat, lon), res in zip(batch, batch_results):
                    coord_key = (round(lat, 4), round(lon, 4))
                    processed_coords.add(coord_key)
                    if res is not None:
                        d = res.to_dict()
                        new_records.append(d)
                        solar_records.append(d)

                self.save_checkpoint(processed_coords, new_records)
                logger.info(
                    f"Progress: {len(processed_coords):,}/{len(coordinates):,} "
                    f"({len(processed_coords)/len(coordinates)*100:.1f}%) | "
                    f"Batch valid: {len(new_records)}/{len(batch)}"
                )

        return solar_records

    def close(self):
        """Close cache resources."""
        self.cache.close()
