"""Unit tests for NASA POWER data collector and rate limiter."""

import asyncio
import os
import shutil
import pytest
from raydium.collector import NASADataCollector, TokenBucketRateLimiter
from raydium.models import SolarPoint


@pytest.mark.asyncio
async def test_token_bucket_rate_limiter():
    limiter = TokenBucketRateLimiter(rate=20.0, capacity=5.0)
    await limiter.acquire(1.0)
    assert limiter.tokens < 5.0


def test_collector_simulation():
    test_cache = "test_cache_temp"
    collector = NASADataCollector(cache_dir=test_cache, checkpoint_dir="test_checkpoints_temp")
    try:
        point = collector.simulate_solar_data(lat=27.0, lon=71.5)
        assert isinstance(point, SolarPoint)
        assert 5.0 <= point.ghi_daily <= 7.0
        assert point.ghi_annual > 1800
        assert "latitude" in point.to_dict()
    finally:
        collector.close()
        if os.path.exists(test_cache):
            shutil.rmtree(test_cache, ignore_errors=True)
        if os.path.exists("test_checkpoints_temp"):
            shutil.rmtree("test_checkpoints_temp", ignore_errors=True)


@pytest.mark.asyncio
async def test_collector_collect_simulate():
    collector = NASADataCollector(cache_dir="test_cache_temp2", checkpoint_dir="test_checkpoints_temp2")
    try:
        coords = [(26.0, 72.0), (15.0, 78.0), (28.0, 77.0)]
        results = await collector.collect(coordinates=coords, simulate=True)
        assert len(results) == 3
        assert results[0]["ghi_daily"] > 0
    finally:
        collector.close()
        if os.path.exists("test_cache_temp2"):
            shutil.rmtree("test_cache_temp2", ignore_errors=True)
        if os.path.exists("test_checkpoints_temp2"):
            shutil.rmtree("test_checkpoints_temp2", ignore_errors=True)
