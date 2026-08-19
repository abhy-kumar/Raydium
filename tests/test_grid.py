"""Unit tests for spatial grid generation."""

import os
import pytest
from raydium.grid import generate_india_grid


def test_generate_india_grid_all():
    geojson = "india-soi.geojson"
    if not os.path.exists(geojson):
        pytest.skip("GeoJSON boundary not present")

    gdf = generate_india_grid(geojson_path=geojson, resolution_deg=1.0, region="all")
    assert len(gdf) > 100
    assert "latitude" in gdf.columns
    assert "longitude" in gdf.columns
    assert "geometry" in gdf.columns
    assert gdf.crs.to_string() == "EPSG:4326"


def test_generate_india_grid_regional():
    geojson = "india-soi.geojson"
    if not os.path.exists(geojson):
        pytest.skip("GeoJSON boundary not present")

    gdf_rajasthan = generate_india_grid(geojson_path=geojson, resolution_deg=0.5, region="rajasthan_thar")
    assert len(gdf_rajasthan) > 10
    # All points in Thar region should have longitude between 69 and 76
    assert (gdf_rajasthan["longitude"] >= 69.0).all()
    assert (gdf_rajasthan["longitude"] <= 76.5).all()
