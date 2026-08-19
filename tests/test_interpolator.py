"""Unit tests for spatial interpolator and boundary rasterization."""

import os
import pandas as pd
import pytest
from raydium.interpolator import SpatialInterpolator


def test_spatial_interpolator():
    geojson = "india-soi.geojson"
    if not os.path.exists(geojson):
        pytest.skip("Boundary GeoJSON not available")

    interpolator = SpatialInterpolator(geojson_path=geojson)
    df = pd.DataFrame([
        {"latitude": 27.0, "longitude": 71.0, "suitability_score": 92.0},
        {"latitude": 15.0, "longitude": 78.0, "suitability_score": 85.0},
        {"latitude": 22.0, "longitude": 80.0, "suitability_score": 75.0},
        {"latitude": 28.0, "longitude": 85.0, "suitability_score": 68.0},
        {"latitude": 10.0, "longitude": 77.0, "suitability_score": 70.0},
    ])

    result = interpolator.interpolate_surface(df, value_column="suitability_score", grid_resolution=100)
    assert "raster" in result
    assert "transform" in result
    assert "vmin" in result
    assert "vmax" in result
    assert result["raster"].shape[1] == 100
