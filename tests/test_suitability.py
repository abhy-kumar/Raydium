"""Unit tests for Multi-Criteria Solar Plant Suitability Scoring."""

import pandas as pd
from raydium.suitability import compute_point_suitability, calculate_suitability


def test_compute_point_suitability_thar():
    # Thar Desert high irradiance & flat terrain
    res = compute_point_suitability(ghi_daily=6.2, temp_ambient=30.0, latitude=27.0, longitude=71.5)
    assert res["suitability_score"] >= 80.0
    assert "Tier 1" in res["suitability_tier"] or "Tier 2" in res["suitability_tier"]


def test_compute_point_suitability_himalayas():
    # Steep rugged high mountains with low radiation
    res = compute_point_suitability(ghi_daily=3.2, temp_ambient=10.0, latitude=34.0, longitude=75.0)
    assert res["suitability_score"] < 65.0


def test_calculate_suitability_dataframe():
    df = pd.DataFrame([
        {"latitude": 27.0, "longitude": 71.5, "ghi_daily": 6.2, "temp_ambient": 30.0},
        {"latitude": 13.0, "longitude": 80.0, "ghi_daily": 4.8, "temp_ambient": 28.0},
    ])
    scored_df = calculate_suitability(df)
    assert "suitability_score" in scored_df.columns
    assert "suitability_tier" in scored_df.columns
    assert len(scored_df) == 2
