"""Unit tests for SolarAnalyzer and plant sizing calculations."""

import pandas as pd
from raydium.analyzer import SolarAnalyzer


def test_estimate_plant_yield():
    res = SolarAnalyzer.estimate_plant_yield(ghi_daily=6.0, land_acres=100.0, tariff_inr_kwh=2.60)
    assert res["capacity_mw"] > 20.0
    assert res["cuf_percent"] > 18.0
    assert res["annual_generation_gwh"] > 30.0
    assert res["annual_revenue_cr_inr"] > 5.0
    assert res["annual_co2_offset_tonnes"] > 20000.0


def test_generate_summary_report():
    df = pd.DataFrame([
        {"latitude": 27.0, "longitude": 71.0, "ghi_daily": 6.2, "suitability_score": 95.0, "suitability_tier": "Tier 1 - Prime Location"},
        {"latitude": 15.0, "longitude": 78.0, "ghi_daily": 5.4, "suitability_score": 80.0, "suitability_tier": "Tier 2 - Highly Suitable"},
        {"latitude": 22.0, "longitude": 80.0, "ghi_daily": 4.8, "suitability_score": 65.0, "suitability_tier": "Tier 3 - Moderately Suitable"},
    ])
    summary = SolarAnalyzer.generate_summary_report(df)
    assert summary["total_sampled_points"] == 3
    assert summary["solar_resource"]["mean_daily_ghi"] > 5.0
    assert "Tier 1 - Prime Location" in summary["suitability_index"]["tier_breakdown"]
