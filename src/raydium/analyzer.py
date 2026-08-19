"""Solar energy analytics, capacity estimation, and economic yield assessment."""

import logging
from typing import Dict, List, Optional
import numpy as np
import pandas as pd

from raydium.models import SUITABILITY_TIERS, MEGA_SOLAR_PARKS

logger = logging.getLogger(__name__)


class SolarAnalyzer:
    """Performs statistical analysis, PV capacity sizing, and carbon offset calculations."""

    # India CEA standard grid emission factor: 0.82 kg CO2 / kWh
    GRID_EMISSION_FACTOR_KG_KWH = 0.82
    # Standard utility solar land requirement: ~4.5 acres per MWp
    ACRES_PER_MW = 4.5
    # Standard performance ratio (PR)
    PERFORMANCE_RATIO = 0.78

    @classmethod
    def estimate_plant_yield(
        cls,
        ghi_daily: float,
        land_acres: float = 100.0,
        tariff_inr_kwh: float = 2.60,
    ) -> Dict:
        """Estimate solar power plant capacity, annual yield, revenue, and carbon reduction.

        Args:
            ghi_daily: Daily Global Horizontal Irradiance (kWh/m²/day).
            land_acres: Total land area in acres.
            tariff_inr_kwh: Power Purchase Agreement (PPA) tariff in INR per kWh.

        Returns:
            Dictionary with plant sizing and economic projections.
        """
        # Installed Capacity
        capacity_mw = round(land_acres / cls.ACRES_PER_MW, 2)
        capacity_kw = capacity_mw * 1000.0

        # Annual specific yield (kWh / kWp / year) = GHI_annual * PR
        annual_ghi = ghi_daily * 365.0
        specific_yield_kwh_kwp = annual_ghi * cls.PERFORMANCE_RATIO

        # Total annual generation in MWh and GWh
        annual_generation_kwh = capacity_kw * specific_yield_kwh_kwp
        annual_generation_mwh = round(annual_generation_kwh / 1000.0, 1)
        annual_generation_gwh = round(annual_generation_mwh / 1000.0, 3)

        # Capacity Utilization Factor (CUF %) = (Annual Generation) / (Capacity * 8760) * 100
        cuf_percent = round((annual_generation_kwh / (capacity_kw * 8760.0)) * 100.0, 2)

        # Economic Revenue
        annual_revenue_cr_inr = round((annual_generation_kwh * tariff_inr_kwh) / 10000000.0, 2)

        # Carbon Offset
        annual_co2_offset_tonnes = round((annual_generation_kwh * cls.GRID_EMISSION_FACTOR_KG_KWH) / 1000.0, 1)

        return {
            "land_acres": land_acres,
            "capacity_mw": capacity_mw,
            "cuf_percent": cuf_percent,
            "specific_yield_kwh_kwp": round(specific_yield_kwh_kwp, 1),
            "annual_generation_mwh": annual_generation_mwh,
            "annual_generation_gwh": annual_generation_gwh,
            "annual_revenue_cr_inr": annual_revenue_cr_inr,
            "annual_co2_offset_tonnes": annual_co2_offset_tonnes,
        }

    @classmethod
    def generate_summary_report(cls, df: pd.DataFrame) -> Dict:
        """Compute nationwide statistical and suitability summary."""
        val_col = "ghi_daily" if "ghi_daily" in df.columns else ("potential" if "potential" in df.columns else None)
        if val_col is None:
            raise KeyError("DataFrame missing solar potential column.")

        ghi_values = df[val_col].dropna().values
        suit_col = "suitability_score" if "suitability_score" in df.columns else None
        suit_values = df[suit_col].dropna().values if suit_col else None

        total_points = len(df)

        # Tier breakdown
        tier_counts = {}
        if "suitability_tier" in df.columns:
            counts = df["suitability_tier"].value_counts().to_dict()
            for tier_name in SUITABILITY_TIERS.keys():
                count = counts.get(tier_name, 0)
                pct = round((count / total_points) * 100.0, 1) if total_points > 0 else 0
                tier_counts[tier_name] = {"count": count, "percentage": pct}

        # Prime solar locations (Top 10 highest potential points)
        top_prime_df = df.sort_values(by=val_col, ascending=False).head(10)
        top_prime_locations = [
            {
                "latitude": round(r["latitude"], 4),
                "longitude": round(r["longitude"], 4),
                "ghi_daily": round(r[val_col], 2),
                "suitability_score": round(r.get("suitability_score", 0), 1),
                "region": r.get("region", "India"),
            }
            for _, r in top_prime_df.iterrows()
        ]

        # Overall National Solar Potential estimate (assuming 1% land utilization)
        # India land area ~ 3.287 million sq km. 1% = 32,870 sq km.
        avg_ghi = float(np.mean(ghi_values))
        est_gw_capacity = round(32870.0 * 50.0, 0)  # ~50 MW / sq km
        est_annual_twh = round((est_gw_capacity * 1000.0 * avg_ghi * 365.0 * 0.78) / 1000000.0, 1)

        summary = {
            "total_sampled_points": total_points,
            "solar_resource": {
                "mean_daily_ghi": round(avg_ghi, 2),
                "median_daily_ghi": round(float(np.median(ghi_values)), 2),
                "min_daily_ghi": round(float(np.min(ghi_values)), 2),
                "max_daily_ghi": round(float(np.max(ghi_values)), 2),
                "std_dev_ghi": round(float(np.std(ghi_values)), 2),
                "p10_ghi": round(float(np.percentile(ghi_values, 10)), 2),
                "p90_ghi": round(float(np.percentile(ghi_values, 90)), 2),
                "mean_annual_ghi": round(avg_ghi * 365.0, 1),
            },
            "suitability_index": {
                "mean_score": round(float(np.mean(suit_values)), 1) if suit_values is not None else None,
                "max_score": round(float(np.max(suit_values)), 1) if suit_values is not None else None,
                "tier_breakdown": tier_counts,
            },
            "national_potential_projection": {
                "assumed_land_utilization_percent": 1.0,
                "estimated_potential_gw": est_gw_capacity,
                "estimated_annual_twh": est_annual_twh,
                "annual_co2_abatement_million_tonnes": round((est_annual_twh * 1000000000.0 * 0.82) / 1000000000.0, 1),
            },
            "mega_solar_parks_count": len(MEGA_SOLAR_PARKS),
            "top_prime_locations": top_prime_locations,
        }

        return summary
