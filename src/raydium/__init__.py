"""Raydium: Solar Potential Analysis and Ideal Solar Plant Siting for India."""

__version__ = "2.0.0"
__author__ = "abhy-kumar"

from raydium.models import (
    SolarPoint,
    SolarPark,
    MEGA_SOLAR_PARKS,
    REGIONAL_BOUNDS,
    SUITABILITY_TIERS,
)
from raydium.grid import generate_india_grid
from raydium.collector import NASADataCollector
from raydium.suitability import calculate_suitability
from raydium.interpolator import SpatialInterpolator
from raydium.analyzer import SolarAnalyzer
from raydium.visualizer import MapVisualizer

__all__ = [
    "SolarPoint",
    "SolarPark",
    "MEGA_SOLAR_PARKS",
    "REGIONAL_BOUNDS",
    "SUITABILITY_TIERS",
    "generate_india_grid",
    "NASADataCollector",
    "calculate_suitability",
    "SpatialInterpolator",
    "SolarAnalyzer",
    "MapVisualizer",
]
