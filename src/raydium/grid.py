"""High-performance vectorized spatial grid generation across India."""

import logging
import os
from typing import Optional, Tuple, Union
import geopandas as gpd
import numpy as np
import shapely

from raydium.models import REGIONAL_BOUNDS

logger = logging.getLogger(__name__)


def generate_india_grid(
    geojson_path: str = "india-soi.geojson",
    resolution_deg: float = 0.25,
    region: str = "all",
    custom_bbox: Optional[Tuple[float, float, float, float]] = None,
    output_path: Optional[str] = None,
) -> gpd.GeoDataFrame:
    """Generate a high-density, boundary-conforming spatial grid across India.

    Uses vectorized Shapely 2.0 C-level spatial containment (`contains_xy`)
    for sub-second generation of tens of thousands of candidate points.

    Args:
        geojson_path: Path to the official boundary GeoJSON.
        resolution_deg: Grid resolution in decimal degrees (~0.25° ≈ 25km, ~0.1° ≈ 10km).
        region: Regional filter ('all', 'north', 'south', 'west', 'east', 'central', 'northeast', 'rajasthan_thar', etc.).
        custom_bbox: Optional custom (min_lon, min_lat, max_lon, max_lat).
        output_path: Optional file path to persist the grid points (.geojson or .parquet).

    Returns:
        GeoDataFrame containing valid points within India's boundary.
    """
    if not os.path.exists(geojson_path):
        raise FileNotFoundError(f"Boundary GeoJSON not found at: {geojson_path}")

    logger.info(f"Loading boundary from {geojson_path}...")
    india = gpd.read_file(geojson_path)
    if india.crs != "EPSG:4326":
        india = india.to_crs("EPSG:4326")

    india_geom = india.geometry.union_all()
    bounds = india_geom.bounds  # (minx, miny, maxx, maxy)

    # Apply regional or custom bounding box clipping
    if custom_bbox:
        min_lon = max(bounds[0], custom_bbox[0])
        min_lat = max(bounds[1], custom_bbox[1])
        max_lon = min(bounds[2], custom_bbox[2])
        max_lat = min(bounds[3], custom_bbox[3])
    elif region.lower() in REGIONAL_BOUNDS and region.lower() != "all":
        reg_box = REGIONAL_BOUNDS[region.lower()]
        min_lon = max(bounds[0], reg_box[0])
        min_lat = max(bounds[1], reg_box[1])
        max_lon = min(bounds[2], reg_box[2])
        max_lat = min(bounds[3], reg_box[3])
    else:
        min_lon, min_lat, max_lon, max_lat = bounds

    logger.info(f"Generating candidate grid: lon [{min_lon:.2f}, {max_lon:.2f}], lat [{min_lat:.2f}, {max_lat:.2f}] at {resolution_deg}° step...")
    lons = np.arange(min_lon, max_lon + resolution_deg / 2.0, resolution_deg)
    lats = np.arange(min_lat, max_lat + resolution_deg / 2.0, resolution_deg)

    xx, yy = np.meshgrid(lons, lats)
    cand_x = xx.ravel()
    cand_y = yy.ravel()

    # Fast vectorized containment test with Shapely 2.0
    logger.info(f"Evaluating containment for {len(cand_x):,} candidate points...")
    valid_mask = shapely.contains_xy(india_geom, cand_x, cand_y)

    valid_lons = cand_x[valid_mask]
    valid_lats = cand_y[valid_mask]

    if len(valid_lons) == 0:
        raise ValueError(f"No grid points fell inside India boundary for region '{region}'.")

    logger.info(f"Successfully generated {len(valid_lons):,} valid grid points within India ({len(valid_lons)/len(cand_x)*100:.1f}% land coverage).")

    # Construct GeoDataFrame
    points_geometry = gpd.points_from_xy(valid_lons, valid_lats, crs="EPSG:4326")
    gdf = gpd.GeoDataFrame(
        {
            "latitude": np.round(valid_lats, 4),
            "longitude": np.round(valid_lons, 4),
            "region": region,
        },
        geometry=points_geometry,
        crs="EPSG:4326",
    )

    if output_path:
        os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
        if output_path.endswith(".parquet"):
            gdf.to_parquet(output_path)
        else:
            gdf.to_file(output_path, driver="GeoJSON")
        logger.info(f"Saved grid points to {output_path}")

    return gdf
