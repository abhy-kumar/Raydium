"""High-precision 2D spatial interpolation and boundary-masked raster generation."""

import logging
import os
from typing import Dict, Optional, Tuple

import geopandas as gpd
import numpy as np
import pandas as pd
from rasterio.features import geometry_mask
from rasterio.transform import from_bounds
import scipy.interpolate as interp

logger = logging.getLogger(__name__)


class SpatialInterpolator:
    """Interpolates discrete solar/suitability point samples into continuous high-resolution rasters."""

    def __init__(
        self,
        geojson_path: str = "india-soi.geojson",
        target_crs: str = "EPSG:4326",
    ):
        self.geojson_path = geojson_path
        self.target_crs = target_crs

        if not os.path.exists(geojson_path):
            raise FileNotFoundError(f"GeoJSON boundary file not found at: {geojson_path}")

        self.india_gdf = gpd.read_file(geojson_path)
        if self.india_gdf.crs != target_crs:
            self.india_gdf = self.india_gdf.to_crs(target_crs)

        self.india_geom = self.india_gdf.geometry.union_all()
        self.bounds = self.india_gdf.total_bounds  # [minx, miny, maxx, maxy]

    def interpolate_surface(
        self,
        df: pd.DataFrame,
        value_column: str = "suitability_score",
        grid_resolution: int = 500,
        method: str = "linear",
    ) -> Dict:
        """Interpolate point values across the entire boundary bounding box and mask to India outline.

        Args:
            df: DataFrame containing 'latitude', 'longitude', and the target value_column.
            value_column: Column name to interpolate (e.g., 'suitability_score', 'ghi_daily', 'potential').
            grid_resolution: Number of raster grid cells along max dimension (e.g. 500x500).
            method: 'linear', 'cubic', or 'nearest'.

        Returns:
            Dictionary with masked raster, transform, bounds, vmin, vmax, and raw statistics.
        """
        if value_column not in df.columns:
            if "potential" in df.columns:
                value_column = "potential"
            else:
                raise KeyError(f"Column '{value_column}' not found in dataframe.")

        pts_x = df["longitude"].values
        pts_y = df["latitude"].values
        vals = df[value_column].values

        if len(vals) < 4:
            raise ValueError(f"At least 4 points required for spatial interpolation, got {len(vals)}")

        minx, miny, maxx, maxy = self.bounds

        # Calculate raster dimensions preserving aspect ratio
        aspect = (maxy - miny) / (maxx - minx)
        width = grid_resolution
        height = int(grid_resolution * aspect)

        logger.info(f"Interpolating '{value_column}' ({len(vals):,} points) onto {width}x{height} raster grid ({method})...")

        # Generate 2D target grid coordinates
        grid_x, grid_y = np.mgrid[
            minx : maxx : complex(0, width),
            maxy : miny : complex(0, height)  # Top to bottom for raster convention
        ]

        # Primary interpolation (linear or cubic)
        points = np.column_stack((pts_x, pts_y))
        grid_z = interp.griddata(points, vals, (grid_x, grid_y), method=method)

        # Fallback with nearest neighbor for boundary edges and coastal peninsulas
        if np.isnan(grid_z).any():
            grid_z_near = interp.griddata(points, vals, (grid_x, grid_y), method="nearest")
            nan_mask = np.isnan(grid_z)
            grid_z[nan_mask] = grid_z_near[nan_mask]

        # Transpose to (height, width) for standard image/raster representation
        raster_data = grid_z.T

        # Create affine transform from bounds
        transform = from_bounds(minx, miny, maxx, maxy, width, height)

        # Apply exact geometry mask using the official boundary polygon
        logger.info("Applying official Survey of India boundary mask...")
        features_geom = [feature["geometry"] for feature in self.india_gdf.__geo_interface__["features"]]
        mask = geometry_mask(features_geom, out_shape=(height, width), transform=transform, invert=True)

        masked_raster = np.ma.masked_array(raster_data, ~mask)
        valid_values = masked_raster.compressed()

        if valid_values.size == 0:
            raise ValueError("No valid raster data inside boundary mask.")

        vmin = float(np.percentile(valid_values, 1))
        vmax = float(np.percentile(valid_values, 99))

        logger.info(f"Raster generated: {width}x{height} | Valid cells: {valid_values.size:,} | Range: [{vmin:.2f}, {vmax:.2f}]")

        return {
            "raster": masked_raster,
            "raw_raster": raster_data,
            "mask": mask,
            "transform": transform,
            "bounds": [minx, miny, maxx, maxy],
            "width": width,
            "height": height,
            "vmin": vmin,
            "vmax": vmax,
            "mean": float(np.mean(valid_values)),
            "std": float(np.std(valid_values)),
        }
