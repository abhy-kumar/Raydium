"""Legacy compatibility wrapper for solar data visualization.

Redirects to the optimized raydium.interpolator and raydium.visualizer engines.
"""

import argparse
import logging
import os
import sys
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))

from raydium.interpolator import SpatialInterpolator
from raydium.suitability import calculate_suitability
from raydium.visualizer import MapVisualizer

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.FileHandler("visualization.log"), logging.StreamHandler()],
)
logger = logging.getLogger(__name__)


def create_solar_map(solar_data_path="india_solar_data.csv", geojson_path="india-soi.geojson"):
    if not os.path.exists(solar_data_path):
        logger.error(f"Solar data not found at {solar_data_path}. Please collect data first.")
        return

    logger.info(f"Loading data from {solar_data_path}...")
    df = pd.read_csv(solar_data_path)
    if "suitability_score" not in df.columns:
        df = calculate_suitability(df)

    logger.info("Performing 2D spatial interpolation...")
    interpolator = SpatialInterpolator(geojson_path=geojson_path)
    val_col = "suitability_score" if "suitability_score" in df.columns else "potential"
    raster_dict = interpolator.interpolate_surface(df, value_column=val_col, grid_resolution=500)

    logger.info("Generating high-resolution cartographic map (solar_potential_high_res.png)...")
    visualizer = MapVisualizer(geojson_path=geojson_path)
    visualizer.render_static_map(raster_dict, output_image="solar_potential_high_res.png", dpi=300)

    logger.info("Generating interactive web dashboard (index.html & india_solar_potential.html)...")
    visualizer.render_interactive_dashboard(df, raster_dict, output_html="index.html")
    visualizer.render_interactive_dashboard(df, raster_dict, output_html="india_solar_potential.html")

    logger.info("Visualization pipeline completed successfully!")


def main():
    parser = argparse.ArgumentParser(description="Visualize India solar potential and plant suitability.")
    parser.add_argument("--data", type=str, default="india_solar_data.csv", help="Path to solar data CSV.")
    parser.add_argument("--geojson", type=str, default="india-soi.geojson", help="Path to boundary GeoJSON.")
    args = parser.parse_args()

    create_solar_map(solar_data_path=args.data, geojson_path=args.geojson)


if __name__ == "__main__":
    main()
