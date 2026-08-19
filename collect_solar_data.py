"""Legacy compatibility wrapper for solar data collection.

Redirects to the optimized raydium.collector and raydium.grid engines.
"""

import argparse
import asyncio
import logging
import os
import sys

# Ensure src is on python path if run directly
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))

from raydium.collector import NASADataCollector
from raydium.grid import generate_india_grid
from raydium.suitability import calculate_suitability
import pandas as pd

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.FileHandler("solar_data_collection.log"), logging.StreamHandler()],
)
logger = logging.getLogger(__name__)


def parse_args():
    parser = argparse.ArgumentParser(description="Collect solar potential data for India.")
    parser.add_argument("--region", type=str, default="all", help="Region to collect data for (north, south, east, west, all).")
    parser.add_argument("--resolution", type=float, default=0.25, help="Grid resolution in degrees.")
    parser.add_argument("--simulate", action="store_true", help="Use offline climate simulation.")
    parser.add_argument("--output", type=str, default="india_solar_data.csv", help="Output CSV path.")
    return parser.parse_args()


async def main_async():
    args = parse_args()
    logger.info(f"Starting solar data collection for region='{args.region}' at resolution={args.resolution} deg...")

    output_csv = args.output
    if args.region != "all" and output_csv == "india_solar_data.csv":
        output_csv = f"india_solar_data_{args.region}.csv"

    grid_gdf = generate_india_grid(geojson_path="india-soi.geojson", resolution_deg=args.resolution, region=args.region)
    coords = list(zip(grid_gdf["latitude"], grid_gdf["longitude"]))

    collector = NASADataCollector()
    records = await collector.collect(coordinates=coords, simulate=args.simulate)
    collector.close()

    df = pd.DataFrame(records)
    df = calculate_suitability(df)
    df.to_csv(output_csv, index=False)
    logger.info(f"Saved {len(df):,} solar data records to {output_csv}")


def main():
    asyncio.run(main_async())


if __name__ == "__main__":
    main()