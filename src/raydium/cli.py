"""Rich & Typer command-line interface for the Raydium platform."""

import asyncio
import http.server
import json
import logging
import os
import socketserver
import webbrowser
from typing import Optional

import pandas as pd
from rich.console import Console
from rich.panel import Panel
from rich.table import Table
import typer

from raydium import __version__
from raydium.analyzer import SolarAnalyzer
from raydium.collector import NASADataCollector
from raydium.grid import generate_india_grid
from raydium.interpolator import SpatialInterpolator
from raydium.models import MEGA_SOLAR_PARKS, REGIONAL_BOUNDS, SUITABILITY_TIERS
from raydium.suitability import calculate_suitability
from raydium.visualizer import MapVisualizer

app = typer.Typer(
    name="raydium",
    help="Raydium: Solar Potential Analysis and Ideal Solar Plant Siting for India.",
    add_completion=False,
)
console = Console()


def configure_logging(verbose: bool = False):
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(asctime)s [%(levelname)s] %(message)s",
        handlers=[logging.StreamHandler()],
    )


@app.command()
def info():
    """Display Raydium system information, version, and tracked solar parks."""
    console.print(
        Panel(
            f"[bold gold1]Raydium - Solar Siting Platform for India[/bold gold1]\n"
            f"[dim]Version: {__version__} | Author: abhy-kumar <abhiks177@gmail.com>[/dim]",
            border_style="gold1",
        )
    )

    table = Table(title="Major Tracked Mega Solar Parks in India", header_style="bold cyan")
    table.add_column("Solar Park", style="bold white")
    table.add_column("State", style="yellow")
    table.add_column("Capacity (MW)", justify="right", style="green")
    table.add_column("Status", style="magenta")
    table.add_column("Developer", style="dim")

    for p in MEGA_SOLAR_PARKS:
        table.add_row(p.name, p.state, f"{p.capacity_mw:,.0f} MW", p.status, p.developer)

    console.print(table)


@app.command()
def collect(
    geojson: str = typer.Option("india-soi.geojson", "--geojson", "-g", help="Path to boundary GeoJSON."),
    resolution: float = typer.Option(0.25, "--resolution", "-r", help="Grid resolution in degrees (~0.25° ≈ 25km, 0.1° ≈ 10km)."),
    region: str = typer.Option("all", "--region", help="Region filter (all, north, south, west, east, central, rajasthan_thar, etc.)."),
    simulate: bool = typer.Option(False, "--simulate", "-s", help="Use offline climate simulation for testing without API calls."),
    output: str = typer.Option("india_solar_data.csv", "--output", "-o", help="Output CSV path."),
    mode: str = typer.Option("climatology", "--mode", help="NASA POWER mode: 'climatology' (30-year avg) or 'daily'."),
    verbose: bool = typer.Option(False, "--verbose", "-v", help="Enable verbose logging."),
):
    """Generate spatial grid and collect solar insolation data from NASA POWER API."""
    configure_logging(verbose)
    console.print(f"[bold cyan]Starting solar data collection for region='{region}' at resolution={resolution}°...[/bold cyan]")

    grid_gdf = generate_india_grid(geojson_path=geojson, resolution_deg=resolution, region=region)
    coords = list(zip(grid_gdf["latitude"], grid_gdf["longitude"]))

    collector = NASADataCollector()
    solar_records = asyncio.run(collector.collect(coordinates=coords, simulate=simulate, mode=mode))
    collector.close()

    df = pd.DataFrame(solar_records)
    # Apply suitability scoring
    df = calculate_suitability(df)
    df.to_csv(output, index=False)

    console.print(f"[bold green][OK] Successfully collected and saved {len(df):,} solar data records to {output}[/bold green]")


@app.command()
def visualize(
    data: str = typer.Option("india_solar_data.csv", "--data", "-d", help="Path to collected solar data CSV."),
    geojson: str = typer.Option("india-soi.geojson", "--geojson", "-g", help="Path to boundary GeoJSON."),
    image_out: str = typer.Option("solar_potential_high_res.png", "--image-out", help="Output high-res PNG image path."),
    html_out: str = typer.Option("index.html", "--html-out", help="Output interactive HTML dashboard path."),
    resolution: int = typer.Option(500, "--grid-res", help="Interpolation raster resolution."),
    dpi: int = typer.Option(300, "--dpi", help="PNG Image DPI."),
    verbose: bool = typer.Option(False, "--verbose", "-v", help="Enable verbose logging."),
):
    """Interpolate solar potential surface and generate static PNG + interactive HTML dashboard."""
    configure_logging(verbose)
    if not os.path.exists(data):
        console.print(f"[bold red]Data file not found at '{data}'. Run 'raydium collect' first.[/bold red]")
        raise typer.Exit(1)

    df = pd.read_csv(data)
    if "suitability_score" not in df.columns:
        df = calculate_suitability(df)

    console.print("[bold cyan]Running 2D continuous spatial interpolation & boundary masking...[/bold cyan]")
    interpolator = SpatialInterpolator(geojson_path=geojson)
    raster_dict = interpolator.interpolate_surface(
        df,
        value_column="suitability_score" if "suitability_score" in df.columns else "potential",
        grid_resolution=resolution,
    )

    console.print("[bold cyan]Generating high-resolution cartographic map...[/bold cyan]")
    visualizer = MapVisualizer(geojson_path=geojson)
    visualizer.render_static_map(raster_dict, output_image=image_out, dpi=dpi)

    console.print("[bold cyan]Generating modern interactive web dashboard...[/bold cyan]")
    visualizer.render_interactive_dashboard(df, raster_dict, output_html=html_out)
    
    # Also save copy to india_solar_potential.html for legacy compatibility
    visualizer.render_interactive_dashboard(df, raster_dict, output_html="india_solar_potential.html")

    console.print(f"[bold green][OK] High-res Map created: {image_out}[/bold green]")
    console.print(f"[bold green][OK] Interactive Dashboard created: {html_out} and india_solar_potential.html[/bold green]")


@app.command()
def analyze(
    data: str = typer.Option("india_solar_data.csv", "--data", "-d", help="Path to solar data CSV."),
    json_out: Optional[str] = typer.Option(None, "--json-out", help="Optional path to write JSON analysis summary."),
):
    """Run comprehensive solar resource and economic analysis on collected data."""
    if not os.path.exists(data):
        console.print(f"[bold red]Data file not found at '{data}'. Run 'raydium collect' first.[/bold red]")
        raise typer.Exit(1)

    df = pd.read_csv(data)
    if "suitability_score" not in df.columns:
        df = calculate_suitability(df)

    summary = SolarAnalyzer.generate_summary_report(df)

    # Display Rich summary table
    table = Table(title="Raydium National Solar Resource & Suitability Report", header_style="bold gold1")
    table.add_column("Metric", style="white")
    table.add_column("Value", style="bold cyan")

    res = summary["solar_resource"]
    table.add_row("Total Sampled Grid Points", f"{summary['total_sampled_points']:,}")
    table.add_row("Mean Daily Irradiance (GHI)", f"{res['mean_daily_ghi']} kWh/m2/day")
    table.add_row("Mean Annual Insolation", f"{res['mean_annual_ghi']} kWh/m2/year")
    table.add_row("Max Peak Daily Irradiance", f"{res['max_daily_ghi']} kWh/m2/day")
    table.add_row("Min Daily Irradiance", f"{res['min_daily_ghi']} kWh/m2/day")
    table.add_row("Standard Deviation", f"{res['std_dev_ghi']} kWh/m2/day")
    table.add_row("90th Percentile Irradiance", f"{res['p90_ghi']} kWh/m2/day")

    proj = summary["national_potential_projection"]
    table.add_row("Estimated National Potential (1% land)", f"{proj['estimated_potential_gw']:,.0f} GW")
    table.add_row("Estimated Annual Generation", f"{proj['estimated_annual_twh']:,.1f} TWh/year")
    table.add_row("Annual CO2 Offset Potential", f"{proj['annual_co2_abatement_million_tonnes']:,.1f} Million Tonnes")

    console.print(table)

    # Suitability Breakdown table
    tier_table = Table(title="Solar Plant Siting Suitability Breakdown", header_style="bold green")
    tier_table.add_column("Suitability Tier", style="bold")
    tier_table.add_column("Points Count", justify="right", style="cyan")
    tier_table.add_column("Percentage (%)", justify="right", style="yellow")

    for tier_name, info in summary["suitability_index"]["tier_breakdown"].items():
        tier_table.add_row(tier_name, f"{info['count']:,}", f"{info['percentage']:.1f} %")

    console.print(tier_table)

    if json_out:
        with open(json_out, "w") as f:
            json.dump(summary, f, indent=2)
        console.print(f"[bold green][OK] Analysis exported to {json_out}[/bold green]")


@app.command()
def pipeline(
    geojson: str = typer.Option("india-soi.geojson", "--geojson", "-g", help="Path to boundary GeoJSON."),
    resolution: float = typer.Option(0.25, "--resolution", "-r", help="Grid resolution in degrees (~0.25 deg ≈ 25km)."),
    region: str = typer.Option("all", "--region", help="Region filter."),
    simulate: bool = typer.Option(False, "--simulate", "-s", help="Use offline climate simulation for rapid testing."),
    output_csv: str = typer.Option("india_solar_data.csv", "--csv-out", help="Output CSV path."),
    output_png: str = typer.Option("solar_potential_high_res.png", "--png-out", help="Output PNG map path."),
    output_html: str = typer.Option("index.html", "--html-out", help="Output HTML dashboard path."),
    verbose: bool = typer.Option(False, "--verbose", "-v", help="Enable verbose logging."),
):
    """Execute complete end-to-end pipeline: Grid -> Collect -> Suitability -> Interpolate -> Analyze -> Visualize."""
    configure_logging(verbose)
    console.print(
        Panel(
            "[bold gold1]Starting Raydium End-to-End Solar Intelligence Pipeline[/bold gold1]\n"
            f"[dim]Region: {region} | Resolution: {resolution} deg | Simulation: {simulate}[/dim]",
            border_style="gold1",
        )
    )

    # 1. Grid & Collect
    grid_gdf = generate_india_grid(geojson_path=geojson, resolution_deg=resolution, region=region)
    coords = list(zip(grid_gdf["latitude"], grid_gdf["longitude"]))

    collector = NASADataCollector()
    records = asyncio.run(collector.collect(coordinates=coords, simulate=simulate))
    collector.close()

    df = pd.DataFrame(records)
    df = calculate_suitability(df)
    df.to_csv(output_csv, index=False)
    console.print(f"[green][OK] Stage 1 Complete: {len(df):,} points collected and scored -> {output_csv}[/green]")

    # 2. Interpolate & Visualize
    interpolator = SpatialInterpolator(geojson_path=geojson)
    raster_dict = interpolator.interpolate_surface(df, value_column="suitability_score", grid_resolution=500)

    visualizer = MapVisualizer(geojson_path=geojson)
    visualizer.render_static_map(raster_dict, output_image=output_png)
    visualizer.render_interactive_dashboard(df, raster_dict, output_html=output_html)
    visualizer.render_interactive_dashboard(df, raster_dict, output_html="india_solar_potential.html")
    console.print(f"[green][OK] Stage 2 Complete: High-res PNG ({output_png}) and Interactive Dashboard ({output_html}) created[/green]")

    # 3. Analyze
    summary = SolarAnalyzer.generate_summary_report(df)
    console.print(f"[green][OK] Stage 3 Complete: Mean GHI = {summary['solar_resource']['mean_daily_ghi']} kWh/m2/day, Prime Area = {summary['suitability_index']['tier_breakdown'].get('Tier 1 - Prime Location', {}).get('percentage', 0)}%[/green]")
    console.print("[bold gold1][OK] Pipeline Execution Finished Successfully![/bold gold1]")


@app.command()
def serve(
    port: int = typer.Option(8000, "--port", "-p", help="Port to serve dashboard on."),
    open_browser: bool = typer.Option(True, "--open/--no-open", help="Open browser automatically."),
):
    """Launch local web server to preview the interactive solar dashboard."""
    if not os.path.exists("index.html"):
        console.print("[bold red]index.html not found! Run 'raydium visualize' or 'raydium pipeline' first.[/bold red]")
        raise typer.Exit(1)

    url = f"http://localhost:{port}/index.html"
    console.print(f"[bold green]Serving Raydium Solar Dashboard at [link={url}]{url}[/link]...[/bold green]")
    console.print("[dim]Press Ctrl+C to stop the server.[/dim]")

    if open_browser:
        webbrowser.open(url)

    handler = http.server.SimpleHTTPRequestHandler
    with socketserver.TCPServer(("", port), handler) as httpd:
        try:
            httpd.serve_forever()
        except KeyboardInterrupt:
            console.print("\n[yellow]Server stopped.[/yellow]")


if __name__ == "__main__":
    app()
