"""High-resolution cartography and continuous GIS dashboard generator for solar plant siting in India."""

import json
import logging
import os
from typing import Dict, List, Optional

import geopandas as gpd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
import numpy as np
import pandas as pd

from raydium.models import CANDIDATE_SOLAR_ZONES, MEGA_SOLAR_PARKS, SUITABILITY_TIERS

logger = logging.getLogger(__name__)


class MapVisualizer:
    """Renders cartographic publication maps, continuous surface overlays, and interactive GIS dashboards."""

    def __init__(self, geojson_path: str = "india-soi.geojson"):
        self.geojson_path = geojson_path
        if not os.path.exists(geojson_path):
            raise FileNotFoundError(f"Boundary GeoJSON not found at: {geojson_path}")
        self.india_gdf = gpd.read_file(geojson_path)
        if self.india_gdf.crs != "EPSG:4326":
            self.india_gdf = self.india_gdf.to_crs("EPSG:4326")

    def export_transparent_surface_png(
        self,
        raster_dict: Dict,
        output_surface: str = "solar_suitability_surface.png",
        dpi: int = 250,
    ) -> str:
        """Export the continuous interpolated raster as a georeferenced transparent PNG for Leaflet ImageOverlay."""
        logger.info(f"Exporting continuous surface raster to {output_surface}...")
        masked_raster = raster_dict["raster"]
        bounds = raster_dict["bounds"]  # [minx, miny, maxx, maxy]
        vmin = raster_dict.get("vmin", float(np.nanmin(masked_raster)))
        vmax = raster_dict.get("vmax", float(np.nanmax(masked_raster)))

        aspect = (bounds[3] - bounds[1]) / (bounds[2] - bounds[0])
        fig, ax = plt.subplots(figsize=(10, 10 * aspect), dpi=dpi)
        fig.patch.set_alpha(0)
        ax.patch.set_alpha(0)
        ax.set_axis_off()

        # Scientific Solar Thermal Colormap: Deep Navy -> Indigo -> Crimson -> Amber -> Bright Solar Gold
        colors = [
            (0.06, 0.08, 0.16, 0.85),
            (0.18, 0.14, 0.45, 0.88),
            (0.48, 0.16, 0.52, 0.90),
            (0.82, 0.28, 0.32, 0.92),
            (0.96, 0.60, 0.14, 0.95),
            (0.98, 0.90, 0.38, 0.98),
            (1.00, 1.00, 0.92, 1.00),
        ]
        cmap = LinearSegmentedColormap.from_list("SolarThermalSurface", colors, N=256)

        ax.imshow(
            masked_raster,
            extent=[bounds[0], bounds[2], bounds[1], bounds[3]],
            cmap=cmap,
            origin="upper",
            vmin=vmin,
            vmax=vmax,
            interpolation="bicubic",
        )

        plt.subplots_adjust(left=0, right=1, top=1, bottom=0)
        os.makedirs(os.path.dirname(os.path.abspath(output_surface)) if os.path.dirname(output_surface) else ".", exist_ok=True)
        plt.savefig(output_surface, transparent=True, dpi=dpi, bbox_inches="tight", pad_inches=0)
        plt.close(fig)

        logger.info(f"Saved transparent surface raster: {output_surface}")
        return output_surface

    def render_static_map(
        self,
        raster_dict: Dict,
        output_image: str = "solar_potential_high_res.png",
        title: str = "India: Solar Plant Siting Suitability & Resource Map",
        dpi: int = 300,
        show_parks: bool = True,
    ) -> str:
        """Render publication-quality static map with boundary overlay and solar parks."""
        logger.info(f"Rendering cartographic map to {output_image} at {dpi} DPI...")

        masked_raster = raster_dict["raster"]
        bounds = raster_dict["bounds"]  # [minx, miny, maxx, maxy]
        vmin = raster_dict.get("vmin", float(np.nanmin(masked_raster)))
        vmax = raster_dict.get("vmax", float(np.nanmax(masked_raster)))

        fig, ax = plt.subplots(figsize=(14, 16), dpi=dpi, facecolor="#0b0f19")
        ax.set_facecolor("#0b0f19")

        # Custom Solar Thermal Palette
        colors = [
            (0.05, 0.07, 0.14),
            (0.18, 0.12, 0.42),
            (0.48, 0.15, 0.50),
            (0.82, 0.28, 0.32),
            (0.96, 0.60, 0.14),
            (0.98, 0.90, 0.38),
            (1.00, 1.00, 0.92),
        ]
        solar_cmap = LinearSegmentedColormap.from_list("SolarThermal", colors, N=256)

        # Plot interpolated raster surface
        extent = [bounds[0], bounds[2], bounds[1], bounds[3]]
        img = ax.imshow(
            masked_raster,
            extent=extent,
            cmap=solar_cmap,
            origin="upper",
            vmin=vmin,
            vmax=vmax,
            interpolation="bicubic",
        )

        # Boundary outline
        self.india_gdf.boundary.plot(ax=ax, color="#cbd5e1", linewidth=1.1, alpha=0.85)

        # Overlay candidate zones (stars) and existing parks (circles)
        if show_parks:
            # Candidate zones
            cand_lons = [z.longitude for z in CANDIDATE_SOLAR_ZONES]
            cand_lats = [z.latitude for z in CANDIDATE_SOLAR_ZONES]
            ax.scatter(
                cand_lons,
                cand_lats,
                s=120,
                c="#10b981",
                edgecolors="#ffffff",
                linewidths=1.8,
                marker="*",
                zorder=12,
                label="Candidate Siting Zones",
            )

            # Existing mega parks
            park_lons = [p.longitude for p in MEGA_SOLAR_PARKS]
            park_lats = [p.latitude for p in MEGA_SOLAR_PARKS]
            park_sizes = [min(280, max(45, p.capacity_mw / 25.0)) for p in MEGA_SOLAR_PARKS]
            ax.scatter(
                park_lons,
                park_lats,
                s=park_sizes,
                c="#38bdf8",
                edgecolors="#ffffff",
                linewidths=1.2,
                zorder=10,
                alpha=0.9,
                label="Operational Mega Parks",
            )

            # Annotate top 4 candidate zones
            for z in CANDIDATE_SOLAR_ZONES[:4]:
                ax.annotate(
                    f"#{z.rank} {z.name}\n({z.potential_capacity_mw:,.0f} MW)",
                    xy=(z.longitude, z.latitude),
                    xytext=(12, -14),
                    textcoords="offset points",
                    color="#ffffff",
                    fontsize=8.5,
                    fontfamily="sans-serif",
                    fontweight="bold",
                    bbox=dict(boxstyle="round,pad=0.3", fc="#064e3b", ec="#10b981", alpha=0.92, lw=1),
                    arrowprops=dict(arrowstyle="->", color="#10b981", lw=1.2),
                    zorder=16,
                )

        ax.set_title(title, fontsize=17, fontweight="bold", color="#ffffff", pad=16)
        ax.set_xlim(bounds[0] - 0.5, bounds[2] + 0.5)
        ax.set_ylim(bounds[1] - 0.5, bounds[3] + 0.5)
        ax.set_axis_off()

        # Colorbar
        cbar = plt.colorbar(
            img,
            ax=ax,
            orientation="horizontal",
            fraction=0.032,
            pad=0.03,
            aspect=36,
        )
        cbar.set_label("Solar Siting Suitability Score (0 - 100) / Insolation Index", color="#cbd5e1", fontsize=10.5)
        cbar.ax.tick_params(labelsize=8.5, colors="#94a3b8")

        # Cartographic metadata badge
        metadata_text = (
            "RAYDIUM PRECISION SITING PLATFORM\n"
            f"Mean Daily GHI: ~5.18 kWh/m2/day\n"
            f"Ranked Candidate Zones: {len(CANDIDATE_SOLAR_ZONES)} Sites (>40 GW)\n"
            f"Operational Mega Parks: {len(MEGA_SOLAR_PARKS)} Projects\n"
            "Data: NASA POWER Climatology & Survey of India"
        )
        ax.text(
            0.02, 0.04,
            metadata_text,
            transform=ax.transAxes,
            fontsize=8,
            color="#94a3b8",
            fontfamily="monospace",
            bbox=dict(boxstyle="square,pad=0.5", fc="#0b0f19", ec="#1e293b", alpha=0.95),
            verticalalignment="bottom",
        )

        os.makedirs(os.path.dirname(os.path.abspath(output_image)) if os.path.dirname(output_image) else ".", exist_ok=True)
        plt.savefig(output_image, bbox_inches="tight", pad_inches=0.2, dpi=dpi, facecolor="#0b0f19")
        plt.close(fig)

        logger.info(f"Static map saved to {output_image}")
        return output_image

    def render_interactive_dashboard(
        self,
        df: pd.DataFrame,
        raster_dict: Optional[Dict] = None,
        output_html: str = "index.html",
        surface_png_path: str = "solar_suitability_surface.png",
    ) -> str:
        """Generate an interactive GIS dashboard with vector grid cells rendered via Canvas.

        Each data point is rendered as a colored rectangle that stays crisp at
        every zoom level. No external PNG dependency. Clicking any cell shows
        its real GHI, DNI, temperature, and suitability metrics in the side panel.
        """
        logger.info(f"Generating vector-cell interactive dashboard at {output_html}...")

        # Build compact grid data for embedding
        cols_needed = ["latitude", "longitude", "ghi_daily", "dni_daily",
                       "temp_ambient", "suitability_score", "suitability_tier"]
        grid_rows = []
        for _, row in df.iterrows():
            grid_rows.append({
                "la": round(float(row.get("latitude", 0)), 4),
                "lo": round(float(row.get("longitude", 0)), 4),
                "g": round(float(row.get("ghi_daily", 0)), 3),
                "d": round(float(row.get("dni_daily", 0)), 3),
                "t": round(float(row.get("temp_ambient", 25)), 1),
                "s": round(float(row.get("suitability_score", 0)), 1),
                "ti": str(row.get("suitability_tier", "Unclassified")),
            })

        # Auto-detect grid step from data
        lats_sorted = sorted(set(r["la"] for r in grid_rows))
        if len(lats_sorted) > 1:
            diffs = [lats_sorted[i+1] - lats_sorted[i] for i in range(min(20, len(lats_sorted) - 1))]
            grid_step = round(min(d for d in diffs if d > 0.001), 4)
        else:
            grid_step = 0.25

        # Candidate Zones Data
        candidate_zones_data = [
            {
                "rank": z.rank, "name": z.name, "district": z.district,
                "state": z.state, "lat": z.latitude, "lon": z.longitude,
                "capacity_mw": z.potential_capacity_mw,
                "area_acres": z.estimated_area_acres,
                "ghi": z.ghi_daily, "dni": z.dni_daily,
                "score": z.suitability_score, "slope": z.terrain_slope_pct,
                "land_type": z.land_type, "substation": z.nearest_substation,
                "grid_dist_km": z.substation_distance_km,
                "water_clean": z.water_cleaning_index,
                "lcoe": z.lcoe_estimate_inr,
                "advantages": z.key_advantages,
            }
            for z in CANDIDATE_SOLAR_ZONES
        ]

        # Existing Parks Data
        parks_data = [
            {
                "name": p.name, "state": p.state, "capacity": p.capacity_mw,
                "lat": p.latitude, "lon": p.longitude, "status": p.status,
                "developer": p.developer, "substation": p.substation,
                "year": p.commissioned_year, "desc": p.description,
            }
            for p in MEGA_SOLAR_PARKS
        ]

        # Simplified boundary GeoJSON
        simplified_gdf = self.india_gdf.copy()
        simplified_gdf.geometry = self.india_gdf.geometry.simplify(0.012, preserve_topology=True)
        geojson_str = simplified_gdf.to_json()

        html_template = """<!DOCTYPE html>
<html lang="en" class="dark">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Raydium: Solar Plant Siting Tool for India</title>
    <link rel="preconnect" href="https://fonts.googleapis.com">
    <link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
    <link href="https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&family=JetBrains+Mono:wght@400;500;600&display=swap" rel="stylesheet">
    <script src="https://cdn.tailwindcss.com"></script>
    <script>
        tailwind.config = {
            darkMode: 'class',
            theme: {
                extend: {
                    fontFamily: {
                        sans: ['Inter', 'sans-serif'],
                        mono: ['JetBrains Mono', 'monospace'],
                    }
                }
            }
        }
    </script>
    <link rel="stylesheet" href="https://unpkg.com/leaflet@1.9.4/dist/leaflet.css" />
    <script src="https://unpkg.com/leaflet@1.9.4/dist/leaflet.js"></script>
    <script src="https://unpkg.com/lucide@latest"></script>
    <style>
        body { font-family: 'Inter', sans-serif; }
        #map { height: calc(100vh - 56px); width: 100%; z-index: 10; background: #050811; }
        .custom-popup .leaflet-popup-content-wrapper {
            background: #0f172a; color: #f1f5f9;
            border: 1px solid #1e293b; border-radius: 8px;
            box-shadow: 0 10px 20px -5px rgb(0 0 0 / 0.6); padding: 2px;
        }
        .custom-popup .leaflet-popup-tip { background: #0f172a; }
        .panel {
            background: rgba(15, 23, 42, 0.92);
            backdrop-filter: blur(12px);
            border: 1px solid rgba(30, 41, 59, 0.7);
        }
        ::-webkit-scrollbar { width: 5px; }
        ::-webkit-scrollbar-track { background: transparent; }
        ::-webkit-scrollbar-thumb { background: #334155; border-radius: 3px; }
    </style>
</head>
<body class="bg-slate-950 text-slate-100 font-sans antialiased overflow-hidden flex flex-col h-screen">

    <!-- Navbar -->
    <header class="h-14 bg-slate-900 border-b border-slate-800 px-5 flex items-center justify-between z-30 shrink-0">
        <div class="flex items-center gap-3">
            <div class="w-8 h-8 rounded-lg bg-amber-500/10 border border-amber-500/30 flex items-center justify-center text-amber-400">
                <i data-lucide="sun" class="w-4 h-4"></i>
            </div>
            <h1 class="text-sm font-semibold tracking-tight text-white">RAYDIUM</h1>
            <span class="text-[10px] font-mono bg-emerald-500/15 text-emerald-400 px-1.5 py-0.5 rounded border border-emerald-500/25">SITING</span>
        </div>
        <div class="hidden md:flex items-center gap-2 text-xs">
            <select id="siteSelect" onchange="zoomToCandidate(this.value)"
                class="bg-slate-800 text-slate-200 text-xs rounded px-2.5 py-1.5 border border-slate-700 focus:outline-none focus:border-amber-500 cursor-pointer">
                <option value="">Jump to candidate zone...</option>
            </select>
        </div>
        <div class="flex items-center gap-2">
            <button onclick="toggleSidePanel()" class="flex items-center gap-1.5 px-2.5 py-1.5 rounded bg-slate-800 hover:bg-slate-700 text-xs text-slate-200 border border-slate-700 transition">
                <i data-lucide="sliders" class="w-3.5 h-3.5 text-amber-400"></i> Sizing
            </button>
            <button onclick="downloadCSV()" class="flex items-center gap-1.5 px-2.5 py-1.5 rounded bg-emerald-600 hover:bg-emerald-500 text-xs text-white font-medium transition">
                <i data-lucide="download" class="w-3.5 h-3.5"></i> CSV
            </button>
        </div>
    </header>

    <div class="relative flex-1 w-full overflow-hidden">
        <div id="map"></div>

        <!-- Legend (bottom-left) -->
        <div class="absolute bottom-5 left-5 z-20 panel rounded-lg p-3.5 shadow-lg w-64 text-xs">
            <h4 class="font-semibold text-slate-200 text-[11px] uppercase tracking-wider mb-2">Layers</h4>
            <div class="space-y-1.5 text-[11px] mb-2.5">
                <div class="flex items-center justify-between">
                    <span class="flex items-center gap-2 text-slate-300">
                        <span class="w-4 h-3 rounded-sm" style="background:linear-gradient(90deg,#0d0887,#7e03a8,#cc4778,#f89540,#f0f921)"></span>
                        Solar suitability (0-100)
                    </span>
                </div>
                <div class="flex items-center gap-2 text-slate-300">
                    <span class="w-2.5 h-2.5 rounded-full bg-emerald-500 border border-white"></span>
                    Candidate zones (new sites)
                </div>
                <div class="flex items-center gap-2 text-slate-300">
                    <span class="w-2.5 h-2.5 rounded-full bg-sky-400"></span>
                    Operational solar parks
                </div>
            </div>
            <div class="pt-2 border-t border-slate-800 space-y-2">
                <div class="flex justify-between items-center text-[11px]">
                    <span class="text-slate-400">Basemap</span>
                    <div class="flex rounded bg-slate-800 p-0.5 text-[10px]">
                        <button id="btnDark" onclick="switchBasemap('dark')" class="px-2 py-0.5 rounded bg-slate-700 text-white font-medium">Dark</button>
                        <button id="btnSat" onclick="switchBasemap('sat')" class="px-2 py-0.5 rounded text-slate-400 hover:text-white">Satellite</button>
                    </div>
                </div>
                <div>
                    <div class="flex justify-between text-slate-400 text-[10px] mb-0.5">
                        <span>Grid opacity</span>
                        <span id="opVal" class="text-slate-200 font-mono">80%</span>
                    </div>
                    <input id="opSlider" type="range" min="10" max="100" value="80"
                        class="w-full accent-amber-400 h-1.5 bg-slate-800 rounded cursor-pointer"
                        oninput="setGridOpacity(this.value)">
                </div>
            </div>
        </div>

        <!-- Side panel (right) -->
        <div id="sidePanel" class="absolute top-3 right-3 bottom-4 z-20 panel rounded-lg p-4 shadow-xl w-80 overflow-y-auto flex flex-col transition-transform duration-200 translate-x-0">
            <div class="flex items-center justify-between pb-2.5 border-b border-slate-800 mb-3">
                <h3 class="font-semibold text-slate-100 text-sm flex items-center gap-2">
                    <i data-lucide="map-pin" class="w-4 h-4 text-emerald-400"></i>
                    <span id="panelTitle">Site Details</span>
                </h3>
                <button onclick="toggleSidePanel()" class="text-slate-400 hover:text-white p-0.5 rounded hover:bg-slate-800 transition">
                    <i data-lucide="x" class="w-4 h-4"></i>
                </button>
            </div>

            <div class="space-y-2.5 flex-1">
                <!-- Location header -->
                <div class="bg-slate-800/60 rounded-lg p-2.5 border border-slate-700/50 text-xs">
                    <div class="flex justify-between items-start mb-0.5">
                        <h4 id="siteName" class="font-semibold text-emerald-400 text-[13px] leading-tight">Click a grid cell or zone</h4>
                        <span id="siteBadge" class="text-[10px] font-mono font-bold bg-slate-700 text-slate-300 px-1.5 py-0.5 rounded shrink-0 ml-2">--</span>
                    </div>
                    <div id="siteLocation" class="text-slate-400 text-[11px]">Select a location on the map</div>
                    <div id="siteCoords" class="text-[10px] font-mono text-slate-500 mt-0.5"></div>
                </div>

                <!-- Metrics grid -->
                <div class="grid grid-cols-2 gap-1.5 text-xs">
                    <div class="bg-slate-800/50 p-2 rounded border border-slate-700/40">
                        <span class="text-slate-500 block text-[10px]">GHI (daily)</span>
                        <span id="valGhi" class="text-sm font-bold text-amber-400 font-mono">--</span>
                        <span class="text-[10px] text-slate-500 block">kWh/m2/day</span>
                    </div>
                    <div class="bg-slate-800/50 p-2 rounded border border-slate-700/40">
                        <span class="text-slate-500 block text-[10px]">DNI (daily)</span>
                        <span id="valDni" class="text-sm font-bold text-orange-400 font-mono">--</span>
                        <span class="text-[10px] text-slate-500 block">kWh/m2/day</span>
                    </div>
                    <div class="bg-slate-800/50 p-2 rounded border border-slate-700/40">
                        <span class="text-slate-500 block text-[10px]">Suitability</span>
                        <span id="valScore" class="text-sm font-bold text-emerald-400 font-mono">--</span>
                        <span class="text-[10px] text-slate-500 block">/ 100</span>
                    </div>
                    <div class="bg-slate-800/50 p-2 rounded border border-slate-700/40">
                        <span class="text-slate-500 block text-[10px]">Avg. Temp</span>
                        <span id="valTemp" class="text-sm font-bold text-sky-400 font-mono">--</span>
                        <span class="text-[10px] text-slate-500 block">deg C</span>
                    </div>
                </div>

                <!-- Tier & extra info -->
                <div class="bg-slate-800/60 rounded-lg p-2.5 border border-slate-700/50 text-xs space-y-1.5">
                    <div class="flex justify-between items-center">
                        <span class="text-slate-400">Tier:</span>
                        <span id="valTier" class="font-mono text-slate-200 text-[11px]">--</span>
                    </div>
                    <div id="extraInfo" class="hidden space-y-1.5">
                        <div class="flex justify-between items-center">
                            <span class="text-slate-400">Terrain slope:</span>
                            <span id="valSlope" class="font-mono text-slate-200">--</span>
                        </div>
                        <div class="flex justify-between items-center">
                            <span class="text-slate-400">Land type:</span>
                            <span id="valLand" class="text-right text-slate-300 text-[11px] max-w-[55%]">--</span>
                        </div>
                        <div>
                            <span class="text-slate-400 block text-[10px]">Grid substation:</span>
                            <span id="valSub" class="text-slate-200 text-[11px]">--</span>
                        </div>
                        <div class="flex justify-between items-center">
                            <span class="text-slate-400">Est. LCOE:</span>
                            <span id="valLcoe" class="font-mono font-semibold text-emerald-400">--</span>
                        </div>
                        <div>
                            <span class="text-slate-400 block text-[10px] mb-0.5">Advantages:</span>
                            <p id="valAdv" class="text-slate-300 text-[11px] leading-relaxed"></p>
                        </div>
                    </div>
                </div>

                <!-- Sizing calculator -->
                <div class="pt-1">
                    <div class="flex justify-between items-center mb-1">
                        <label class="text-xs font-medium text-slate-200">Land area</label>
                        <span id="landVal" class="text-xs font-mono font-semibold text-amber-400">500 Acres</span>
                    </div>
                    <input id="landSlider" type="range" min="10" max="5000" step="25" value="500"
                        class="w-full accent-amber-400 h-1.5 bg-slate-800 rounded cursor-pointer" oninput="calcYield(this.value)">
                    <div class="bg-slate-800/60 rounded p-2 border border-slate-700/40 text-[11px] mt-1.5 grid grid-cols-2 gap-1 font-mono">
                        <div>Capacity: <span id="cMw" class="text-white font-bold">111 MW</span></div>
                        <div>Gen: <span id="cGen" class="text-amber-400 font-bold">201 GWh</span></div>
                        <div>Revenue: <span id="cRev" class="text-emerald-400 font-bold">52.3 Cr</span></div>
                        <div>CO2: <span id="cCo2" class="text-emerald-300 font-bold">165k T</span></div>
                    </div>
                </div>
            </div>

            <div class="mt-2.5 pt-2 border-t border-slate-800 text-[10px] text-slate-500">
                Click any grid cell or candidate marker to inspect.
            </div>
        </div>
    </div>

    <script>
        const GRID = """ + json.dumps(grid_rows) + """;
        const STEP = """ + str(grid_step) + """;
        const CANDIDATES = """ + json.dumps(candidate_zones_data) + """;
        const PARKS = """ + json.dumps(parks_data) + """;
        const BOUNDARY = """ + geojson_str + """;

        let map, gridLayer, darkTiles, satTiles;
        let currentGhi = 5.0;
        let gridOpacity = 0.80;

        // Plasma colormap (matches matplotlib plasma)
        function scoreColor(s) {
            const stops = [
                [0,   13,  8, 135],
                [20,  84,  2, 163],
                [35, 139, 10, 165],
                [50, 185, 50, 137],
                [62, 219, 92, 104],
                [72, 244,136,  73],
                [82, 254,188,  43],
                [92, 240,230,  33],
                [100,240,249,  33]
            ];
            s = Math.max(0, Math.min(100, s));
            let lo = stops[0], hi = stops[stops.length-1];
            for (let i = 0; i < stops.length - 1; i++) {
                if (s >= stops[i][0] && s <= stops[i+1][0]) {
                    lo = stops[i]; hi = stops[i+1]; break;
                }
            }
            const t = (hi[0] === lo[0]) ? 0 : (s - lo[0]) / (hi[0] - lo[0]);
            const r = Math.round(lo[1] + t * (hi[1] - lo[1]));
            const g = Math.round(lo[2] + t * (hi[2] - lo[2]));
            const b = Math.round(lo[3] + t * (hi[3] - lo[3]));
            return `rgb(${r},${g},${b})`;
        }

        function initMap() {
            map = L.map('map', {
                center: [22.5, 79.0],
                zoom: 5,
                minZoom: 4,
                maxZoom: 18,
                preferCanvas: true,
                zoomControl: false
            });
            L.control.zoom({ position: 'topleft' }).addTo(map);

            darkTiles = L.tileLayer('https://{s}.basemaps.cartocdn.com/dark_all/{z}/{x}/{y}{r}.png', {
                attribution: '&copy; OpenStreetMap &copy; CARTO',
                subdomains: 'abcd', maxZoom: 19
            }).addTo(map);

            satTiles = L.tileLayer('https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}', {
                attribution: '&copy; Esri'
            });

            // Boundary outline
            L.geoJSON(BOUNDARY, {
                style: { color: '#94a3b8', weight: 1.0, fillOpacity: 0, interactive: false }
            }).addTo(map);

            // Vector grid cells
            gridLayer = L.layerGroup().addTo(map);
            const half = STEP / 2;
            GRID.forEach(pt => {
                const rect = L.rectangle(
                    [[pt.la - half, pt.lo - half], [pt.la + half, pt.lo + half]],
                    {
                        color: scoreColor(pt.s),
                        fillColor: scoreColor(pt.s),
                        fillOpacity: gridOpacity,
                        weight: 0.3,
                        opacity: 0.4
                    }
                );
                rect._ptData = pt;
                rect.on('click', function() { showCellInfo(this._ptData); });
                gridLayer.addLayer(rect);
            });

            // Candidate zones
            const selEl = document.getElementById('siteSelect');
            CANDIDATES.forEach((z, i) => {
                const opt = document.createElement('option');
                opt.value = i;
                opt.innerText = '#' + z.rank + ' ' + z.name + ' (' + z.capacity_mw.toLocaleString() + ' MW)';
                selEl.appendChild(opt);

                const m = L.circleMarker([z.lat, z.lon], {
                    radius: 10, fillColor: '#10b981', color: '#fff',
                    weight: 2.5, fillOpacity: 0.95
                });
                m.bindPopup('<div class="custom-popup text-xs p-1.5"><b class="text-emerald-400">#' + z.rank + ' ' + z.name + '</b><br>' + z.district + ', ' + z.state + '<br>GHI: ' + z.ghi + ' | Score: ' + z.score + '</div>');
                m.on('click', function() { showZoneInfo(z); });
                m.addTo(map);
            });

            // Existing parks
            PARKS.forEach(p => {
                const m = L.circleMarker([p.lat, p.lon], {
                    radius: Math.max(4, Math.min(11, p.capacity / 300)),
                    fillColor: '#38bdf8', color: '#fff',
                    weight: 1.2, fillOpacity: 0.85
                });
                m.bindPopup('<div class="custom-popup text-xs p-1.5"><b class="text-sky-400">' + p.name + '</b><br>' + p.state + ' | ' + p.capacity.toLocaleString() + ' MW | ' + p.status + '</div>');
                m.addTo(map);
            });
        }

        function showCellInfo(pt) {
            currentGhi = pt.g;
            document.getElementById('panelTitle').innerText = 'Grid Cell';
            document.getElementById('siteName').innerText = 'Grid Cell (' + pt.la.toFixed(2) + ', ' + pt.lo.toFixed(2) + ')';
            document.getElementById('siteBadge').innerText = pt.ti.replace('Tier ', 'T').split(' -')[0];
            document.getElementById('siteBadge').className = 'text-[10px] font-mono font-bold px-1.5 py-0.5 rounded shrink-0 ml-2 ' +
                (pt.s >= 85 ? 'bg-emerald-500/20 text-emerald-300 border border-emerald-500/30' :
                 pt.s >= 70 ? 'bg-sky-500/20 text-sky-300 border border-sky-500/30' :
                 pt.s >= 50 ? 'bg-amber-500/20 text-amber-300 border border-amber-500/30' :
                              'bg-slate-700 text-slate-400 border border-slate-600');
            document.getElementById('siteLocation').innerText = pt.ti;
            document.getElementById('siteCoords').innerText = pt.la.toFixed(4) + ' N, ' + pt.lo.toFixed(4) + ' E';
            document.getElementById('valGhi').innerText = pt.g.toFixed(2);
            document.getElementById('valDni').innerText = pt.d.toFixed(2);
            document.getElementById('valScore').innerText = pt.s.toFixed(1);
            document.getElementById('valTemp').innerText = pt.t.toFixed(1);
            document.getElementById('valTier').innerText = pt.ti;
            document.getElementById('extraInfo').classList.add('hidden');
            calcYield(document.getElementById('landSlider').value);

            // Open panel if closed
            const panel = document.getElementById('sidePanel');
            if (panel.classList.contains('translate-x-full')) toggleSidePanel();
        }

        function showZoneInfo(z) {
            currentGhi = z.ghi;
            document.getElementById('panelTitle').innerText = 'Candidate Zone';
            document.getElementById('siteName').innerText = z.name;
            document.getElementById('siteBadge').innerText = '#' + z.rank;
            document.getElementById('siteBadge').className = 'text-[10px] font-mono font-bold bg-emerald-500/20 text-emerald-300 px-1.5 py-0.5 rounded border border-emerald-500/30 shrink-0 ml-2';
            document.getElementById('siteLocation').innerText = z.district + ', ' + z.state;
            document.getElementById('siteCoords').innerText = z.lat.toFixed(4) + ' N, ' + z.lon.toFixed(4) + ' E | ' + z.capacity_mw.toLocaleString() + ' MW / ' + z.area_acres.toLocaleString() + ' acres';
            document.getElementById('valGhi').innerText = z.ghi.toFixed(2);
            document.getElementById('valDni').innerText = z.dni.toFixed(2);
            document.getElementById('valScore').innerText = z.score.toFixed(1);
            document.getElementById('valTemp').innerText = '--';
            document.getElementById('valTier').innerText = 'Tier 1 - Prime';
            document.getElementById('valSlope').innerText = z.slope + '%';
            document.getElementById('valLand').innerText = z.land_type;
            document.getElementById('valSub').innerText = z.substation + ' (' + z.grid_dist_km + ' km)';
            document.getElementById('valLcoe').innerText = 'Rs ' + z.lcoe.toFixed(2) + '/kWh';
            document.getElementById('valAdv').innerText = z.advantages;
            document.getElementById('extraInfo').classList.remove('hidden');
            calcYield(document.getElementById('landSlider').value);

            const panel = document.getElementById('sidePanel');
            if (panel.classList.contains('translate-x-full')) toggleSidePanel();
        }

        function zoomToCandidate(idx) {
            if (idx === '' || idx === null) return;
            const z = CANDIDATES[parseInt(idx)];
            map.flyTo([z.lat, z.lon], 8, { duration: 1.2 });
            showZoneInfo(z);
        }

        function calcYield(acres) {
            document.getElementById('landVal').innerText = acres + ' Acres';
            const mw = acres / 4.5;
            const genGwh = mw * currentGhi * 365 * 0.78 / 1000;
            const revCr = genGwh * 1000000 * 2.60 / 10000000;
            const co2k = genGwh * 1000000 * 0.82 / 1000000;
            document.getElementById('cMw').innerText = mw.toFixed(0) + ' MW';
            document.getElementById('cGen').innerText = genGwh.toFixed(1) + ' GWh';
            document.getElementById('cRev').innerText = revCr.toFixed(1) + ' Cr';
            document.getElementById('cCo2').innerText = co2k.toFixed(0) + 'k T';
        }

        function setGridOpacity(val) {
            gridOpacity = val / 100;
            document.getElementById('opVal').innerText = val + '%';
            gridLayer.eachLayer(function(layer) {
                layer.setStyle({ fillOpacity: gridOpacity });
            });
        }

        function switchBasemap(mode) {
            if (mode === 'sat') {
                map.removeLayer(darkTiles); map.addLayer(satTiles);
                document.getElementById('btnSat').className = 'px-2 py-0.5 rounded bg-slate-700 text-white font-medium';
                document.getElementById('btnDark').className = 'px-2 py-0.5 rounded text-slate-400 hover:text-white';
            } else {
                map.removeLayer(satTiles); map.addLayer(darkTiles);
                document.getElementById('btnDark').className = 'px-2 py-0.5 rounded bg-slate-700 text-white font-medium';
                document.getElementById('btnSat').className = 'px-2 py-0.5 rounded text-slate-400 hover:text-white';
            }
        }

        function toggleSidePanel() {
            const p = document.getElementById('sidePanel');
            p.classList.toggle('translate-x-full');
            p.classList.toggle('translate-x-0');
        }

        function downloadCSV() {
            let csv = 'rank,name,district,state,lat,lon,capacity_mw,area_acres,ghi,dni,score,slope,land_type,substation,grid_km,lcoe\\n';
            CANDIDATES.forEach(z => {
                csv += z.rank + ',"' + z.name + '","' + z.district + '","' + z.state + '",' + z.lat + ',' + z.lon + ',' + z.capacity_mw + ',' + z.area_acres + ',' + z.ghi + ',' + z.dni + ',' + z.score + ',' + z.slope + ',"' + z.land_type + '","' + z.substation + '",' + z.grid_dist_km + ',' + z.lcoe + '\\n';
            });
            const a = document.createElement('a');
            a.href = URL.createObjectURL(new Blob([csv], {type:'text/csv'}));
            a.download = 'raydium_candidate_solar_sites.csv';
            a.click();
        }

        window.onload = function() {
            initMap();
            lucide.createIcons();
            calcYield(500);
        };
    </script>
</body>
</html>"""

        with open(output_html, "w", encoding="utf-8") as f:
            f.write(html_template)

        file_size_kb = os.path.getsize(output_html) / 1024
        logger.info(f"Dashboard generated: {output_html} ({file_size_kb:.1f} KB, {len(grid_rows)} grid cells)")
        return output_html

