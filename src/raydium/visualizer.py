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
        """Generate high-precision interactive GIS platform with continuous canvas tile rendering.

        Uses a dynamic per-pixel Canvas tile engine (L.GridLayer) to render a silky-smooth,
        infinite-resolution solar suitability surface at any zoom level, with 15 candidate
        site parcel polygons, transmission interconnection paths, multi-criteria screening,
        and real-time coordinate probing.
        """
        logger.info(f"Generating high-precision GIS platform at {output_html}...")

        # Ensure spatial interpolation on regular uniform dense grid
        from raydium.interpolator import SpatialInterpolator
        interpolator = SpatialInterpolator(geojson_path=self.geojson_path)
        
        score_res = interpolator.interpolate_surface(df, value_column="suitability_score", grid_resolution=120)
        ghi_res   = interpolator.interpolate_surface(df, value_column="ghi_daily", grid_resolution=120)
        dni_res   = interpolator.interpolate_surface(df, value_column="dni_daily", grid_resolution=120)
        temp_res  = interpolator.interpolate_surface(df, value_column="temp_ambient", grid_resolution=120)

        minx, miny, maxx, maxy = score_res["bounds"]
        mask = score_res["mask"]  # (height, width), True where inside India
        height, width = score_res["raw_raster"].shape

        raw_score = score_res["raw_raster"]
        raw_ghi   = ghi_res["raw_raster"]
        raw_dni   = dni_res["raw_raster"]
        raw_temp  = temp_res["raw_raster"]

        # Build compact flat matrix: row 0 is lat_max, row height-1 is lat_min
        grid_matrix = []
        for r in range(height):
            for c in range(width):
                if mask[r, c]:
                    grid_matrix.append([
                        round(float(raw_score[r, c]), 1),
                        round(float(raw_ghi[r, c]), 2),
                        round(float(raw_dni[r, c]), 2),
                        round(float(raw_temp[r, c]), 1)
                    ])
                else:
                    grid_matrix.append(None)

        grid_meta = {
            "lat_min": round(float(miny), 4),
            "lat_max": round(float(maxy), 4),
            "lon_min": round(float(minx), 4),
            "lon_max": round(float(maxx), 4),
            "n_lat": height,
            "n_lon": width,
        }

        # Candidate Zones Data with Polygons and Substations
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
                "polygon": z.polygon_bounds,
                "sub_coords": z.substation_coords,
            }
            for z in CANDIDATE_SOLAR_ZONES
        ]

        # Existing Mega Solar Parks
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
        simplified_gdf.geometry = self.india_gdf.geometry.simplify(0.010, preserve_topology=True)
        geojson_str = simplified_gdf.to_json()

        html_template = """<!DOCTYPE html>
<html lang="en" class="dark">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Raydium: High-Precision Solar Siting Platform for India</title>
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
            border: 1px solid #334155; border-radius: 8px;
            box-shadow: 0 12px 24px -6px rgb(0 0 0 / 0.7); padding: 2px;
        }
        .custom-popup .leaflet-popup-tip { background: #0f172a; }
        .panel {
            background: rgba(15, 23, 42, 0.94);
            backdrop-filter: blur(14px);
            border: 1px solid rgba(51, 65, 85, 0.7);
        }
        ::-webkit-scrollbar { width: 5px; }
        ::-webkit-scrollbar-track { background: transparent; }
        ::-webkit-scrollbar-thumb { background: #334155; border-radius: 3px; }
    </style>
</head>
<body class="bg-slate-950 text-slate-100 font-sans antialiased overflow-hidden flex flex-col h-screen">

    <!-- Header -->
    <header class="h-14 bg-slate-900 border-b border-slate-800 px-5 flex items-center justify-between z-30 shrink-0">
        <div class="flex items-center gap-3">
            <div class="w-8 h-8 rounded-lg bg-amber-500/10 border border-amber-500/30 flex items-center justify-center text-amber-400">
                <i data-lucide="sun" class="w-4 h-4"></i>
            </div>
            <div>
                <div class="flex items-center gap-2">
                    <h1 class="text-sm font-semibold tracking-tight text-white">RAYDIUM</h1>
                    <span class="text-[10px] font-mono bg-emerald-500/20 text-emerald-300 px-1.5 py-0.5 rounded border border-emerald-500/30">GIS SITING</span>
                </div>
            </div>
        </div>

        <div class="hidden lg:flex items-center gap-3 text-xs">
            <select id="siteSelect" onchange="zoomToCandidate(this.value)"
                class="bg-slate-800 text-slate-200 text-xs rounded-md px-3 py-1.5 border border-slate-700 focus:outline-none focus:border-amber-500 cursor-pointer">
                <option value="">Jump to candidate site zone...</option>
            </select>
            <button onclick="toggleFilterDrawer()" class="flex items-center gap-1.5 px-3 py-1.5 rounded-md bg-slate-800 hover:bg-slate-700 text-xs text-slate-200 border border-slate-700 transition">
                <i data-lucide="filter" class="w-3.5 h-3.5 text-amber-400"></i>
                <span>Siting Screener</span>
                <span id="filterCountBadge" class="bg-emerald-500/20 text-emerald-300 text-[10px] font-mono px-1.5 rounded ml-1">15/15</span>
            </button>
        </div>

        <div class="flex items-center gap-2">
            <button onclick="toggleSidePanel()" class="flex items-center gap-1.5 px-3 py-1.5 rounded-md bg-slate-800 hover:bg-slate-700 text-xs text-slate-200 border border-slate-700 transition">
                <i data-lucide="sliders" class="w-3.5 h-3.5 text-amber-400"></i> Sizing
            </button>
            <button onclick="downloadCSV()" class="flex items-center gap-1.5 px-3 py-1.5 rounded-md bg-emerald-600 hover:bg-emerald-500 text-xs text-white font-medium transition">
                <i data-lucide="download" class="w-3.5 h-3.5"></i> Export CSV
            </button>
        </div>
    </header>

    <div class="relative flex-1 w-full overflow-hidden">
        <div id="map"></div>

        <!-- Filter Screener Drawer (Top Center Dropdown) -->
        <div id="filterDrawer" class="hidden absolute top-3 left-1/2 -translate-x-1/2 z-20 panel rounded-lg p-4 shadow-2xl w-full max-w-2xl text-xs space-y-3">
            <div class="flex items-center justify-between pb-2 border-b border-slate-800">
                <div class="flex items-center gap-2">
                    <i data-lucide="sliders-horizontal" class="w-4 h-4 text-emerald-400"></i>
                    <h3 class="font-semibold text-slate-100">Multi-Criteria Solar Siting Screener</h3>
                </div>
                <button onclick="toggleFilterDrawer()" class="text-slate-400 hover:text-white">
                    <i data-lucide="x" class="w-4 h-4"></i>
                </button>
            </div>

            <div class="grid grid-cols-1 md:grid-cols-4 gap-3">
                <div>
                    <div class="flex justify-between text-[11px] text-slate-400 mb-1">
                        <span>Min GHI</span>
                        <span id="filterGhiVal" class="font-mono text-amber-400 font-semibold">5.0 kWh</span>
                    </div>
                    <input id="filterGhi" type="range" min="5.0" max="6.4" step="0.1" value="5.0"
                        class="w-full accent-amber-400 h-1.5 bg-slate-800 rounded cursor-pointer" oninput="applyFilters()">
                </div>

                <div>
                    <div class="flex justify-between text-[11px] text-slate-400 mb-1">
                        <span>Max Slope</span>
                        <span id="filterSlopeVal" class="font-mono text-slate-200 font-semibold">2.5%</span>
                    </div>
                    <input id="filterSlope" type="range" min="0.5" max="3.0" step="0.1" value="2.5"
                        class="w-full accent-emerald-400 h-1.5 bg-slate-800 rounded cursor-pointer" oninput="applyFilters()">
                </div>

                <div>
                    <div class="flex justify-between text-[11px] text-slate-400 mb-1">
                        <span>Max Grid Dist</span>
                        <span id="filterGridVal" class="font-mono text-slate-200 font-semibold">40 km</span>
                    </div>
                    <input id="filterGrid" type="range" min="10" max="50" step="5" value="40"
                        class="w-full accent-sky-400 h-1.5 bg-slate-800 rounded cursor-pointer" oninput="applyFilters()">
                </div>

                <div>
                    <label class="block text-[11px] text-slate-400 mb-1">State</label>
                    <select id="filterState" onchange="applyFilters()"
                        class="w-full bg-slate-800 text-slate-200 text-xs rounded px-2 py-1 border border-slate-700 focus:outline-none">
                        <option value="ALL">All States (15 Sites)</option>
                        <option value="Rajasthan">Rajasthan (4 Sites)</option>
                        <option value="Gujarat">Gujarat (3 Sites)</option>
                        <option value="Ladakh">Ladakh (1 Site)</option>
                        <option value="Andhra Pradesh">Andhra Pradesh (2 Sites)</option>
                        <option value="Karnataka">Karnataka (2 Sites)</option>
                        <option value="Madhya Pradesh">Madhya Pradesh (1 Site)</option>
                        <option value="Maharashtra">Maharashtra (1 Site)</option>
                        <option value="Tamil Nadu">Tamil Nadu (1 Site)</option>
                    </select>
                </div>
            </div>

            <div class="flex items-center justify-between pt-2 border-t border-slate-800 text-[11px]">
                <span id="screenerSummary" class="text-emerald-400 font-mono">15 of 15 candidate site zones qualify</span>
                <button onclick="resetFilters()" class="text-slate-400 hover:text-slate-200 underline">Reset all filters</button>
            </div>
        </div>

        <!-- Legend & Controls (Bottom Left) -->
        <div class="absolute bottom-5 left-5 z-20 panel rounded-lg p-3.5 shadow-xl w-72 text-xs">
            <h4 class="font-semibold text-slate-200 text-[11px] uppercase tracking-wider mb-2">GIS Layers & Cartography</h4>
            
            <div class="space-y-1.5 text-[11px] mb-3">
                <div class="flex items-center justify-between">
                    <span class="flex items-center gap-2 text-slate-300">
                        <span class="w-4 h-3 rounded-sm" style="background:linear-gradient(90deg,#0d0887,#7e03a8,#cc4778,#f89540,#f0f921)"></span>
                        Suitability Index (0-100)
                    </span>
                    <span class="font-mono text-[10px] text-amber-400">Continuous</span>
                </div>
                <div class="flex items-center justify-between text-slate-300">
                    <span class="flex items-center gap-2">
                        <span class="w-3.5 h-2.5 rounded-sm border border-emerald-400 bg-emerald-500/30"></span>
                        Candidate Site Parcels
                    </span>
                    <span class="font-mono text-[10px] text-emerald-400">15 Sites</span>
                </div>
                <div class="flex items-center justify-between text-slate-300">
                    <span class="flex items-center gap-2">
                        <span class="w-2.5 h-2.5 rounded-full bg-sky-400 border border-white"></span>
                        Operational Mega Parks
                    </span>
                    <span class="font-mono text-[10px] text-sky-400">Bhadla/Khavda</span>
                </div>
                <div class="flex items-center gap-2 text-slate-400">
                    <span class="w-4 border-t border-dashed border-amber-400"></span>
                    <span>765/400 kV Grid Lines</span>
                </div>
            </div>

            <div class="pt-2 border-t border-slate-800 space-y-2">
                <div class="flex justify-between items-center text-[11px]">
                    <span class="text-slate-400">Basemap</span>
                    <div class="flex rounded bg-slate-800 p-0.5 text-[10px]">
                        <button id="btnDark" onclick="switchBasemap('dark')" class="px-2 py-0.5 rounded bg-slate-700 text-white font-medium">Dark</button>
                        <button id="btnSat" onclick="switchBasemap('satellite')" class="px-2 py-0.5 rounded text-slate-400 hover:text-white">Satellite</button>
                    </div>
                </div>

                <div>
                    <div class="flex justify-between text-slate-400 text-[10px] mb-0.5">
                        <span>Surface Opacity</span>
                        <span id="opacityVal" class="text-slate-200 font-mono">75%</span>
                    </div>
                    <input id="heatOpacity" type="range" min="0" max="100" value="75"
                        class="w-full accent-amber-400 h-1.5 bg-slate-800 rounded cursor-pointer" oninput="updateSurfaceOpacity(this.value)">
                </div>
            </div>
        </div>

        <!-- Siting Dossier & Sizing Drawer (Right Side) -->
        <div id="sidePanel" class="absolute top-3 right-3 bottom-4 z-20 panel rounded-lg p-4 shadow-xl w-84 max-w-[340px] overflow-y-auto flex flex-col justify-between transition-transform duration-200 translate-x-0">
            <div>
                <div class="flex items-center justify-between pb-2 border-b border-slate-800 mb-2.5">
                    <div>
                        <h3 class="font-semibold text-slate-100 text-sm flex items-center gap-1.5">
                            <i data-lucide="map-pin" class="w-4 h-4 text-emerald-400"></i>
                            <span id="panelTitle">Site Dossier</span>
                        </h3>
                        <p id="siteSubtitle" class="text-[10px] text-slate-400">Rank #1 Candidate Solar Zone</p>
                    </div>
                    <button onclick="toggleSidePanel()" class="text-slate-400 hover:text-white p-0.5 rounded hover:bg-slate-800">
                        <i data-lucide="x" class="w-4 h-4"></i>
                    </button>
                </div>

                <!-- Location Header -->
                <div class="space-y-2.5 text-xs">
                    <div class="bg-slate-800/70 rounded-lg p-2.5 border border-slate-700/60">
                        <div class="flex justify-between items-start mb-0.5">
                            <h4 id="siteName" class="font-semibold text-emerald-400 text-[13px] leading-tight">Jaisalmer-Fatehgarh West</h4>
                            <span id="siteBadge" class="text-[10px] font-mono font-bold bg-emerald-500/20 text-emerald-300 px-1.5 py-0.5 rounded border border-emerald-500/30 shrink-0 ml-1">RANK #1</span>
                        </div>
                        <div id="siteLocation" class="text-slate-300 text-[11px]">Jaisalmer, Rajasthan</div>
                        <div id="siteCoords" class="text-[10px] font-mono text-slate-400 mt-0.5">26.9500 N, 70.9200 E | 8,000 MW (~35,000 Acres)</div>
                    </div>

                    <!-- Metrics Grid -->
                    <div class="grid grid-cols-2 gap-1.5">
                        <div class="bg-slate-800/50 p-2 rounded border border-slate-700/40">
                            <span class="text-slate-500 block text-[10px]">Daily GHI</span>
                            <span id="valGhi" class="text-sm font-bold text-amber-400 font-mono">6.35</span>
                            <span class="text-[10px] text-slate-500 block">kWh/m2/day</span>
                        </div>
                        <div class="bg-slate-800/50 p-2 rounded border border-slate-700/40">
                            <span class="text-slate-500 block text-[10px]">Daily DNI</span>
                            <span id="valDni" class="text-sm font-bold text-orange-400 font-mono">6.90</span>
                            <span class="text-[10px] text-slate-500 block">kWh/m2/day</span>
                        </div>
                        <div class="bg-slate-800/50 p-2 rounded border border-slate-700/40">
                            <span class="text-slate-500 block text-[10px]">Suitability Score</span>
                            <span id="valScore" class="text-sm font-bold text-emerald-400 font-mono">98.5</span>
                            <span class="text-[10px] text-slate-500 block">Tier 1 - Prime</span>
                        </div>
                        <div class="bg-slate-800/50 p-2 rounded border border-slate-700/40">
                            <span class="text-slate-500 block text-[10px]">Terrain Slope</span>
                            <span id="valSlope" class="text-sm font-bold text-sky-400 font-mono">0.8%</span>
                            <span class="text-[10px] text-slate-500 block">Ultra-Flat</span>
                        </div>
                    </div>

                    <!-- Geospatial Dossier -->
                    <div class="bg-slate-800/60 rounded-lg p-2.5 border border-slate-700/50 space-y-1.5 text-[11px]">
                        <div class="flex justify-between items-start py-0.5 border-b border-slate-800/60">
                            <span class="text-slate-400">Land Type:</span>
                            <span id="valLand" class="text-right text-slate-200 max-w-[55%]">Barren Sandy & Rocky Desert</span>
                        </div>
                        <div class="py-0.5 border-b border-slate-800/60">
                            <span class="text-slate-400 block text-[10px]">ISTS Substation:</span>
                            <span id="valSubstation" class="text-slate-200 font-medium">765/400 kV Fatehgarh-II (18 km)</span>
                        </div>
                        <div class="flex justify-between items-center py-0.5 border-b border-slate-800/60">
                            <span class="text-slate-400">Estimated LCOE:</span>
                            <span id="valLcoe" class="font-mono font-semibold text-emerald-400">Rs 2.35 / kWh</span>
                        </div>
                        <div class="py-0.5">
                            <span class="text-slate-400 block text-[10px] mb-0.5">Key Site Advantages:</span>
                            <p id="valAdvantages" class="text-slate-300 text-[11px] leading-relaxed">Highest GHI in South Asia (>325 sunny days), zero agricultural conflict, direct access to Green Energy Corridor.</p>
                        </div>
                    </div>

                    <!-- Custom Sizing Calculator -->
                    <div class="pt-1">
                        <div class="flex justify-between items-center mb-1 text-xs">
                            <label class="font-medium text-slate-200">Custom Land Area</label>
                            <span id="landAreaDisplay" class="font-mono font-semibold text-amber-400">500 Acres</span>
                        </div>
                        <input id="landAreaSlider" type="range" min="50" max="5000" step="25" value="500"
                            class="w-full accent-amber-400 h-1.5 bg-slate-800 rounded cursor-pointer" oninput="calculateCustomYield(this.value)">
                        <div class="bg-slate-800/60 rounded p-2 border border-slate-700/40 text-[11px] mt-1.5 grid grid-cols-2 gap-1 font-mono">
                            <div>Capacity: <span id="customMw" class="text-white font-bold">111 MW</span></div>
                            <div>Gen: <span id="customGen" class="text-amber-400 font-bold">201 GWh/yr</span></div>
                            <div>Revenue: <span id="customRev" class="text-emerald-400 font-bold">Rs 52.3 Cr</span></div>
                            <div>CO2: <span id="customCo2" class="text-emerald-300 font-bold">165k T</span></div>
                        </div>
                    </div>
                </div>
            </div>

            <div class="mt-2.5 pt-2 border-t border-slate-800 text-[10px] text-slate-500">
                Click any candidate parcel or click anywhere in India to inspect.
            </div>
        </div>
    </div>

    <script>
        const GRID_META = """ + json.dumps(grid_meta) + """;
        const GRID_MATRIX = """ + json.dumps(grid_matrix) + """;
        const CANDIDATE_ZONES = """ + json.dumps(candidate_zones_data) + """;
        const MEGA_PARKS = """ + json.dumps(parks_data) + """;
        const INDIA_GEOJSON = """ + geojson_str + """;

        let map, darkTiles, satelliteTiles;
        let continuousCanvasLayer, candidatePolygonsLayer, candidateLinesLayer, parksLayer, boundaryLayer;
        let currentGhi = 6.35;
        let surfaceOpacity = 0.75;

        // Scientific Plasma/Turbo Colormap for Suitability (0-100)
        function getSuitabilityRgb(s) {
            const stops = [
                [0,   13,  8, 135],
                [25,  84,  2, 163],
                [45, 139, 10, 165],
                [60, 185, 50, 137],
                [72, 219, 92, 104],
                [82, 244,136,  73],
                [90, 254,188,  43],
                [96, 240,230,  33],
                [100,240,249,  33]
            ];
            s = Math.max(0, Math.min(100, s));
            let lo = stops[0], hi = stops[stops.length - 1];
            for (let i = 0; i < stops.length - 1; i++) {
                if (s >= stops[i][0] && s <= stops[i+1][0]) {
                    lo = stops[i]; hi = stops[i+1]; break;
                }
            }
            const t = (hi[0] === lo[0]) ? 0 : (s - lo[0]) / (hi[0] - lo[0]);
            return [
                Math.round(lo[1] + t * (hi[1] - lo[1])),
                Math.round(lo[2] + t * (hi[2] - lo[2])),
                Math.round(lo[3] + t * (hi[3] - lo[3]))
            ];
        }

        // Fast Bilinear Spatial Interpolator on Uniform 2D Matrix
        function sampleBilinear(lat, lon) {
            const { lat_min, lat_max, lon_min, lon_max, n_lat, n_lon } = GRID_META;
            if (lat < lat_min || lat > lat_max || lon < lon_min || lon > lon_max) return null;

            // Row 0 is at lat_max, row n_lat-1 is at lat_min
            const row = ((lat_max - lat) / (lat_max - lat_min)) * (n_lat - 1);
            // Col 0 is at lon_min, col n_lon-1 is at lon_max
            const col = ((lon - lon_min) / (lon_max - lon_min)) * (n_lon - 1);

            const r0 = Math.floor(row);
            const r1 = Math.min(n_lat - 1, r0 + 1);
            const c0 = Math.floor(col);
            const c1 = Math.min(n_lon - 1, c0 + 1);

            const tr = row - r0;
            const tc = col - c0;

            const v00 = GRID_MATRIX[r0 * n_lon + c0];
            const v01 = GRID_MATRIX[r0 * n_lon + c1];
            const v10 = GRID_MATRIX[r1 * n_lon + c0];
            const v11 = GRID_MATRIX[r1 * n_lon + c1];

            const valid = [v00, v01, v10, v11].filter(v => v !== null && v !== undefined);
            if (valid.length === 0) return null;
            if (valid.length < 4) {
                const v = valid[0];
                return { score: v[0], ghi: v[1], dni: v[2], temp: v[3] };
            }

            const score = (1 - tr) * ((1 - tc) * v00[0] + tc * v01[0]) + tr * ((1 - tc) * v10[0] + tc * v11[0]);
            const ghi   = (1 - tr) * ((1 - tc) * v00[1] + tc * v01[1]) + tr * ((1 - tc) * v10[1] + tc * v11[1]);
            const dni   = (1 - tr) * ((1 - tc) * v00[2] + tc * v01[2]) + tr * ((1 - tc) * v10[2] + tc * v11[2]);
            const temp  = (1 - tr) * ((1 - tc) * v00[3] + tc * v01[3]) + tr * ((1 - tc) * v10[3] + tc * v11[3]);

            return { score, ghi, dni, temp };
        }

        function tile2lat(y, z) {
            const n = Math.PI - 2 * Math.PI * y / Math.pow(2, z);
            return (180 / Math.PI * Math.atan(0.5 * (Math.exp(n) - Math.exp(-n))));
        }

        function tile2lon(x, z) {
            return (x / Math.pow(2, z) * 360 - 180);
        }

        function initMap() {
            map = L.map('map', {
                center: [22.8, 79.5],
                zoom: 5,
                minZoom: 4,
                maxZoom: 18,
                zoomControl: false,
                preferCanvas: true
            });
            L.control.zoom({ position: 'topleft' }).addTo(map);

            darkTiles = L.tileLayer('https://{s}.basemaps.cartocdn.com/dark_all/{z}/{x}/{y}{r}.png', {
                attribution: '&copy; OpenStreetMap &copy; CARTO',
                subdomains: 'abcd', maxZoom: 19
            }).addTo(map);

            satelliteTiles = L.tileLayer('https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}', {
                attribution: '&copy; Esri World Imagery', maxZoom: 18
            });

            // 1. Dynamic Infinite-Resolution Canvas Tile Layer
            const ContinuousGridLayer = L.GridLayer.extend({
                createTile: function(coords, done) {
                    const tile = document.createElement('canvas');
                    const tileSize = this.getTileSize();
                    tile.width = tileSize.x;
                    tile.height = tileSize.y;

                    const ctx = tile.getContext('2d');
                    
                    // Render at 64x64 sub-resolution and smoothly upscale via canvas
                    const subW = 64, subH = 64;
                    const offscreen = document.createElement('canvas');
                    offscreen.width = subW;
                    offscreen.height = subH;
                    const oCtx = offscreen.getContext('2d');
                    const imgData = oCtx.createImageData(subW, subH);
                    const data = imgData.data;

                    const z = coords.z;
                    const x0 = coords.x, y0 = coords.y;

                    let hasData = false;

                    for (let py = 0; py < subH; py++) {
                        const tileY = y0 + py / subH;
                        const lat = tile2lat(tileY, z);

                        for (let px = 0; px < subW; px++) {
                            const tileX = x0 + px / subW;
                            const lon = tile2lon(tileX, z);

                            const sample = sampleBilinear(lat, lon);
                            const idx = (py * subW + px) * 4;

                            if (sample && sample.score > 0) {
                                const rgb = getSuitabilityRgb(sample.score);
                                data[idx]     = rgb[0];
                                data[idx + 1] = rgb[1];
                                data[idx + 2] = rgb[2];
                                data[idx + 3] = Math.round(255 * surfaceOpacity);
                                hasData = true;
                            } else {
                                data[idx + 3] = 0;
                            }
                        }
                    }

                    if (hasData) {
                        oCtx.putImageData(imgData, 0, 0);
                        ctx.imageSmoothingEnabled = true;
                        ctx.imageSmoothingQuality = 'high';
                        ctx.drawImage(offscreen, 0, 0, tileSize.x, tileSize.y);
                    }

                    setTimeout(() => done(null, tile), 0);
                    return tile;
                }
            });

            continuousCanvasLayer = new ContinuousGridLayer({ tileSize: 256, zIndex: 10 }).addTo(map);

            // 2. Official Boundary Line
            boundaryLayer = L.geoJSON(INDIA_GEOJSON, {
                style: { color: '#64748b', weight: 1.2, fillOpacity: 0.0, interactive: false }
            }).addTo(map);

            // 3. Candidate Site Parcels & Transmission Lines Layer
            candidatePolygonsLayer = L.layerGroup().addTo(map);
            candidateLinesLayer = L.layerGroup().addTo(map);

            renderCandidateLayers();

            // 4. Operational Mega Solar Parks
            parksLayer = L.layerGroup().addTo(map);
            MEGA_PARKS.forEach(park => {
                const marker = L.circleMarker([park.lat, park.lon], {
                    radius: Math.max(5, Math.min(12, park.capacity / 250)),
                    fillColor: '#38bdf8', color: '#ffffff',
                    weight: 1.5, fillOpacity: 0.85
                });
                marker.bindPopup(`
                    <div class="p-2 custom-popup text-xs">
                        <div class="font-semibold text-sky-400 text-sm flex items-center justify-between mb-1">
                            <span>${park.name}</span>
                            <span class="text-[10px] font-mono bg-sky-500/20 text-sky-300 px-1.5 py-0.5 rounded font-bold">${park.capacity.toLocaleString()} MW</span>
                        </div>
                        <div class="text-slate-300"><strong>State:</strong> ${park.state} | <strong>Status:</strong> ${park.status}</div>
                        <div class="text-slate-300"><strong>Substation:</strong> ${park.substation || 'Grid Substation'}</div>
                        <p class="text-slate-400 text-[11px] mt-1">${park.desc}</p>
                    </div>
                `);
                parksLayer.addLayer(marker);
            });

            // 5. Point Investigation Probe on Map Click
            map.on('click', (e) => {
                probePoint(e.latlng.lat, e.latlng.lng);
            });

            // Populate selector
            const sel = document.getElementById('siteSelect');
            CANDIDATE_ZONES.forEach((z, i) => {
                const opt = document.createElement('option');
                opt.value = i;
                opt.innerText = `#${z.rank} ${z.name} (${z.capacity_mw.toLocaleString()} MW)`;
                sel.appendChild(opt);
            });

            displayCandidateDossier(CANDIDATE_ZONES[0]);
        }

        function renderCandidateLayers(filterFn) {
            candidatePolygonsLayer.clearLayers();
            candidateLinesLayer.clearLayers();

            CANDIDATE_ZONES.forEach((zone, idx) => {
                const pass = filterFn ? filterFn(zone) : true;
                if (!pass) return;

                // 1. Polygon Parcel
                if (zone.polygon && zone.polygon.length >= 3) {
                    const poly = L.polygon(zone.polygon, {
                        color: '#10b981',
                        weight: 2,
                        fillColor: '#10b981',
                        fillOpacity: 0.25,
                        dashArray: '4, 4'
                    });
                    poly.bindPopup(`
                        <div class="p-1.5 custom-popup text-xs">
                            <b class="text-emerald-400 text-sm">#${zone.rank} ${zone.name}</b><br>
                            <span class="text-slate-300">${zone.district}, ${zone.state}</span><br>
                            <strong>Capacity:</strong> ${zone.capacity_mw.toLocaleString()} MW (~${zone.area_acres.toLocaleString()} Acres)<br>
                            <strong>Daily GHI:</strong> ${zone.ghi} kWh/m2/day | <strong>LCOE:</strong> Rs ${zone.lcoe}/kWh
                        </div>
                    `);
                    poly.on('click', () => displayCandidateDossier(zone));
                    candidatePolygonsLayer.addLayer(poly);
                }

                // 2. Centroid Marker with Badge
                const m = L.circleMarker([zone.lat, zone.lon], {
                    radius: 11,
                    fillColor: '#10b981',
                    color: '#ffffff',
                    weight: 2.5,
                    fillOpacity: 0.95
                });
                m.on('click', () => displayCandidateDossier(zone));
                candidatePolygonsLayer.addLayer(m);

                // 3. Transmission Interconnection Line
                if (zone.sub_coords && zone.sub_coords.length === 2) {
                    const line = L.polyline([[zone.lat, zone.lon], zone.sub_coords], {
                        color: '#f59e0b',
                        weight: 1.5,
                        dashArray: '3, 6',
                        opacity: 0.8
                    });
                    candidateLinesLayer.addLayer(line);

                    const subMarker = L.circleMarker(zone.sub_coords, {
                        radius: 5,
                        fillColor: '#f59e0b',
                        color: '#ffffff',
                        weight: 1.5,
                        fillOpacity: 0.9
                    });
                    subMarker.bindPopup(`<div class="custom-popup text-xs p-1"><b class="text-amber-400">${zone.substation}</b><br>Distance: ${zone.grid_dist_km} km</div>`);
                    candidateLinesLayer.addLayer(subMarker);
                }
            });
        }

        function displayCandidateDossier(zone) {
            currentGhi = zone.ghi;
            document.getElementById('panelTitle').innerText = "Site Dossier";
            document.getElementById('siteSubtitle').innerText = `Rank #${zone.rank} Candidate Solar Zone`;
            document.getElementById('siteName').innerText = zone.name;
            document.getElementById('siteBadge').innerText = `RANK #${zone.rank}`;
            document.getElementById('siteBadge').className = "text-[10px] font-mono font-bold px-1.5 py-0.5 rounded border shrink-0 ml-1 bg-emerald-500/20 text-emerald-300 border-emerald-500/30";
            document.getElementById('siteLocation').innerText = `${zone.district}, ${zone.state}`;
            document.getElementById('siteCoords').innerText = `${zone.lat.toFixed(4)} N, ${zone.lon.toFixed(4)} E | ${zone.capacity_mw.toLocaleString()} MW (~${zone.area_acres.toLocaleString()} Acres)`;
            document.getElementById('valGhi').innerText = zone.ghi.toFixed(2);
            document.getElementById('valDni').innerText = zone.dni.toFixed(2);
            document.getElementById('valScore').innerText = zone.score.toFixed(1);
            document.getElementById('valSlope').innerText = `${zone.slope}%`;
            document.getElementById('valLand').innerText = zone.land_type;
            document.getElementById('valSubstation').innerText = `${zone.substation} (${zone.grid_dist_km} km)`;
            document.getElementById('valLcoe').innerText = `Rs ${zone.lcoe.toFixed(2)} / kWh`;
            document.getElementById('valAdvantages').innerText = zone.advantages;

            calculateCustomYield(document.getElementById('landAreaSlider').value);

            const panel = document.getElementById('sidePanel');
            if (panel.classList.contains('translate-x-full')) toggleSidePanel();
        }

        // National Inter-State Transmission System (ISTS) 765/400 kV Substation Network
        const ISTS_SUBSTATIONS = [
            { name: "765/400 kV Fatehgarh-II Pooling Station", lat: 26.78, lon: 71.12, kv: "765/400 kV", state: "Rajasthan" },
            { name: "765/400 kV Bhadla-II Pooling Station", lat: 27.54, lon: 71.92, kv: "765/400 kV", state: "Rajasthan" },
            { name: "765/400 kV Bikaner-II Pooling Station", lat: 28.05, lon: 73.15, kv: "765/400 kV", state: "Rajasthan" },
            { name: "400 kV Barmer Pooling Substation", lat: 25.75, lon: 71.40, kv: "400 kV", state: "Rajasthan" },
            { name: "400 kV Ramgarh Gas/Solar Substation", lat: 27.35, lon: 70.52, kv: "400 kV", state: "Rajasthan" },
            { name: "765 kV Khavda Pooling Station (ISTS)", lat: 23.85, lon: 69.75, kv: "765 kV", state: "Gujarat" },
            { name: "765/400 kV Banaskantha Substation", lat: 23.68, lon: 71.85, kv: "765/400 kV", state: "Gujarat" },
            { name: "400 kV Dholera SIR Substation", lat: 22.24, lon: 72.19, kv: "400 kV", state: "Gujarat" },
            { name: "400 kV Surendranagar Substation", lat: 22.72, lon: 71.64, kv: "400 kV", state: "Gujarat" },
            { name: "400 kV Bhuj Pooling Substation", lat: 23.25, lon: 69.66, kv: "400 kV", state: "Gujarat" },
            { name: "Proposed 765 kV Pang-Kaithal HVDC Terminal", lat: 32.90, lon: 77.80, kv: "765 kV HVDC", state: "Ladakh" },
            { name: "220/66 kV Leh PGCIL Substation", lat: 34.15, lon: 77.58, kv: "220 kV", state: "Ladakh" },
            { name: "400/220 kV Kurnool / Orvakal Pooling Substation", lat: 15.80, lon: 78.05, kv: "400 kV", state: "Andhra Pradesh" },
            { name: "400/220 kV NP Kunta Pooling Station", lat: 14.15, lon: 78.25, kv: "400 kV", state: "Andhra Pradesh" },
            { name: "400 kV Kadapa Substation", lat: 14.50, lon: 78.80, kv: "400 kV", state: "Andhra Pradesh" },
            { name: "400 kV Gooty Substation", lat: 15.12, lon: 77.64, kv: "400 kV", state: "Andhra Pradesh" },
            { name: "400/220 kV Pavagada Pooling Substation", lat: 14.17, lon: 77.27, kv: "400 kV", state: "Karnataka" },
            { name: "765/400 kV Koppal-II Pooling Station", lat: 15.35, lon: 76.15, kv: "765/400 kV", state: "Karnataka" },
            { name: "400 kV Hiriyur Substation", lat: 13.95, lon: 76.62, kv: "400 kV", state: "Karnataka" },
            { name: "400 kV Bellary Substation", lat: 15.15, lon: 76.92, kv: "400 kV", state: "Karnataka" },
            { name: "400 kV Rewa PGCIL Substation", lat: 24.48, lon: 81.58, kv: "400 kV", state: "Madhya Pradesh" },
            { name: "400 kV Neemuch / Mandsaur Substation", lat: 24.46, lon: 74.87, kv: "400 kV", state: "Madhya Pradesh" },
            { name: "765/400 kV Vindhyachal Pooling Substation", lat: 24.10, lon: 82.68, kv: "765/400 kV", state: "Madhya Pradesh" },
            { name: "765/400 kV Solapur PGCIL Substation", lat: 17.68, lon: 75.92, kv: "765/400 kV", state: "Maharashtra" },
            { name: "400 kV Dhule Substation", lat: 20.90, lon: 74.78, kv: "400 kV", state: "Maharashtra" },
            { name: "765 kV Wardha Powergrid Substation", lat: 20.74, lon: 78.60, kv: "765 kV", state: "Maharashtra" },
            { name: "400 kV Kamuthi Pooling Substation", lat: 9.35, lon: 78.40, kv: "400 kV", state: "Tamil Nadu" },
            { name: "400 kV Tuticorin Pooling Station", lat: 8.76, lon: 78.13, kv: "400 kV", state: "Tamil Nadu" },
            { name: "765 kV Agra PGCIL Substation", lat: 27.18, lon: 78.02, kv: "765 kV", state: "Uttar Pradesh" },
            { name: "400 kV Varanasi Substation", lat: 25.32, lon: 82.97, kv: "400 kV", state: "Uttar Pradesh" }
        ];

        function getClosestSubstation(lat, lon) {
            let closest = ISTS_SUBSTATIONS[0];
            let minDist = 99999;
            ISTS_SUBSTATIONS.forEach(sub => {
                const dLat = (sub.lat - lat) * (Math.PI / 180);
                const dLon = (sub.lon - lon) * (Math.PI / 180);
                const a = Math.sin(dLat / 2) * Math.sin(dLat / 2) +
                          Math.cos(lat * Math.PI / 180) * Math.cos(sub.lat * Math.PI / 180) *
                          Math.sin(dLon / 2) * Math.sin(dLon / 2);
                const c = 2 * Math.atan2(Math.sqrt(a), Math.sqrt(1 - a));
                const distKm = 6371 * c;
                if (distKm < minDist) {
                    minDist = distKm;
                    closest = sub;
                }
            });
            return { sub: closest, distKm: Math.round(minDist * 10) / 10 };
        }

        // Physics-Based Regional Geo-Classification Engine
        function getRegionalProfile(lat, lon, sample) {
            const nearest = getClosestSubstation(lat, lon);
            const ghi = sample.ghi;
            const temp = sample.temp;

            // 1. Thar Desert / Western Rajasthan
            if (lat >= 24.5 && lat <= 29.8 && lon >= 69.5 && lon <= 75.2) {
                return {
                    regionName: "Thar Hyper-Arid Desert Basin",
                    state: "Rajasthan",
                    landType: "Barren Sandy & Rocky Desert Wasteland (Govt Class 1)",
                    slope: "0.7% (Ultra-Flat Plain)",
                    substation: `${nearest.sub.name} (${nearest.distKm} km)`,
                    gridDist: nearest.distKm,
                    lcoe: 2.32 + Math.max(0, (nearest.distKm - 15) * 0.005),
                    cuf: 26.8,
                    cleaning: "Automated Waterless Dry Robotic Cleaning",
                    advantages: "Highest GHI & DNI in South Asia (>325 clear days), low moisture, minimal agricultural conflict, direct Green Energy Corridor interconnection."
                };
            }

            // 2. Rann of Kutch & Saurashtra / Gujarat
            if (lat >= 20.8 && lat <= 24.8 && lon >= 68.4 && lon <= 73.2) {
                return {
                    regionName: "Rann of Kutch & Saurashtra Solar Corridor",
                    state: "Gujarat",
                    landType: "Saline Mudflat & Non-Arable Scrub (Class 1 Wasteland)",
                    slope: "0.4% (Ultra-Flat Salt Mudflat)",
                    substation: `${nearest.sub.name} (${nearest.distKm} km)`,
                    gridDist: nearest.distKm,
                    lcoe: 2.38 + Math.max(0, (nearest.distKm - 15) * 0.005),
                    cuf: 25.6,
                    cleaning: "Anti-Corrosive Saline Coating + Dry Robotic Wash",
                    advantages: "Ultra-flat natural salt terrain with zero human resettlement, optimal single-axis tracker geometry, and high transmission capacity."
                };
            }

            // 3. Ladakh High-Altitude Plateau
            if (lat >= 32.2 && lat <= 35.8 && lon >= 75.8 && lon <= 80.2) {
                return {
                    regionName: "Ladakh High-Altitude Cold Desert Plateau",
                    state: "Ladakh",
                    landType: "High-Altitude Barren Cold Gravel Plain (4,200m ASL)",
                    slope: "2.1% (High Mountain Plateau Basin)",
                    substation: `${nearest.sub.name} (${nearest.distKm} km)`,
                    gridDist: nearest.distKm,
                    lcoe: 2.50,
                    cuf: 28.4,
                    cleaning: "Low-Humidity Dry Air Jet Cleaning / Snow Clearing",
                    advantages: "Exceptional thin-atmosphere DNI (>7.2 kWh/m2); sub-zero ambient temperatures boost silicon cell efficiency by +11.5%."
                };
            }

            // 4. Andhra Pradesh / Telangana Deccan Ridge
            if (lat >= 13.5 && lat <= 19.5 && lon >= 77.0 && lon <= 81.2) {
                return {
                    regionName: "Rayalaseema / Deccan Rocky Scrubland",
                    state: "Andhra Pradesh",
                    landType: "Uncultivable Non-Arable Rocky Red Soil Wasteland",
                    slope: "1.4% (Gentle Undulating Plateau)",
                    substation: `${nearest.sub.name} (${nearest.distKm} km)`,
                    gridDist: nearest.distKm,
                    lcoe: 2.52 + Math.max(0, (nearest.distKm - 15) * 0.006),
                    cuf: 24.6,
                    cleaning: "Semi-Dry Robotic Wash (Treated Water)",
                    advantages: "Consistent year-round Southern Grid solar profile, high capacity factor (>24% CUF), and robust 400 kV transmission evacuation."
                };
            }

            // 5. Karnataka Sun Belt (Tumkur, Bellary, Koppal)
            if (lat >= 12.5 && lat <= 17.5 && lon >= 74.8 && lon <= 77.8) {
                return {
                    regionName: "Karnataka Dryland Solar Belt",
                    state: "Karnataka",
                    landType: "Semi-Arid Dry Scrubland & Degraded Pasture",
                    slope: "1.3% (Gentle Plains)",
                    substation: `${nearest.sub.name} (${nearest.distKm} km)`,
                    gridDist: nearest.distKm,
                    lcoe: 2.56 + Math.max(0, (nearest.distKm - 15) * 0.006),
                    cuf: 24.3,
                    cleaning: "Dry Robotic Cleaning System",
                    advantages: "Hub of Karnataka Green Energy Corridor Phase-II, high transmission headroom, and active land-lease policy framework."
                };
            }

            // 6. Central India / MP Malwa & Vindhyas
            if (lat >= 21.5 && lat <= 26.5 && lon >= 74.5 && lon <= 82.8) {
                return {
                    regionName: "Central India Vindhyan Solar Basin",
                    state: "Madhya Pradesh",
                    landType: "Barren Stony Deccan Basalt Plateau",
                    slope: "1.7% (Stony Undulations)",
                    substation: `${nearest.sub.name} (${nearest.distKm} km)`,
                    gridDist: nearest.distKm,
                    lcoe: 2.62,
                    cuf: 23.8,
                    cleaning: "Standard Cleaning with Pipeline Water Access",
                    advantages: "Central national grid node with dual interconnection feeding Northern and Western industrial corridors."
                };
            }

            // 7. Maharashtra Deccan Basin
            if (lat >= 16.5 && lat <= 21.8 && lon >= 73.8 && lon <= 80.5) {
                return {
                    regionName: "Maharashtra Semi-Arid Deccan Plateau",
                    state: "Maharashtra",
                    landType: "Barren Black-Rock Deccan Trap Ridge",
                    slope: "1.8% (Basalt Ridges)",
                    substation: `${nearest.sub.name} (${nearest.distKm} km)`,
                    gridDist: nearest.distKm,
                    lcoe: 2.64,
                    cuf: 23.5,
                    cleaning: "Semi-Dry Robotic Wash",
                    advantages: "Direct access to 765 kV Western Region grid lines feeding Mumbai-Pune industrial load centers."
                };
            }

            // 8. Tamil Nadu Southern Plains
            if (lat >= 8.2 && lat <= 13.0 && lon >= 77.0 && lon <= 80.2) {
                return {
                    regionName: "Tamil Nadu Southern Solar Belt",
                    state: "Tamil Nadu",
                    landType: "Flat Coastal Arid Plain (Non-Arable)",
                    slope: "0.8% (Flat Plain)",
                    substation: `${nearest.sub.name} (${nearest.distKm} km)`,
                    gridDist: nearest.distKm,
                    lcoe: 2.65,
                    cuf: 24.0,
                    cleaning: "Automated Robotic Washing (High Salt Resistance)",
                    advantages: "Southern grid stabilization, year-round maritime airflow cooling modules, and high local industrial power demand."
                };
            }

            // 9. High Relief / Mountainous / Forested Regions (Western Ghats, High Himalayas, NE)
            if ((lat > 31.0 && lon < 76.5) || (lon >= 73.2 && lon <= 75.8 && lat >= 8.5 && lat <= 18.0) || lon > 89.0) {
                return {
                    regionName: "High Relief / Forested Mountain Escarpment",
                    state: "Regional Mountain Zone",
                    landType: "Forested / Mountain Slope Eco-Sensitive Zone",
                    slope: "> 6.5% (High Mountain Relief)",
                    substation: `${nearest.sub.name} (${nearest.distKm} km)`,
                    gridDist: nearest.distKm,
                    lcoe: 3.10,
                    cuf: 20.2,
                    cleaning: "Manual Wet Wash / Natural Rainfall",
                    advantages: "Constrained for utility ground-mount solar due to slope and forest clearance. Rooftop & distributed solar recommended."
                };
            }

            // 10. General Indo-Gangetic Plains
            return {
                regionName: "Indo-Gangetic Basin",
                state: "North-Central India",
                landType: "Alluvial Agricultural Land & Scrub",
                slope: "0.9% (Flat Alluvial Plain)",
                substation: `${nearest.sub.name} (${nearest.distKm} km)`,
                gridDist: nearest.distKm,
                lcoe: 2.78,
                cuf: 22.0,
                cleaning: "Standard Washing Regime",
                advantages: "Proximity to high-density power consumers; land acquisition requires careful non-agricultural parcel selection."
            };
        }

        function probePoint(lat, lon) {
            const sample = sampleBilinear(lat, lon);
            if (!sample || sample.score <= 0) return;

            const profile = getRegionalProfile(lat, lon, sample);
            currentGhi = sample.ghi;

            document.getElementById('panelTitle').innerText = "Investigated Site";
            document.getElementById('siteSubtitle').innerText = profile.regionName;
            document.getElementById('siteName').innerText = `${profile.regionName}`;
            
            const tierBadge = sample.score >= 85 ? "TIER 1 - PRIME" : sample.score >= 70 ? "TIER 2 - HIGH" : sample.score >= 55 ? "TIER 3 - MODERATE" : "TIER 4 - CONSTRAINED";
            const badgeClass = sample.score >= 85 ? "bg-emerald-500/20 text-emerald-300 border-emerald-500/30" :
                               sample.score >= 70 ? "bg-sky-500/20 text-sky-300 border-sky-500/30" :
                               sample.score >= 55 ? "bg-amber-500/20 text-amber-300 border-amber-500/30" :
                                                    "bg-rose-500/20 text-rose-300 border-rose-500/30";
            
            document.getElementById('siteBadge').innerText = tierBadge;
            document.getElementById('siteBadge').className = `text-[10px] font-mono font-bold px-1.5 py-0.5 rounded border shrink-0 ml-1 ${badgeClass}`;
            
            document.getElementById('siteLocation').innerText = `${profile.state} | Coord: ${lat.toFixed(4)} N, ${lon.toFixed(4)} E`;
            document.getElementById('siteCoords').innerText = `Specific Yield: ~${Math.round(sample.ghi * 365 * 0.78)} kWh/kWp/yr | CUF: ${profile.cuf}% | Temp: ${sample.temp.toFixed(1)} C`;
            
            document.getElementById('valGhi').innerText = sample.ghi.toFixed(2);
            document.getElementById('valDni').innerText = sample.dni.toFixed(2);
            document.getElementById('valScore').innerText = sample.score.toFixed(1);
            document.getElementById('valSlope').innerText = profile.slope;
            document.getElementById('valLand').innerText = profile.landType;
            document.getElementById('valSubstation').innerText = profile.substation;
            document.getElementById('valLcoe').innerText = `Rs ${profile.lcoe.toFixed(2)} / kWh`;
            document.getElementById('valAdvantages').innerText = profile.advantages;

            calculateCustomYield(document.getElementById('landAreaSlider').value);

            const panel = document.getElementById('sidePanel');
            if (panel.classList.contains('translate-x-full')) toggleSidePanel();
        }

        function zoomToCandidate(idx) {
            if (idx === "" || idx === null) return;
            const zone = CANDIDATE_ZONES[parseInt(idx)];
            if (!zone) return;
            map.flyTo([zone.lat, zone.lon], 9, { duration: 1.4 });
            displayCandidateDossier(zone);
        }

        function calculateCustomYield(landAcres) {
            document.getElementById('landAreaDisplay').innerText = `${landAcres} Acres`;
            const mw = landAcres / 4.5;
            const genGwh = (mw * currentGhi * 365 * 0.78) / 1000;
            const revCr = (genGwh * 1000000 * 2.60) / 10000000;
            const co2k = (genGwh * 1000000 * 0.82) / 1000000;

            document.getElementById('customMw').innerText = `${mw.toFixed(0)} MW`;
            document.getElementById('customGen').innerText = `${genGwh.toFixed(1)} GWh/yr`;
            document.getElementById('customRev').innerText = `Rs ${revCr.toFixed(1)} Cr`;
            document.getElementById('customCo2').innerText = `${co2k.toFixed(0)}k T`;
        }

        function updateSurfaceOpacity(val) {
            surfaceOpacity = val / 100;
            document.getElementById('opacityVal').innerText = `${val}%`;
            if (continuousCanvasLayer) {
                continuousCanvasLayer.redraw();
            }
        }

        function switchBasemap(mode) {
            if (mode === 'satellite') {
                map.removeLayer(darkTiles);
                map.addLayer(satelliteTiles);
                document.getElementById('btnSat').className = "px-2 py-0.5 rounded bg-slate-700 text-white font-medium";
                document.getElementById('btnDark').className = "px-2 py-0.5 rounded text-slate-400 hover:text-white";
            } else {
                map.removeLayer(satelliteTiles);
                map.addLayer(darkTiles);
                document.getElementById('btnDark').className = "px-2 py-0.5 rounded bg-slate-700 text-white font-medium";
                document.getElementById('btnSat').className = "px-2 py-0.5 rounded text-slate-400 hover:text-white";
            }
        }

        function toggleSidePanel() {
            const p = document.getElementById('sidePanel');
            p.classList.toggle('translate-x-full');
            p.classList.toggle('translate-x-0');
        }

        function toggleFilterDrawer() {
            const d = document.getElementById('filterDrawer');
            d.classList.toggle('hidden');
        }

        function applyFilters() {
            const minGhi = parseFloat(document.getElementById('filterGhi').value);
            const maxSlope = parseFloat(document.getElementById('filterSlope').value);
            const maxGrid = parseFloat(document.getElementById('filterGrid').value);
            const state = document.getElementById('filterState').value;

            document.getElementById('filterGhiVal').innerText = `${minGhi.toFixed(1)} kWh`;
            document.getElementById('filterSlopeVal').innerText = `${maxSlope.toFixed(1)}%`;
            document.getElementById('filterGridVal').innerText = `${maxGrid} km`;

            let matchCount = 0;
            const filterFn = (zone) => {
                const pass = (zone.ghi >= minGhi) &&
                             (zone.slope <= maxSlope) &&
                             (zone.grid_dist_km <= maxGrid) &&
                             (state === 'ALL' || zone.state === state);
                if (pass) matchCount++;
                return pass;
            };

            renderCandidateLayers(filterFn);

            document.getElementById('filterCountBadge').innerText = `${matchCount}/${CANDIDATE_ZONES.length}`;
            document.getElementById('screenerSummary').innerText = `${matchCount} of ${CANDIDATE_ZONES.length} candidate site zones qualify`;
        }

        function resetFilters() {
            document.getElementById('filterGhi').value = 5.0;
            document.getElementById('filterSlope').value = 2.5;
            document.getElementById('filterGrid').value = 40;
            document.getElementById('filterState').value = "ALL";
            applyFilters();
        }

        function downloadCSV() {
            let csv = "rank,name,district,state,latitude,longitude,potential_capacity_mw,area_acres,ghi_daily,dni_daily,suitability_score,terrain_slope_pct,land_type,nearest_substation,substation_distance_km,lcoe_inr_kwh\\n";
            CANDIDATE_ZONES.forEach(z => {
                csv += `${z.rank},"${z.name}","${z.district}","${z.state}",${z.lat},${z.lon},${z.capacity_mw},${z.area_acres},${z.ghi},${z.dni},${z.score},${z.slope},"${z.land_type}","${z.substation}",${z.grid_dist_km},${z.lcoe}\\n`;
            });
            const blob = new Blob([csv], { type: 'text/csv' });
            const url = window.URL.createObjectURL(blob);
            const a = document.createElement('a');
            a.setAttribute('href', url);
            a.setAttribute('download', 'india_candidate_solar_parks_raydium.csv');
            a.click();
        }

        window.onload = () => {
            initMap();
            lucide.createIcons();
            calculateCustomYield(500);
        };
    </script>
</body>
</html>"""

        with open(output_html, "w", encoding="utf-8") as f:
            f.write(html_template)

        file_size_kb = os.path.getsize(output_html) / 1024
        logger.info(f"Dashboard generated: {output_html} ({file_size_kb:.1f} KB, {height}x{width} grid matrix, 15 candidate zones)")
        return output_html

