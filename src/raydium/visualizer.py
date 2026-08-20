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
        raster_dict: Dict,
        output_html: str = "index.html",
        surface_png_path: str = "solar_suitability_surface.png",
    ) -> str:
        """Generate a continuous, precision GIS workbench with candidate site zones and satellite terrain inspection."""
        logger.info(f"Generating precision interactive dashboard at {output_html}...")

        # Ensure transparent raster surface is generated
        self.export_transparent_surface_png(raster_dict, output_surface=surface_png_path)

        bounds = raster_dict["bounds"]  # [minx, miny, maxx, maxy] -> [min_lon, min_lat, max_lon, max_lat]
        bounds_json = [
            [float(bounds[1]), float(bounds[0])],  # South-West: [lat, lon]
            [float(bounds[3]), float(bounds[2])]   # North-East: [lat, lon]
        ]

        # Candidate Zones Data
        candidate_zones_data = [
            {
                "rank": z.rank,
                "name": z.name,
                "district": z.district,
                "state": z.state,
                "lat": z.latitude,
                "lon": z.longitude,
                "capacity_mw": z.potential_capacity_mw,
                "area_acres": z.estimated_area_acres,
                "ghi": z.ghi_daily,
                "dni": z.dni_daily,
                "score": z.suitability_score,
                "slope": z.terrain_slope_pct,
                "land_type": z.land_type,
                "substation": z.nearest_substation,
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
                "name": p.name,
                "state": p.state,
                "capacity": p.capacity_mw,
                "lat": p.latitude,
                "lon": p.longitude,
                "status": p.status,
                "developer": p.developer,
                "substation": p.substation,
                "year": p.commissioned_year,
                "desc": p.description,
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
    
    <!-- Google Fonts: Inter & JetBrains Mono -->
    <link rel="preconnect" href="https://fonts.googleapis.com">
    <link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
    <link href="https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&family=JetBrains+Mono:wght@400;500;600&display=swap" rel="stylesheet">

    <!-- Tailwind CSS CDN -->
    <script src="https://cdn.tailwindcss.com"></script>
    <script>
        tailwind.config = {
            darkMode: 'class',
            theme: {
                extend: {
                    fontFamily: {
                        sans: ['Inter', 'sans-serif'],
                        mono: ['JetBrains Mono', 'monospace'],
                    },
                    colors: {
                        slate: {
                            850: '#111827',
                            900: '#0b0f19',
                            950: '#050811',
                        }
                    }
                }
            }
        }
    </script>
    
    <!-- Leaflet CSS & JS -->
    <link rel="stylesheet" href="https://unpkg.com/leaflet@1.9.4/dist/leaflet.css" />
    <script src="https://unpkg.com/leaflet@1.9.4/dist/leaflet.js"></script>
    <!-- Lucide Icons -->
    <script src="https://unpkg.com/lucide@latest"></script>

    <style>
        body { font-family: 'Inter', sans-serif; }
        #map { height: calc(100vh - 64px); width: 100%; z-index: 10; background: #050811; }
        .custom-popup .leaflet-popup-content-wrapper {
            background: #0b0f19;
            color: #f1f5f9;
            border: 1px solid #1e293b;
            border-radius: 10px;
            box-shadow: 0 20px 25px -5px rgb(0 0 0 / 0.7);
            padding: 4px;
        }
        .custom-popup .leaflet-popup-tip { background: #0b0f19; }
        .glass-panel {
            background: rgba(11, 15, 25, 0.90);
            backdrop-filter: blur(16px);
            border: 1px solid rgba(30, 41, 59, 0.75);
        }
        ::-webkit-scrollbar { width: 6px; }
        ::-webkit-scrollbar-track { background: #0b0f19; }
        ::-webkit-scrollbar-thumb { background: #334155; border-radius: 4px; }
        ::-webkit-scrollbar-thumb:hover { background: #475569; }
    </style>
</head>
<body class="bg-slate-950 text-slate-100 font-sans antialiased overflow-hidden flex flex-col h-screen">

    <!-- Top Engineering Navbar -->
    <header class="h-16 bg-slate-900 border-b border-slate-800 px-6 py-2.5 flex items-center justify-between z-30">
        <div class="flex items-center space-x-3.5">
            <div class="w-9 h-9 rounded-lg bg-amber-500/10 border border-amber-500/30 flex items-center justify-center text-amber-400">
                <i data-lucide="sun" class="w-5 h-5"></i>
            </div>
            <div>
                <div class="flex items-center gap-2">
                    <h1 class="text-base font-semibold tracking-tight text-white">RAYDIUM</h1>
                    <span class="text-[10px] font-mono bg-emerald-500/20 text-emerald-300 px-2 py-0.5 rounded border border-emerald-500/30">SITING TOOL</span>
                    <span class="text-[11px] text-slate-400 font-normal hidden sm:inline">| Solar Siting for India</span>
                </div>
            </div>
        </div>

        <!-- Candidate Zones Quick Selector -->
        <div class="hidden md:flex items-center space-x-3 text-xs">
            <label class="text-slate-400 text-[11px] font-medium flex items-center gap-1.5">
                <i data-lucide="target" class="w-3.5 h-3.5 text-emerald-400"></i> Jump to Site:
            </label>
            <select id="siteSelect" onchange="zoomToCandidate(this.value)" class="bg-slate-800 text-slate-200 text-xs rounded-lg px-3 py-1.5 border border-slate-700 font-medium focus:outline-none focus:border-amber-500 cursor-pointer">
                <option value="">-- Select Candidate Solar Zone --</option>
            </select>
        </div>

        <!-- Utility Buttons -->
        <div class="flex items-center space-x-2.5">
            <button onclick="toggleSidePanel()" class="flex items-center gap-1.5 px-3 py-1.5 rounded-lg bg-slate-800 hover:bg-slate-700 text-xs font-medium text-slate-200 border border-slate-700 transition">
                <i data-lucide="sliders" class="w-3.5 h-3.5 text-amber-400"></i> Sizing Tool
            </button>
            <button onclick="downloadCSV()" class="flex items-center gap-1.5 px-3 py-1.5 rounded-lg bg-emerald-600 hover:bg-emerald-500 text-xs font-medium text-white font-semibold shadow-sm transition">
                <i data-lucide="download" class="w-3.5 h-3.5"></i> Export CSV
            </button>
        </div>
    </header>

    <!-- Main Workspace -->
    <div class="relative flex-1 w-full overflow-hidden">
        <!-- Interactive Map Container -->
        <div id="map"></div>

        <!-- Floating Legend & Layer Controller (Bottom Left) -->
        <div class="absolute bottom-6 left-6 z-20 glass-panel rounded-xl p-4 shadow-xl max-w-xs w-full text-xs">
            <div class="flex items-center justify-between mb-2.5">
                <h4 class="font-semibold text-slate-200 uppercase tracking-wider text-[11px]">Map Layers</h4>
                <span class="text-[10px] text-emerald-400 font-mono">10 Ranked Sites</span>
            </div>
            
            <div class="space-y-2 text-[11px] mb-3">
                <div class="flex items-center justify-between">
                    <span class="flex items-center gap-2 text-slate-300">
                        <span class="w-3 h-3 rounded-full bg-emerald-500 border border-white flex items-center justify-center text-[8px] font-bold text-slate-950">★</span>
                        Candidate Siting Zones
                    </span>
                    <span class="text-emerald-400 font-mono font-semibold">New Sites</span>
                </div>
                <div class="flex items-center justify-between">
                    <span class="flex items-center gap-2 text-slate-300">
                        <span class="w-2.5 h-2.5 rounded-full bg-sky-400"></span>
                        Operational Mega Solar Parks
                    </span>
                    <span class="text-sky-400 font-mono">Bhadla/Khavda</span>
                </div>
                <div class="flex items-center justify-between">
                    <span class="flex items-center gap-2 text-slate-300">
                        <span class="w-2.5 h-2.5 rounded-sm bg-gradient-to-r from-indigo-500 via-amber-500 to-yellow-300"></span>
                        Solar Insolation Surface
                    </span>
                    <span class="text-amber-400 font-mono">Continuous</span>
                </div>
            </div>

            <!-- Layer Controls -->
            <div class="pt-2.5 border-t border-slate-800 space-y-2">
                <div class="flex justify-between items-center text-[11px]">
                    <span class="text-slate-400">Basemap:</span>
                    <div class="flex rounded bg-slate-800 p-0.5 text-[10px]">
                        <button id="btnDark" onclick="switchBasemap('dark')" class="px-2 py-0.5 rounded bg-slate-700 text-white font-medium">Dark</button>
                        <button id="btnSat" onclick="switchBasemap('satellite')" class="px-2 py-0.5 rounded text-slate-400 hover:text-white">Satellite</button>
                    </div>
                </div>

                <!-- Surface Opacity Slider -->
                <div>
                    <div class="flex justify-between text-slate-400 text-[10px] mb-1">
                        <span>Surface Opacity</span>
                        <span id="opacityVal" class="text-slate-200 font-mono">75%</span>
                    </div>
                    <input id="heatOpacity" type="range" min="0" max="100" value="75" class="w-full accent-amber-400 h-1.5 bg-slate-800 rounded cursor-pointer" oninput="updateSurfaceOpacity(this.value)">
                </div>
            </div>
        </div>

        <!-- Floating Siting Feasibility Drawer (Right Side) -->
        <div id="sidePanel" class="absolute top-4 right-4 bottom-6 z-20 glass-panel rounded-xl p-5 shadow-2xl w-96 overflow-y-auto flex flex-col justify-between transition-all duration-250 transform translate-x-0">
            <div>
                <div class="flex items-center justify-between pb-3 border-b border-slate-800 mb-3.5">
                    <div>
                        <h3 class="font-semibold text-slate-100 text-sm flex items-center gap-2">
                            <i data-lucide="map-pin" class="w-4 h-4 text-emerald-400"></i> Site Details & Sizing
                        </h3>
                        <p id="siteSubtitle" class="text-[11px] text-slate-400 mt-0.5">Rank #1 Candidate Siting Zone</p>
                    </div>
                    <button onclick="toggleSidePanel()" class="text-slate-400 hover:text-white p-1 rounded hover:bg-slate-800 transition">
                        <i data-lucide="x" class="w-4 h-4"></i>
                    </button>
                </div>

                <!-- Selected Location Dossier -->
                <div class="space-y-3">
                    <div class="bg-slate-900/90 rounded-lg p-3 border border-slate-800 text-xs">
                        <div class="flex justify-between items-start mb-1">
                            <h4 id="siteName" class="font-semibold text-emerald-400 text-sm">Jaisalmer-Phalodi Hyper-Arid Corridor</h4>
                            <span id="siteBadge" class="text-[10px] font-mono font-bold bg-emerald-500/20 text-emerald-300 px-1.5 py-0.5 rounded border border-emerald-500/30">RANK #1</span>
                        </div>
                        <div id="siteLocation" class="text-slate-300 text-[11px]">Jaisalmer / Phalodi, Rajasthan</div>
                        <div id="siteCoords" class="text-[10px] font-mono text-slate-400 mt-0.5">26.9500° N, 70.9200° E</div>
                    </div>

                    <!-- Key Technical Metrics -->
                    <div class="grid grid-cols-2 gap-2 text-xs">
                        <div class="bg-slate-900/70 p-2.5 rounded-lg border border-slate-800">
                            <span class="text-slate-400 block text-[10px] uppercase">Daily GHI</span>
                            <span id="valGhi" class="text-base font-bold text-amber-400 font-mono">6.35</span>
                            <span class="text-[10px] text-slate-500 block">kWh/m²/day</span>
                        </div>
                        <div class="bg-slate-900/70 p-2.5 rounded-lg border border-slate-800">
                            <span class="text-slate-400 block text-[10px] uppercase">Potential Capacity</span>
                            <span id="valCapacity" class="text-base font-bold text-emerald-400 font-mono">8,000 MW</span>
                            <span id="valArea" class="text-[10px] text-slate-500 block">~35,000 Acres</span>
                        </div>
                    </div>

                    <!-- Detailed Siting Parameters -->
                    <div class="bg-slate-900/80 rounded-lg p-3 border border-slate-800 text-xs space-y-2">
                        <h5 class="font-semibold text-slate-300 text-[10px] uppercase tracking-wider">Geospatial & Grid Readiness</h5>
                        
                        <div class="flex justify-between items-center py-0.5 border-b border-slate-800/60">
                            <span class="text-slate-400">Terrain Slope:</span>
                            <span id="valSlope" class="font-mono text-slate-200">0.8 % (Ultra-flat)</span>
                        </div>
                        <div class="flex justify-between items-center py-0.5 border-b border-slate-800/60">
                            <span class="text-slate-400">Land Category:</span>
                            <span id="valLand" class="text-right text-slate-300 text-[11px]">Barren Sandy Desert</span>
                        </div>
                        <div class="py-0.5 border-b border-slate-800/60">
                            <span class="text-slate-400 block text-[10px]">Nearest ISTS Substation:</span>
                            <span id="valSubstation" class="text-slate-200 font-medium text-[11px]">765/400 kV Fatehgarh-II ISTS (18 km)</span>
                        </div>
                        <div class="flex justify-between items-center py-0.5 border-b border-slate-800/60">
                            <span class="text-slate-400">Estimated LCOE:</span>
                            <span id="valLcoe" class="font-mono font-semibold text-emerald-400">₹ 2.35 / kWh</span>
                        </div>
                        <div class="py-0.5">
                            <span class="text-slate-400 block text-[10px] mb-0.5">Key Site Advantages:</span>
                            <p id="valAdvantages" class="text-slate-300 text-[11px] leading-relaxed italic">Highest GHI in South Asia (>325 sunny days), zero agricultural conflict, direct access to Green Energy Corridor.</p>
                        </div>
                    </div>

                    <!-- Custom Sizing Calculator -->
                    <div class="pt-1">
                        <div class="flex justify-between items-center mb-1">
                            <label class="text-xs font-medium text-slate-200">Test Custom Land Area</label>
                            <span id="landAreaDisplay" class="text-xs font-mono font-semibold text-amber-400">500 Acres</span>
                        </div>
                        <input id="landAreaSlider" type="range" min="10" max="5000" step="25" value="500" class="w-full accent-amber-400 h-1.5 bg-slate-800 rounded cursor-pointer" oninput="calculateCustomYield(this.value)">
                        <div class="bg-slate-900/90 rounded-lg p-2.5 border border-slate-800 text-[11px] mt-1.5 grid grid-cols-2 gap-1.5 font-mono">
                            <div>Capacity: <span id="customMw" class="text-white font-bold">111 MW</span></div>
                            <div>Gen: <span id="customGen" class="text-amber-400 font-bold">201 GWh/yr</span></div>
                            <div>Revenue: <span id="customRev" class="text-emerald-400 font-bold">₹ 52.3 Cr</span></div>
                            <div>CO₂: <span id="customCo2" class="text-emerald-300 font-bold">165k T</span></div>
                        </div>
                    </div>
                </div>
            </div>

            <!-- Engineering footer note -->
            <div class="mt-3.5 pt-2.5 border-t border-slate-800 text-[10px] text-slate-500 flex items-center gap-1.5">
                <i data-lucide="check-circle" class="w-3.5 h-3.5 text-emerald-400 shrink-0"></i>
                <span>Click any candidate zone star or click anywhere on India to inspect local siting metrics.</span>
            </div>
        </div>
    </div>

    <!-- Data Injection & Map Engine -->
    <script>
        const CANDIDATE_ZONES = """ + json.dumps(candidate_zones_data) + """;
        const MEGA_PARKS = """ + json.dumps(parks_data) + """;
        const SURFACE_BOUNDS = """ + json.dumps(bounds_json) + """;
        const SURFACE_PNG = "solar_suitability_surface.png";
        const INDIA_GEOJSON = """ + geojson_str + """;

        let map, surfaceOverlay, candidateLayer, parksLayer, geojsonLayer;
        let darkTiles, satelliteTiles;
        let currentGhi = 6.35;

        function initMap() {
            map = L.map('map', {
                center: [23.5, 78.5],
                zoom: 5,
                minZoom: 4,
                maxZoom: 17,
                zoomControl: false
            });

            L.control.zoom({ position: 'topleft' }).addTo(map);

            // Dark Matter Basemap
            darkTiles = L.tileLayer('https://{s}.basemaps.cartocdn.com/dark_all/{z}/{x}/{y}{r}.png', {
                attribution: '&copy; OpenStreetMap &copy; CARTO',
                subdomains: 'abcd',
                maxZoom: 19
            }).addTo(map);

            // High-Resolution Esri Satellite Imagery
            satelliteTiles = L.tileLayer('https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}', {
                attribution: '&copy; Esri World Imagery'
            });

            // 1. Continuous Seamless Interpolated Solar Surface Overlay (Zero Dots!)
            surfaceOverlay = L.imageOverlay(SURFACE_PNG + '?v=' + Date.now(), SURFACE_BOUNDS, {
                opacity: 0.75,
                interactive: false,
                crossOrigin: true,
                zIndex: 5
            }).addTo(map);

            // 2. Official Survey of India Boundary
            geojsonLayer = L.geoJSON(INDIA_GEOJSON, {
                style: {
                    color: '#94a3b8',
                    weight: 1.2,
                    fillOpacity: 0.0,
                    interactive: false
                }
            }).addTo(map);

            // 3. Ranked Candidate Siting Zones Layer
            candidateLayer = L.layerGroup().addTo(map);
            const selectEl = document.getElementById('siteSelect');

            CANDIDATE_ZONES.forEach((zone, idx) => {
                // Populate Dropdown
                const opt = document.createElement('option');
                opt.value = idx;
                opt.innerText = `#${zone.rank} - ${zone.name} (${zone.capacity_mw.toLocaleString()} MW)`;
                selectEl.appendChild(opt);

                // Add Candidate Marker (Green Star / Highlight Ring)
                const marker = L.circleMarker([zone.lat, zone.lon], {
                    radius: 12,
                    fillColor: '#10b981',
                    color: '#ffffff',
                    weight: 2.5,
                    opacity: 1,
                    fillOpacity: 0.95
                });

                const popupContent = `
                    <div class="p-2 custom-popup text-xs space-y-1.5">
                        <div class="flex items-center justify-between gap-2">
                            <span class="font-bold text-emerald-400 text-sm">#${zone.rank} ${zone.name}</span>
                            <span class="text-[10px] font-mono bg-emerald-500/20 text-emerald-300 px-1.5 py-0.5 rounded font-bold">${zone.capacity_mw.toLocaleString()} MW</span>
                        </div>
                        <div class="text-slate-300"><strong>Location:</strong> ${zone.district}, ${zone.state}</div>
                        <div class="text-slate-300"><strong>Daily GHI:</strong> ${zone.ghi} kWh/m²/day | <strong>DNI:</strong> ${zone.dni}</div>
                        <div class="text-slate-300"><strong>Terrain:</strong> ${zone.slope}% slope (${zone.land_type})</div>
                        <div class="text-slate-300"><strong>ISTS Grid:</strong> ${zone.substation} (${zone.grid_dist_km} km)</div>
                        <div class="text-slate-300"><strong>Est. LCOE:</strong> ₹${zone.lcoe}/kWh</div>
                        <p class="text-slate-400 text-[11px] mt-1 italic border-t border-slate-800 pt-1">${zone.advantages}</p>
                    </div>
                `;
                marker.bindPopup(popupContent);
                marker.on('click', () => {
                    displayCandidateDossier(zone);
                });
                candidateLayer.addLayer(marker);
            });

            // 4. Operational Mega Solar Parks Layer (Blue Markers)
            parksLayer = L.layerGroup().addTo(map);
            MEGA_PARKS.forEach(park => {
                const marker = L.circleMarker([park.lat, park.lon], {
                    radius: Math.max(5, Math.min(13, park.capacity / 250)),
                    fillColor: '#38bdf8',
                    color: '#ffffff',
                    weight: 1.5,
                    opacity: 1,
                    fillOpacity: 0.8
                });

                const popupContent = `
                    <div class="p-2 custom-popup text-xs space-y-1.5">
                        <div class="font-semibold text-sky-400 text-sm flex items-center justify-between">
                            <span>${park.name}</span>
                            <span class="text-[10px] font-mono bg-sky-500/20 text-sky-300 px-1.5 py-0.5 rounded">${park.capacity.toLocaleString()} MW</span>
                        </div>
                        <div class="text-slate-300"><strong>State:</strong> ${park.state} | <strong>Status:</strong> ${park.status}</div>
                        <div class="text-slate-300"><strong>Developer:</strong> ${park.developer}</div>
                        <div class="text-slate-400 text-[11px]"><strong>Substation:</strong> ${park.substation || 'Grid Substation'}</div>
                    </div>
                `;
                marker.bindPopup(popupContent);
                parksLayer.addLayer(marker);
            });

            // 5. Map Click Custom Inspector
            map.on('click', (e) => {
                const lat = e.latlng.lat;
                const lon = e.latlng.lng;
                selectCustomPoint(lat, lon);
            });

            // Default display Rank #1
            displayCandidateDossier(CANDIDATE_ZONES[0]);
        }

        function displayCandidateDossier(zone) {
            currentGhi = zone.ghi;
            document.getElementById('siteSubtitle').innerText = `Rank #${zone.rank} Candidate Siting Zone`;
            document.getElementById('siteName').innerText = zone.name;
            document.getElementById('siteBadge').innerText = `RANK #${zone.rank}`;
            document.getElementById('siteLocation').innerText = `${zone.district}, ${zone.state}`;
            document.getElementById('siteCoords').innerText = `${zone.lat.toFixed(4)}° N, ${zone.lon.toFixed(4)}° E`;
            document.getElementById('valGhi').innerText = zone.ghi.toFixed(2);
            document.getElementById('valCapacity').innerText = `${zone.capacity_mw.toLocaleString()} MW`;
            document.getElementById('valArea').innerText = `~${zone.area_acres.toLocaleString()} Acres`;
            document.getElementById('valSlope').innerText = `${zone.slope} % (${zone.slope < 1.5 ? 'Ultra-flat' : 'Gentle slope'})`;
            document.getElementById('valLand').innerText = zone.land_type;
            document.getElementById('valSubstation').innerText = `${zone.substation} (${zone.grid_dist_km} km)`;
            document.getElementById('valLcoe').innerText = `₹ ${zone.lcoe.toFixed(2)} / kWh`;
            document.getElementById('valAdvantages').innerText = zone.advantages;

            calculateCustomYield(document.getElementById('landAreaSlider').value);
        }

        function selectCustomPoint(lat, lon) {
            // Approximation for custom clicked point
            let estGhi = 5.2 - 0.04 * Math.abs(lat - 22.0);
            if (lon >= 69.0 && lon <= 76.0 && lat >= 23.0 && lat <= 29.5) estGhi += 1.0;
            else if (lon >= 74.0 && lon <= 79.0 && lat >= 11.0 && lat <= 18.0) estGhi += 0.45;
            estGhi = Math.max(3.0, Math.min(6.5, estGhi));
            currentGhi = estGhi;

            document.getElementById('siteSubtitle').innerText = "Custom Investigated Site";
            document.getElementById('siteName').innerText = "Custom Siting Investigation";
            document.getElementById('siteBadge').innerText = "CUSTOM";
            document.getElementById('siteLocation').innerText = `Lat ${lat.toFixed(2)}°, Lon ${lon.toFixed(2)}°`;
            document.getElementById('siteCoords').innerText = `${lat.toFixed(4)}° N, ${lon.toFixed(4)}° E`;
            document.getElementById('valGhi').innerText = estGhi.toFixed(2);
            document.getElementById('valCapacity').innerText = "Variable";
            document.getElementById('valArea').innerText = "User Defined";
            document.getElementById('valSlope').innerText = lat > 32.0 ? "> 5% (Mountainous)" : "< 2% (Plains)";
            document.getElementById('valLand').innerText = "Custom Land Siting";
            document.getElementById('valSubstation').innerText = "Local DISCOM / PGCIL Substation";
            document.getElementById('valLcoe').innerText = `₹ ${(2.80 - (estGhi - 4.5)*0.2).toFixed(2)} / kWh`;
            document.getElementById('valAdvantages').innerText = "Site evaluated based on localized solar insolation and terrain gradients.";

            calculateCustomYield(document.getElementById('landAreaSlider').value);
        }

        function zoomToCandidate(idx) {
            if (idx === "" || idx === null) return;
            const zone = CANDIDATE_ZONES[parseInt(idx)];
            map.flyTo([zone.lat, zone.lon], 9, { duration: 1.5 });
            displayCandidateDossier(zone);
        }

        function calculateCustomYield(landAcres) {
            document.getElementById('landAreaDisplay').innerText = `${landAcres} Acres`;

            const mw = (landAcres / 4.5);
            const kw = mw * 1000;
            const annualGhi = currentGhi * 365;
            const specificYield = annualGhi * 0.78;
            const genKwh = kw * specificYield;
            const genGwh = genKwh / 1000000;
            const revCr = (genKwh * 2.60) / 10000000;
            const co2Tonnes = (genKwh * 0.82) / 1000;

            document.getElementById('customMw').innerText = `${mw.toFixed(0)} MW`;
            document.getElementById('customGen').innerText = `${genGwh.toFixed(1)} GWh/yr`;
            document.getElementById('customRev').innerText = `₹ ${revCr.toFixed(1)} Cr`;
            document.getElementById('customCo2').innerText = `${Math.round(co2Tonnes / 1000)}k T`;
        }

        function updateSurfaceOpacity(val) {
            document.getElementById('opacityVal').innerText = `${val}%`;
            if (surfaceOverlay) {
                surfaceOverlay.setOpacity(val / 100.0);
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
            const panel = document.getElementById('sidePanel');
            panel.classList.toggle('translate-x-full');
            panel.classList.toggle('translate-x-0');
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

        logger.info(f"Dashboard generated successfully at {output_html}")
        return output_html
