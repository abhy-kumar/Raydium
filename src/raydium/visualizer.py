"""High-resolution cartography and clean interactive web GIS dashboard generator."""

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

from raydium.models import MEGA_SOLAR_PARKS, SUITABILITY_TIERS

logger = logging.getLogger(__name__)


class MapVisualizer:
    """Renders cartographic publication maps and interactive HTML5 GIS dashboards."""

    def __init__(self, geojson_path: str = "india-soi.geojson"):
        self.geojson_path = geojson_path
        if not os.path.exists(geojson_path):
            raise FileNotFoundError(f"Boundary GeoJSON not found at: {geojson_path}")
        self.india_gdf = gpd.read_file(geojson_path)
        if self.india_gdf.crs != "EPSG:4326":
            self.india_gdf = self.india_gdf.to_crs("EPSG:4326")

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

        # Custom Solar Thermal Palette: Deep Obsidian -> Indigo -> Crimson -> Amber -> Solar Yellow
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

        # Boundary outline from official Survey of India GeoJSON
        self.india_gdf.boundary.plot(ax=ax, color="#cbd5e1", linewidth=1.1, alpha=0.85)

        # Overlay major operational and planned mega solar parks
        if show_parks:
            park_lons = [p.longitude for p in MEGA_SOLAR_PARKS]
            park_lats = [p.latitude for p in MEGA_SOLAR_PARKS]
            park_sizes = [min(320, max(50, p.capacity_mw / 25.0)) for p in MEGA_SOLAR_PARKS]

            ax.scatter(
                park_lons,
                park_lats,
                s=park_sizes,
                c="#38bdf8",
                edgecolors="#ffffff",
                linewidths=1.4,
                zorder=10,
                alpha=0.95,
                label="Major Solar Parks",
            )

            # Annotate top 5 solar parks by capacity
            top_parks = sorted(MEGA_SOLAR_PARKS, key=lambda x: x.capacity_mw, reverse=True)[:5]
            for p in top_parks:
                ax.annotate(
                    f"{p.name}\n({p.capacity_mw:,.0f} MW)",
                    xy=(p.longitude, p.latitude),
                    xytext=(12, 10),
                    textcoords="offset points",
                    color="#ffffff",
                    fontsize=8.5,
                    fontfamily="sans-serif",
                    fontweight="bold",
                    bbox=dict(boxstyle="round,pad=0.3", fc="#111827", ec="#38bdf8", alpha=0.9, lw=1),
                    arrowprops=dict(arrowstyle="->", color="#38bdf8", lw=1.2),
                    zorder=15,
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
            "RAYDIUM SOLAR PLATFORM\n"
            f"Mean Daily GHI: ~5.18 kWh/m2/day\n"
            f"Prime Siting Corridors: Thar Desert, Kutch, Deccan Plateau\n"
            f"Tracked Projects: {len(MEGA_SOLAR_PARKS)} Sites (>45 GW)\n"
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
        output_html: str = "index.html",
        summary_stats: Optional[Dict] = None,
    ) -> str:
        """Generate a clean, responsive HTML5/Leaflet solar engineering workbench."""
        logger.info(f"Generating interactive dashboard at {output_html}...")

        # Sample for client-side rendering performance if points exceed threshold
        sample_df = df.sample(n=min(len(df), 3000), random_state=42) if len(df) > 3000 else df

        points_records = []
        val_col = "ghi_daily" if "ghi_daily" in sample_df.columns else "potential"

        for _, row in sample_df.iterrows():
            points_records.append({
                "lat": round(float(row["latitude"]), 4),
                "lon": round(float(row["longitude"]), 4),
                "ghi": round(float(row[val_col]), 2),
                "suit": round(float(row.get("suitability_score", 70.0)), 1),
                "tier": row.get("suitability_tier", "Tier 2 - Highly Suitable"),
                "temp": round(float(row.get("temp_ambient", 25.0)), 1),
            })

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

        # Simplified boundary GeoJSON for fast client load
        simplified_gdf = self.india_gdf.copy()
        simplified_gdf.geometry = self.india_gdf.geometry.simplify(0.012, preserve_topology=True)
        geojson_str = simplified_gdf.to_json()

        html_template = """<!DOCTYPE html>
<html lang="en" class="dark">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Raydium — Solar Plant Siting & Resource Intelligence for India</title>
    
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
    <script src="https://unpkg.com/leaflet.heat@0.2.0/dist/leaflet-heat.js"></script>
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
            box-shadow: 0 20px 25px -5px rgb(0 0 0 / 0.6);
            padding: 2px;
        }
        .custom-popup .leaflet-popup-tip { background: #0b0f19; }
        .glass-panel {
            background: rgba(11, 15, 25, 0.88);
            backdrop-filter: blur(14px);
            border: 1px solid rgba(30, 41, 59, 0.7);
        }
        /* Custom scrollbar */
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
                    <span class="text-[10px] font-mono bg-slate-800 text-slate-300 px-2 py-0.5 rounded border border-slate-700">v2.0</span>
                    <span class="text-[11px] text-slate-400 font-normal hidden sm:inline">— Indian Solar Plant Siting & Resource Intelligence</span>
                </div>
            </div>
        </div>

        <!-- Key Metrics Strip -->
        <div class="hidden lg:flex items-center space-x-6 text-xs font-mono">
            <div>
                <span class="text-slate-400 block text-[10px] uppercase tracking-wider">National Mean GHI</span>
                <span class="text-amber-400 font-semibold">5.18 kWh/m²/day</span>
            </div>
            <div class="h-6 w-px bg-slate-800"></div>
            <div>
                <span class="text-slate-400 block text-[10px] uppercase tracking-wider">Peak Siting Hotspot</span>
                <span class="text-emerald-400 font-semibold">Thar Desert & Kutch (>6.2 kWh)</span>
            </div>
            <div class="h-6 w-px bg-slate-800"></div>
            <div>
                <span class="text-slate-400 block text-[10px] uppercase tracking-wider">Tracked Mega Parks</span>
                <span class="text-sky-400 font-semibold">12 Projects (45.3 GW)</span>
            </div>
        </div>

        <!-- Utility Buttons -->
        <div class="flex items-center space-x-2.5">
            <button onclick="toggleSidePanel()" class="flex items-center gap-1.5 px-3 py-1.5 rounded-lg bg-slate-800 hover:bg-slate-700 text-xs font-medium text-slate-200 border border-slate-700 transition">
                <i data-lucide="sliders" class="w-3.5 h-3.5 text-amber-400"></i> Siting Tool
            </button>
            <button onclick="downloadCSV()" class="flex items-center gap-1.5 px-3 py-1.5 rounded-lg bg-amber-500 hover:bg-amber-400 text-xs font-medium text-slate-950 font-semibold shadow-sm transition">
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
            <div class="flex items-center justify-between mb-3">
                <h4 class="font-semibold text-slate-200 uppercase tracking-wider text-[11px]">Siting Suitability Index</h4>
                <span class="text-[10px] text-slate-400 font-mono">0 - 100</span>
            </div>
            
            <div class="space-y-2 text-[11px]">
                <div class="flex items-center justify-between">
                    <span class="flex items-center gap-2 text-slate-300">
                        <span class="w-2.5 h-2.5 rounded-full bg-emerald-500"></span>
                        Tier 1 — Prime (>85)
                    </span>
                    <span class="text-emerald-400 font-mono">Mega Utility (>5.8 kWh)</span>
                </div>
                <div class="flex items-center justify-between">
                    <span class="flex items-center gap-2 text-slate-300">
                        <span class="w-2.5 h-2.5 rounded-full bg-blue-500"></span>
                        Tier 2 — High (70-85)
                    </span>
                    <span class="text-blue-400 font-mono">Commercial / Utility</span>
                </div>
                <div class="flex items-center justify-between">
                    <span class="flex items-center gap-2 text-slate-300">
                        <span class="w-2.5 h-2.5 rounded-full bg-amber-500"></span>
                        Tier 3 — Moderate (50-70)
                    </span>
                    <span class="text-amber-400 font-mono">Rooftop / Agri-PV</span>
                </div>
                <div class="flex items-center justify-between">
                    <span class="flex items-center gap-2 text-slate-300">
                        <span class="w-2.5 h-2.5 rounded-full bg-red-500"></span>
                        Tier 4 — Low (<50)
                    </span>
                    <span class="text-red-400 font-mono">Topographic / Cloud</span>
                </div>
            </div>

            <!-- Heat Layer Opacity -->
            <div class="mt-3.5 pt-3 border-t border-slate-800/80">
                <div class="flex justify-between text-slate-400 text-[11px] mb-1.5">
                    <span>Insolation Layer Opacity</span>
                    <span id="opacityVal" class="text-slate-200 font-mono">75%</span>
                </div>
                <input id="heatOpacity" type="range" min="10" max="100" value="75" class="w-full accent-amber-400 h-1.5 bg-slate-800 rounded cursor-pointer" oninput="updateHeatOpacity(this.value)">
            </div>
        </div>

        <!-- Floating Siting & Yield Engineering Drawer (Right Side) -->
        <div id="sidePanel" class="absolute top-4 right-4 bottom-6 z-20 glass-panel rounded-xl p-5 shadow-2xl w-96 overflow-y-auto flex flex-col justify-between transition-all duration-250 transform translate-x-0">
            <div>
                <div class="flex items-center justify-between pb-3 border-b border-slate-800 mb-4">
                    <div>
                        <h3 class="font-semibold text-slate-100 text-sm flex items-center gap-2">
                            <i data-lucide="compass" class="w-4 h-4 text-amber-400"></i> Location Siting Analysis
                        </h3>
                        <p class="text-[11px] text-slate-400 mt-0.5">PV Capacity & Sizing Calculator</p>
                    </div>
                    <button onclick="toggleSidePanel()" class="text-slate-400 hover:text-white p-1 rounded hover:bg-slate-800 transition">
                        <i data-lucide="x" class="w-4 h-4"></i>
                    </button>
                </div>

                <!-- Selected Location Details -->
                <div class="space-y-3.5">
                    <div class="bg-slate-900/90 rounded-lg p-3 border border-slate-800 text-xs">
                        <div class="text-slate-400 text-[11px] mb-0.5">Selected Coordinates</div>
                        <div id="selectedCoords" class="text-sm font-semibold font-mono text-amber-400">27.5386° N, 71.9167° E</div>
                        <div id="selectedRegion" class="text-[11px] text-slate-300 mt-1">Bhadla Region, Rajasthan</div>
                    </div>

                    <!-- Insolation KPI Grid -->
                    <div class="grid grid-cols-2 gap-2 text-xs">
                        <div class="bg-slate-900/70 p-3 rounded-lg border border-slate-800">
                            <span class="text-slate-400 block text-[10px] uppercase">Daily GHI</span>
                            <span id="valGhi" class="text-base font-bold text-amber-400 font-mono">6.24</span>
                            <span class="text-[10px] text-slate-500 block">kWh/m²/day</span>
                        </div>
                        <div class="bg-slate-900/70 p-3 rounded-lg border border-slate-800">
                            <span class="text-slate-400 block text-[10px] uppercase">Suitability</span>
                            <span id="valScore" class="text-base font-bold text-emerald-400 font-mono">96.5 / 100</span>
                            <span id="valTier" class="text-[10px] text-emerald-400 block font-medium">Tier 1 — Prime</span>
                        </div>
                    </div>

                    <!-- Proposed Land Sizing Slider -->
                    <div class="pt-1">
                        <div class="flex justify-between items-center mb-1.5">
                            <label class="text-xs font-medium text-slate-200">Proposed Land Area</label>
                            <span id="landAreaDisplay" class="text-xs font-mono font-semibold text-amber-400">200 Acres</span>
                        </div>
                        <input id="landAreaSlider" type="range" min="10" max="2500" step="10" value="200" class="w-full accent-amber-400 h-1.5 bg-slate-800 rounded cursor-pointer" oninput="calculateYield(this.value)">
                        <div class="flex justify-between text-[10px] text-slate-500 font-mono mt-1">
                            <span>10 Acres (2.5 MW)</span>
                            <span>2,500 Acres (600 MW)</span>
                        </div>
                    </div>

                    <!-- Yield & Economics Output Card -->
                    <div class="bg-slate-900/90 rounded-lg p-3.5 border border-slate-800 text-xs space-y-2">
                        <h4 class="font-semibold text-slate-300 uppercase tracking-wider text-[10px] mb-1">Projected Plant Yield (Year 1)</h4>
                        
                        <div class="flex justify-between items-center py-0.5 border-b border-slate-800/60">
                            <span class="text-slate-400">Installed Capacity:</span>
                            <span id="estCapacity" class="font-mono font-semibold text-slate-100">44.4 MW</span>
                        </div>
                        <div class="flex justify-between items-center py-0.5 border-b border-slate-800/60">
                            <span class="text-slate-400">Annual Generation:</span>
                            <span id="estGen" class="font-mono font-semibold text-amber-400">78.9 GWh/year</span>
                        </div>
                        <div class="flex justify-between items-center py-0.5 border-b border-slate-800/60">
                            <span class="text-slate-400">Capacity Factor (CUF):</span>
                            <span id="estCuf" class="font-mono font-semibold text-sky-400">24.2 %</span>
                        </div>
                        <div class="flex justify-between items-center py-0.5 border-b border-slate-800/60">
                            <span class="text-slate-400">Est. Annual Revenue:</span>
                            <span id="estRev" class="font-mono font-semibold text-emerald-400">₹ 20.5 Cr / yr</span>
                        </div>
                        <div class="flex justify-between items-center py-0.5 border-b border-slate-800/60">
                            <span class="text-slate-400">Est. Project Capex:</span>
                            <span id="estCapex" class="font-mono font-semibold text-slate-300">₹ 177.6 Cr</span>
                        </div>
                        <div class="flex justify-between items-center py-0.5">
                            <span class="text-slate-400">Annual CO₂ Offset:</span>
                            <span id="estCo2" class="font-mono font-semibold text-emerald-300">64,700 Tonnes</span>
                        </div>
                    </div>
                </div>
            </div>

            <!-- Engineering footer note -->
            <div class="mt-4 pt-3 border-t border-slate-800 text-[10px] text-slate-500 flex items-start gap-2">
                <i data-lucide="info" class="w-3.5 h-3.5 text-slate-400 shrink-0 mt-0.5"></i>
                <span>Calculations assume standard ~4.5 acres/MW land requirement, 0.78 PR, single-axis tracking, and CEA grid emission factor of 0.82 kg CO₂/kWh.</span>
            </div>
        </div>
    </div>

    <!-- Data Injection & Map Engine -->
    <script>
        const SOLAR_DATA = """ + json.dumps(points_records) + """;
        const MEGA_PARKS = """ + json.dumps(parks_data) + """;
        const INDIA_GEOJSON = """ + geojson_str + """;

        let map, heatLayer, parksLayer, geojsonLayer;
        let currentGhi = 6.24;

        function initMap() {
            map = L.map('map', {
                center: [22.8, 79.2],
                zoom: 5,
                minZoom: 4,
                maxZoom: 14,
                zoomControl: false
            });

            L.control.zoom({ position: 'topleft' }).addTo(map);

            // Dark Matter Basemap
            L.tileLayer('https://{s}.basemaps.cartocdn.com/dark_all/{z}/{x}/{y}{r}.png', {
                attribution: '&copy; OpenStreetMap &copy; CARTO',
                subdomains: 'abcd',
                maxZoom: 19
            }).addTo(map);

            // Official Boundary GeoJSON
            geojsonLayer = L.geoJSON(INDIA_GEOJSON, {
                style: {
                    color: '#64748b',
                    weight: 1.2,
                    fillOpacity: 0.04,
                    fillColor: '#f59e0b'
                }
            }).addTo(map);

            // Heatmap Layer of Solar Insolation & Suitability
            const heatPoints = SOLAR_DATA.map(p => [p.lat, p.lon, (p.suit / 100.0) * 1.4]);
            heatLayer = L.heatLayer(heatPoints, {
                radius: 26,
                blur: 22,
                maxZoom: 10,
                max: 1.4,
                gradient: {
                    0.2: '#0b0f19',
                    0.4: '#3b82f6',
                    0.6: '#f59e0b',
                    0.8: '#ef4444',
                    1.0: '#ffffff'
                }
            }).addTo(map);

            // Mega Solar Parks Layer
            parksLayer = L.layerGroup().addTo(map);
            MEGA_PARKS.forEach(park => {
                const marker = L.circleMarker([park.lat, park.lon], {
                    radius: Math.max(6, Math.min(15, park.capacity / 180)),
                    fillColor: '#38bdf8',
                    color: '#ffffff',
                    weight: 1.5,
                    opacity: 1,
                    fillOpacity: 0.85
                });

                const popupContent = `
                    <div class="p-2 custom-popup text-xs space-y-1.5">
                        <div class="font-semibold text-amber-400 text-sm flex items-center justify-between">
                            <span>${park.name}</span>
                            <span class="text-[10px] font-mono bg-sky-500/20 text-sky-300 px-1.5 py-0.5 rounded border border-sky-500/30">${park.capacity.toLocaleString()} MW</span>
                        </div>
                        <div class="text-slate-300"><strong>State:</strong> ${park.state} | <strong>Status:</strong> ${park.status}</div>
                        <div class="text-slate-300"><strong>Developer:</strong> ${park.developer}</div>
                        <div class="text-slate-400 text-[11px]"><strong>Substation:</strong> ${park.substation || 'Grid Pooling Station'}</div>
                        <p class="text-slate-400 text-[11px] mt-1 italic border-t border-slate-800 pt-1">${park.desc}</p>
                    </div>
                `;
                marker.bindPopup(popupContent);
                marker.on('click', () => {
                    selectLocation(park.lat, park.lon, park.name + ', ' + park.state, 6.1, 95.0, 'Tier 1 — Prime');
                });
                parksLayer.addLayer(marker);
            });

            // Map Click Inspector
            map.on('click', (e) => {
                const lat = e.latlng.lat;
                const lon = e.latlng.lng;

                // Find nearest solar data point
                let nearest = SOLAR_DATA[0];
                let minDist = 999999;
                for (let i = 0; i < SOLAR_DATA.length; i++) {
                    const d = Math.hypot(SOLAR_DATA[i].lat - lat, SOLAR_DATA[i].lon - lon);
                    if (d < minDist) {
                        minDist = d;
                        nearest = SOLAR_DATA[i];
                    }
                }

                selectLocation(lat, lon, 'Custom Site (Lat ' + lat.toFixed(2) + ', Lon ' + lon.toFixed(2) + ')', nearest.ghi, nearest.suit, nearest.tier);
            });
        }

        function selectLocation(lat, lon, label, ghi, suit, tier) {
            currentGhi = ghi;
            document.getElementById('selectedCoords').innerText = `${lat.toFixed(4)}° N, ${lon.toFixed(4)}° E`;
            document.getElementById('selectedRegion').innerText = label;
            document.getElementById('valGhi').innerText = ghi.toFixed(2);
            document.getElementById('valScore').innerText = `${suit.toFixed(1)} / 100`;
            
            const tierEl = document.getElementById('valTier');
            tierEl.innerText = tier;
            if (tier.includes('Tier 1')) tierEl.className = 'text-[10px] text-emerald-400 block font-medium';
            else if (tier.includes('Tier 2')) tierEl.className = 'text-[10px] text-blue-400 block font-medium';
            else if (tier.includes('Tier 3')) tierEl.className = 'text-[10px] text-amber-400 block font-medium';
            else tierEl.className = 'text-[10px] text-red-400 block font-medium';

            const land = document.getElementById('landAreaSlider').value;
            calculateYield(land);
        }

        function calculateYield(landAcres) {
            document.getElementById('landAreaDisplay').innerText = `${landAcres} Acres`;

            const mw = (landAcres / 4.5);
            const kw = mw * 1000;
            const annualGhi = currentGhi * 365;
            const specificYield = annualGhi * 0.78; // kWh / kWp
            const genKwh = kw * specificYield;
            const genGwh = genKwh / 1000000;
            const cuf = (genKwh / (kw * 8760)) * 100;
            const revCr = (genKwh * 2.60) / 10000000;
            const capexCr = mw * 4.0; // ~₹4.0 Cr / MW utility benchmark
            const co2Tonnes = (genKwh * 0.82) / 1000;

            document.getElementById('estCapacity').innerText = `${mw.toFixed(1)} MW`;
            document.getElementById('estGen').innerText = `${genGwh.toFixed(1)} GWh/year`;
            document.getElementById('estCuf').innerText = `${cuf.toFixed(1)} %`;
            document.getElementById('estRev').innerText = `₹ ${revCr.toFixed(1)} Cr / yr`;
            document.getElementById('estCapex').innerText = `₹ ${capexCr.toFixed(1)} Cr`;
            document.getElementById('estCo2').innerText = `${Math.round(co2Tonnes).toLocaleString()} Tonnes`;
        }

        function updateHeatOpacity(val) {
            document.getElementById('opacityVal').innerText = `${val}%`;
            if (heatLayer) {
                heatLayer.setOptions({ opacity: val / 100.0 });
            }
        }

        function toggleSidePanel() {
            const panel = document.getElementById('sidePanel');
            panel.classList.toggle('translate-x-full');
            panel.classList.toggle('translate-x-0');
        }

        function downloadCSV() {
            let csv = "latitude,longitude,ghi_daily_kwh_m2,suitability_score,suitability_tier,temp_c\\n";
            SOLAR_DATA.forEach(p => {
                csv += `${p.lat},${p.lon},${p.ghi},${p.suit},"${p.tier}",${p.temp}\\n`;
            });
            const blob = new Blob([csv], { type: 'text/csv' });
            const url = window.URL.createObjectURL(blob);
            const a = document.createElement('a');
            a.setAttribute('href', url);
            a.setAttribute('download', 'india_solar_plant_suitability_raydium.csv');
            a.click();
        }

        window.onload = () => {
            initMap();
            lucide.createIcons();
            calculateYield(200);
        };
    </script>
</body>
</html>"""

        with open(output_html, "w", encoding="utf-8") as f:
            f.write(html_template)

        logger.info(f"Dashboard rendered successfully to {output_html} (Size: {len(html_template)/1024:.1f} KB)")
        return output_html
