"""Publication cartography and modern interactive web dashboard generator."""

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
    """Generates high-resolution publication PNGs and interactive HTML5 dashboards."""

    def __init__(self, geojson_path: str = "india-soi.geojson"):
        self.geojson_path = geojson_path
        if not os.path.exists(geojson_path):
            raise FileNotFoundError(f"GeoJSON not found at: {geojson_path}")
        self.india_gdf = gpd.read_file(geojson_path)
        if self.india_gdf.crs != "EPSG:4326":
            self.india_gdf = self.india_gdf.to_crs("EPSG:4326")

    def render_static_map(
        self,
        raster_dict: Dict,
        output_image: str = "solar_potential_high_res.png",
        title: str = "India - Solar Plant Suitability & Potential Map",
        dpi: int = 300,
        show_parks: bool = True,
    ) -> str:
        """Render high-resolution cartographic map with boundary overlay and solar parks."""
        logger.info(f"Rendering publication static map to {output_image} (DPI={dpi})...")

        masked_raster = raster_dict["raster"]
        bounds = raster_dict["bounds"]  # [minx, miny, maxx, maxy]
        vmin = raster_dict.get("vmin", float(np.nanmin(masked_raster)))
        vmax = raster_dict.get("vmax", float(np.nanmax(masked_raster)))

        fig, ax = plt.subplots(figsize=(14, 16), dpi=dpi, facecolor="#0f172a")
        ax.set_facecolor("#0f172a")

        # Custom high-contrast Solar colormap (Deep Navy -> Amber -> Yellow -> White)
        colors = [
            (0.08, 0.05, 0.22),
            (0.24, 0.15, 0.53),
            (0.55, 0.18, 0.55),
            (0.85, 0.32, 0.38),
            (0.98, 0.65, 0.15),
            (0.99, 0.92, 0.40),
            (1.00, 1.00, 0.90),
        ]
        solar_cmap = LinearSegmentedColormap.from_list("SolarThermal", colors, N=256)

        # Plot raster surface
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

        # Plot official Survey of India boundary
        self.india_gdf.boundary.plot(ax=ax, color="#e2e8f0", linewidth=1.2, alpha=0.9)

        # Overlay Mega Solar Parks
        if show_parks:
            park_lons = [p.longitude for p in MEGA_SOLAR_PARKS]
            park_lats = [p.latitude for p in MEGA_SOLAR_PARKS]
            park_sizes = [min(300, max(60, p.capacity_mw / 20.0)) for p in MEGA_SOLAR_PARKS]

            ax.scatter(
                park_lons,
                park_lats,
                s=park_sizes,
                c="#38bdf8",
                edgecolors="#ffffff",
                linewidths=1.5,
                zorder=10,
                alpha=0.95,
                label="Major Mega Solar Parks",
            )

            # Annotate top 5 solar parks
            top_parks = sorted(MEGA_SOLAR_PARKS, key=lambda x: x.capacity_mw, reverse=True)[:5]
            for p in top_parks:
                ax.annotate(
                    f"{p.name}\n({p.capacity_mw:,.0f} MW)",
                    xy=(p.longitude, p.latitude),
                    xytext=(10, 10),
                    textcoords="offset points",
                    color="#ffffff",
                    fontsize=8,
                    fontweight="bold",
                    bbox=dict(boxstyle="round,pad=0.25", fc="#1e293b", ec="#38bdf8", alpha=0.85, lw=1),
                    arrowprops=dict(arrowstyle="->", color="#38bdf8", lw=1.2),
                    zorder=15,
                )

        # Title & Metadata
        ax.set_title(title, fontsize=18, fontweight="bold", color="#ffffff", pad=16)
        ax.set_xlim(bounds[0] - 0.5, bounds[2] + 0.5)
        ax.set_ylim(bounds[1] - 0.5, bounds[3] + 0.5)
        ax.set_axis_off()

        # Colorbar
        cbar = plt.colorbar(
            img,
            ax=ax,
            orientation="horizontal",
            fraction=0.035,
            pad=0.03,
            aspect=35,
        )
        cbar.set_label("Solar Plant Suitability Index (0 - 100) / Solar Potential", color="#e2e8f0", fontsize=11, fontweight="medium")
        cbar.ax.tick_params(labelsize=9, colors="#cbd5e1")

        # Inset Key Stats box
        stats_text = (
            "RAYDIUM SOLAR PLATFORM\n"
            f"Mean Irradiance: ~5.18 kWh/m²/day\n"
            f"Prime Siting: Thar Desert & Kutch\n"
            f"Tracked Mega Parks: {len(MEGA_SOLAR_PARKS)} Sites\n"
            "Data: NASA POWER & Survey of India"
        )
        ax.text(
            0.02, 0.04,
            stats_text,
            transform=ax.transAxes,
            fontsize=8,
            color="#94a3b8",
            bbox=dict(boxstyle="square,pad=0.5", fc="#0f172a", ec="#334155", alpha=0.9),
            verticalalignment="bottom",
        )

        os.makedirs(os.path.dirname(os.path.abspath(output_image)) if os.path.dirname(output_image) else ".", exist_ok=True)
        plt.savefig(output_image, bbox_inches="tight", pad_inches=0.2, dpi=dpi, facecolor="#0f172a")
        plt.close(fig)

        logger.info(f"Static map saved successfully to {output_image}")
        return output_image

    def render_interactive_dashboard(
        self,
        df: pd.DataFrame,
        output_html: str = "index.html",
        summary_stats: Optional[Dict] = None,
    ) -> str:
        """Generate a state-of-the-art interactive HTML5/Leaflet solar dashboard."""
        logger.info(f"Generating interactive web dashboard at {output_html}...")

        # Sample data down for instant client-side performance if large
        if len(df) > 3000:
            sample_df = df.sample(n=3000, random_state=42)
        else:
            sample_df = df

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
                "year": p.commissioned_year,
                "desc": p.description,
            }
            for p in MEGA_SOLAR_PARKS
        ]

        # Generate lightweight simplified GeoJSON for fast client-side map rendering
        simplified_gdf = self.india_gdf.copy()
        simplified_gdf.geometry = self.india_gdf.geometry.simplify(0.012, preserve_topology=True)
        geojson_str = simplified_gdf.to_json()

        html_template = """<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Raydium - Ideal Solar Plant Locations & Potential Map of India</title>
    
    <!-- Tailwind CSS CDN -->
    <script src="https://cdn.tailwindcss.com"></script>
    <script>
        tailwind.config = {
            darkMode: 'class',
            theme: {
                extend: {
                    colors: {
                        brand: { 50: '#fffbeb', 500: '#f59e0b', 600: '#d97706', 700: '#b45309' },
                        dark: { 800: '#1e293b', 900: '#0f172a', 950: '#020617' }
                    }
                }
            }
        }
    </script>
    
    <!-- Leaflet CSS & JS -->
    <link rel="stylesheet" href="https://unpkg.com/leaflet@1.9.4/dist/leaflet.css" />
    <script src="https://unpkg.com/leaflet@1.9.4/dist/leaflet.js"></script>
    <!-- Leaflet Heatmap Plugin -->
    <script src="https://unpkg.com/leaflet.heat@0.2.0/dist/leaflet-heat.js"></script>
    <!-- Chart.js -->
    <script src="https://cdn.jsdelivr.net/npm/chart.js"></script>
    <!-- Lucide Icons -->
    <script src="https://unpkg.com/lucide@latest"></script>

    <style>
        #map { height: calc(100vh - 72px); width: 100%; z-index: 10; }
        .custom-popup .leaflet-popup-content-wrapper {
            background: #0f172a;
            color: #f8fafc;
            border: 1px solid #334155;
            border-radius: 12px;
            box-shadow: 0 20px 25px -5px rgb(0 0 0 / 0.5);
        }
        .custom-popup .leaflet-popup-tip { background: #0f172a; }
        .glass-panel {
            background: rgba(15, 23, 42, 0.85);
            backdrop-filter: blur(12px);
            border: 1px solid rgba(51, 65, 85, 0.6);
        }
    </style>
</head>
<body class="bg-dark-950 text-slate-100 font-sans antialiased overflow-hidden flex flex-col h-screen">

    <!-- Top Navigation Header -->
    <header class="h-18 bg-dark-900/90 border-b border-slate-800 px-6 py-3 flex items-center justify-between z-30">
        <div class="flex items-center space-x-3">
            <div class="w-10 h-10 rounded-xl bg-gradient-to-tr from-amber-500 to-orange-500 flex items-center justify-center shadow-lg shadow-amber-500/20">
                <i data-lucide="sun" class="w-6 h-6 text-white animate-pulse"></i>
            </div>
            <div>
                <h1 class="text-xl font-bold tracking-tight text-white flex items-center gap-2">
                    RAYDIUM <span class="text-xs bg-amber-500/20 text-amber-400 px-2 py-0.5 rounded-full border border-amber-500/30">v2.0</span>
                </h1>
                <p class="text-xs text-slate-400">Ideal Solar Plant Siting & Potential Intelligence for India</p>
            </div>
        </div>

        <!-- Quick Stats Banner -->
        <div class="hidden md:flex items-center space-x-6 text-sm">
            <div class="text-right">
                <span class="text-xs text-slate-400 block">National Avg Irradiance</span>
                <span class="text-amber-400 font-semibold">5.18 kWh/m²/day</span>
            </div>
            <div class="h-8 w-px bg-slate-800"></div>
            <div class="text-right">
                <span class="text-xs text-slate-400 block">Prime Siting Hotspot</span>
                <span class="text-emerald-400 font-semibold">Thar Desert & Kutch (>6.2 kWh)</span>
            </div>
            <div class="h-8 w-px bg-slate-800"></div>
            <div class="text-right">
                <span class="text-xs text-slate-400 block">Mega Parks Tracked</span>
                <span class="text-sky-400 font-semibold">12 Parks (45+ GW)</span>
            </div>
        </div>

        <!-- Actions -->
        <div class="flex items-center space-x-3">
            <button onclick="toggleSidePanel()" class="flex items-center gap-1.5 px-3.5 py-1.5 rounded-lg bg-slate-800 hover:bg-slate-700 text-xs font-medium text-slate-200 border border-slate-700 transition">
                <i data-lucide="sliders" class="w-4 h-4 text-amber-400"></i> Siting Calculator
            </button>
            <button onclick="downloadCSV()" class="flex items-center gap-1.5 px-3.5 py-1.5 rounded-lg bg-amber-600 hover:bg-amber-500 text-xs font-medium text-white shadow-lg shadow-amber-600/20 transition">
                <i data-lucide="download" class="w-4 h-4"></i> Export Data
            </button>
        </div>
    </header>

    <!-- Main Workspace: Map + Floating Panels -->
    <div class="relative flex-1 w-full overflow-hidden">
        <!-- The Interactive Map -->
        <div id="map"></div>

        <!-- Floating Legend Panel (Bottom Left) -->
        <div class="absolute bottom-6 left-6 z-20 glass-panel rounded-xl p-4 shadow-2xl max-w-xs w-full text-xs">
            <h4 class="font-bold text-slate-200 uppercase tracking-wider mb-2.5 flex items-center justify-between">
                <span>Solar Suitability Index</span>
                <i data-lucide="info" class="w-3.5 h-3.5 text-slate-400"></i>
            </h4>
            <div class="space-y-2">
                <div class="flex items-center justify-between">
                    <span class="flex items-center gap-2 text-slate-300">
                        <span class="w-3 h-3 rounded-full bg-emerald-500 shadow-sm shadow-emerald-500/50"></span>
                        Tier 1 - Prime (>85)
                    </span>
                    <span class="text-emerald-400 font-mono font-medium">Mega Utility</span>
                </div>
                <div class="flex items-center justify-between">
                    <span class="flex items-center gap-2 text-slate-300">
                        <span class="w-3 h-3 rounded-full bg-blue-500 shadow-sm shadow-blue-500/50"></span>
                        Tier 2 - High (70-85)
                    </span>
                    <span class="text-blue-400 font-mono font-medium">Utility / Commercial</span>
                </div>
                <div class="flex items-center justify-between">
                    <span class="flex items-center gap-2 text-slate-300">
                        <span class="w-3 h-3 rounded-full bg-amber-500 shadow-sm shadow-amber-500/50"></span>
                        Tier 3 - Moderate (50-70)
                    </span>
                    <span class="text-amber-400 font-mono font-medium">Rooftop / Agri</span>
                </div>
                <div class="flex items-center justify-between">
                    <span class="flex items-center gap-2 text-slate-300">
                        <span class="w-3 h-3 rounded-full bg-red-500 shadow-sm shadow-red-500/50"></span>
                        Tier 4 - Constrained (<50)
                    </span>
                    <span class="text-red-400 font-mono font-medium">Restricted / Steep</span>
                </div>
            </div>

            <!-- Heat Layer Opacity Slider -->
            <div class="mt-4 pt-3 border-t border-slate-800">
                <label class="flex justify-between text-slate-400 mb-1">
                    <span>Heatmap Opacity</span>
                    <span id="opacityVal" class="text-slate-200">75%</span>
                </label>
                <input id="heatOpacity" type="range" min="10" max="100" value="75" class="w-full accent-amber-500 cursor-pointer" oninput="updateHeatOpacity(this.value)">
            </div>
        </div>

        <!-- Floating Siting & Yield Inspector Panel (Right Drawer) -->
        <div id="sidePanel" class="absolute top-4 right-4 bottom-6 z-20 glass-panel rounded-2xl p-5 shadow-2xl w-96 overflow-y-auto flex flex-col justify-between transition-all duration-300 transform translate-x-0">
            <div>
                <div class="flex items-center justify-between pb-3 border-b border-slate-800 mb-4">
                    <h3 class="font-bold text-slate-100 flex items-center gap-2 text-sm">
                        <i data-lucide="calculator" class="w-4 h-4 text-amber-400"></i> Plant Siting Inspector
                    </h3>
                    <button onclick="toggleSidePanel()" class="text-slate-400 hover:text-white">
                        <i data-lucide="x" class="w-4 h-4"></i>
                    </button>
                </div>

                <!-- Selected Location Details -->
                <div id="inspectorContent" class="space-y-4">
                    <div class="bg-dark-900/80 rounded-xl p-3.5 border border-slate-800 text-xs">
                        <div class="text-slate-400 mb-1 font-medium">Selected Location Coordinates</div>
                        <div id="selectedCoords" class="text-base font-bold font-mono text-amber-400">27.5386° N, 71.9167° E</div>
                        <div id="selectedRegion" class="text-xs text-slate-300 mt-1">Bhadla Region, Rajasthan</div>
                    </div>

                    <!-- Insolation KPI Grid -->
                    <div class="grid grid-cols-2 gap-2 text-xs">
                        <div class="bg-dark-900/60 p-3 rounded-xl border border-slate-800">
                            <span class="text-slate-400 block">Daily GHI</span>
                            <span id="valGhi" class="text-lg font-bold text-amber-400 font-mono">6.24</span>
                            <span class="text-[10px] text-slate-500 block">kWh/m²/day</span>
                        </div>
                        <div class="bg-dark-900/60 p-3 rounded-xl border border-slate-800">
                            <span class="text-slate-400 block">Suitability Score</span>
                            <span id="valScore" class="text-lg font-bold text-emerald-400 font-mono">96.5 / 100</span>
                            <span id="valTier" class="text-[10px] text-emerald-400 block font-medium">Tier 1 - Prime</span>
                        </div>
                    </div>

                    <!-- Interactive Sizing Calculator -->
                    <div class="pt-2">
                        <div class="flex justify-between items-center mb-2">
                            <label class="text-xs font-semibold text-slate-200">Proposed Land Area</label>
                            <span id="landAreaDisplay" class="text-xs font-mono font-bold text-amber-400">200 Acres</span>
                        </div>
                        <input id="landAreaSlider" type="range" min="10" max="2000" step="10" value="200" class="w-full accent-amber-500 cursor-pointer" oninput="calculateYield(this.value)">
                    </div>

                    <!-- Yield Projections -->
                    <div class="bg-dark-900/90 rounded-xl p-3.5 border border-slate-800 text-xs space-y-2.5">
                        <h4 class="font-bold text-slate-300 uppercase tracking-wider text-[11px] mb-1">Projected Plant Yield</h4>
                        <div class="flex justify-between">
                            <span class="text-slate-400">Plant Capacity:</span>
                            <span id="estCapacity" class="font-mono font-bold text-slate-100">44.4 MW</span>
                        </div>
                        <div class="flex justify-between">
                            <span class="text-slate-400">Annual Generation:</span>
                            <span id="estGen" class="font-mono font-bold text-amber-400">78.9 GWh/year</span>
                        </div>
                        <div class="flex justify-between">
                            <span class="text-slate-400">Capacity Factor (CUF):</span>
                            <span id="estCuf" class="font-mono font-bold text-sky-400">24.2 %</span>
                        </div>
                        <div class="flex justify-between">
                            <span class="text-slate-400">Est. Annual Revenue:</span>
                            <span id="estRev" class="font-mono font-bold text-emerald-400">₹ 20.5 Cr / yr</span>
                        </div>
                        <div class="flex justify-between">
                            <span class="text-slate-400">Annual CO₂ Offset:</span>
                            <span id="estCo2" class="font-mono font-bold text-emerald-300">64,700 Tonnes</span>
                        </div>
                    </div>
                </div>
            </div>

            <!-- Hint footer -->
            <div class="mt-4 pt-3 border-t border-slate-800 text-[11px] text-slate-400 flex items-center gap-2">
                <i data-lucide="mouse-pointer-click" class="w-4 h-4 text-amber-400 shrink-0"></i>
                <span>Click anywhere on India to inspect local solar insolation and compute plant sizing.</span>
            </div>
        </div>
    </div>

    <!-- Data Injection -->
    <script>
        const SOLAR_DATA = """ + json.dumps(points_records) + """;
        const MEGA_PARKS = """ + json.dumps(parks_data) + """;
        const INDIA_GEOJSON = """ + geojson_str + """;

        let map, heatLayer, parksLayer, geojsonLayer;
        let currentGhi = 6.24;

        function initMap() {
            // Initialize map centered on India
            map = L.map('map', {
                center: [22.5, 79.5],
                zoom: 5,
                minZoom: 4,
                maxZoom: 14,
                zoomControl: false
            });

            L.control.zoom({ position: 'topleft' }).addTo(map);

            // Dark CartoDB Basemap
            const darkTiles = L.tileLayer('https://{s}.basemaps.cartocdn.com/dark_all/{z}/{x}/{y}{r}.png', {
                attribution: '&copy; OpenStreetMap &copy; CARTO',
                subdomains: 'abcd',
                maxZoom: 19
            }).addTo(map);

            // Satellite Tiles Option
            const satelliteTiles = L.tileLayer('https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}', {
                attribution: '&copy; Esri'
            });

            // Boundary GeoJSON
            geojsonLayer = L.geoJSON(INDIA_GEOJSON, {
                style: {
                    color: '#94a3b8',
                    weight: 1.5,
                    fillOpacity: 0.05,
                    fillColor: '#f59e0b'
                }
            }).addTo(map);

            // Heatmap Layer of Solar Insolation & Suitability
            const heatPoints = SOLAR_DATA.map(p => [p.lat, p.lon, (p.suit / 100.0) * 1.5]);
            heatLayer = L.heatLayer(heatPoints, {
                radius: 28,
                blur: 24,
                maxZoom: 10,
                max: 1.5,
                gradient: {
                    0.2: '#0f172a',
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
                    radius: Math.max(6, Math.min(16, park.capacity / 150)),
                    fillColor: '#38bdf8',
                    color: '#ffffff',
                    weight: 2,
                    opacity: 1,
                    fillOpacity: 0.9
                });

                const popupContent = `
                    <div class="p-2 custom-popup text-xs space-y-1.5">
                        <div class="font-bold text-amber-400 text-sm flex items-center justify-between">
                            <span>${park.name}</span>
                            <span class="text-[10px] bg-sky-500/20 text-sky-300 px-1.5 py-0.5 rounded">${park.capacity} MW</span>
                        </div>
                        <div class="text-slate-300"><strong>State:</strong> ${park.state}</div>
                        <div class="text-slate-300"><strong>Status:</strong> ${park.status} (${park.year || 'Operational'})</div>
                        <div class="text-slate-300"><strong>Developer:</strong> ${park.developer}</div>
                        <p class="text-slate-400 text-[11px] mt-1 italic">${park.desc}</p>
                    </div>
                `;
                marker.bindPopup(popupContent);
                marker.on('click', () => {
                    selectLocation(park.lat, park.lon, park.name + ', ' + park.state, 6.1, 95.0, 'Tier 1 - Prime');
                });
                parksLayer.addLayer(marker);
            });

            // Map Click Inspector
            map.on('click', (e) => {
                const lat = e.latlng.lat;
                const lon = e.latlng.lng;

                // Find nearest solar point
                let nearest = SOLAR_DATA[0];
                let minDist = 999999;
                for (let i = 0; i < SOLAR_DATA.length; i++) {
                    const d = Math.hypot(SOLAR_DATA[i].lat - lat, SOLAR_DATA[i].lon - lon);
                    if (d < minDist) {
                        minDist = d;
                        nearest = SOLAR_DATA[i];
                    }
                }

                selectLocation(lat, lon, 'India Custom Site', nearest.ghi, nearest.suit, nearest.tier);
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
            const co2Tonnes = (genKwh * 0.82) / 1000;

            document.getElementById('estCapacity').innerText = `${mw.toFixed(1)} MW`;
            document.getElementById('estGen').innerText = `${genGwh.toFixed(1)} GWh/year`;
            document.getElementById('estCuf').innerText = `${cuf.toFixed(1)} %`;
            document.getElementById('estRev').innerText = `₹ ${revCr.toFixed(1)} Cr / yr`;
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
