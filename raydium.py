import numpy as np
import pandas as pd
import geopandas as gpd
import folium
from folium import plugins
from scipy.interpolate import griddata
import branca.colormap as cm

def create_improved_solar_map(geojson_path, solar_data_path):
    """
    Create an improved solar potential map with smoother interpolation and better color scheme
    """
    # Read India GeoJSON and solar data
    india = gpd.read_file(geojson_path)
    solar_df = pd.read_csv(solar_data_path)
    
    # Create base map
    m = folium.Map(
        location=[20.5937, 78.9629],
        zoom_start=5,
        tiles='cartodbpositron'  # Using a cleaner base map
    )
    
    # Create a finer grid for smoother interpolation
    grid_size = 200  # Increased from 100 for smoother interpolation
    grid_lat = np.linspace(solar_df['latitude'].min(), solar_df['latitude'].max(), grid_size)
    grid_lon = np.linspace(solar_df['longitude'].min(), solar_df['longitude'].max(), grid_size)
    grid_lon, grid_lat = np.meshgrid(grid_lon, grid_lat)
    
    # Interpolate with cubic method for smoother transitions
    points = solar_df[['longitude', 'latitude']].values
    values = solar_df['potential'].values
    grid_z = griddata(points, values, (grid_lon, grid_lat), method='cubic')
    
    # Create an improved color scheme with better differentiation
    colors = [
        '#313695',  # Deep blue
        '#4575b4',  # Medium blue
        '#74add1',  # Light blue
        '#abd9e9',  # Very light blue
        '#fee090',  # Light yellow
        '#fdae61',  # Orange
        '#f46d43',  # Dark orange
        '#d73027',  # Red
        '#a50026'   # Deep red
    ]
    
    # Add India boundary first (below the heatmap)
    folium.GeoJson(
        india,
        style_function=lambda x: {
            'fillColor': 'none',
            'color': '#2b2b2b',
            'weight': 1,
            'fillOpacity': 0
        }
    ).add_to(m)
    
    # Create improved heatmap
    heat_data = []
    for i in range(len(grid_lat)):
        for j in range(len(grid_lon)):
            if not np.isnan(grid_z[i, j]):
                # Check if point is within India's bounds
                point = gpd.points_from_xy([grid_lon[i, j]], [grid_lat[i, j]])[0]
                if any(india.geometry.contains(point)):
                    heat_data.append([
                        float(grid_lat[i, j]),
                        float(grid_lon[i, j]),
                        float(grid_z[i, j])
                    ])
    
    # Configure heatmap with improved settings
    plugins.HeatMap(
        heat_data,
        min_opacity=0.7,        # Increased for better visibility
        max_zoom=6,
        radius=15,              # Reduced for smoother appearance
        blur=20,                # Increased for smoother transitions
        gradient={              # More gradient steps for smoother color transition
            0.0: colors[0],
            0.2: colors[1],
            0.3: colors[2],
            0.4: colors[3],
            0.5: colors[4],
            0.6: colors[5],
            0.7: colors[6],
            0.8: colors[7],
            1.0: colors[8]
        }
    ).add_to(m)
    
    # Add improved colormap
    colormap = cm.LinearColormap(
        colors=colors,
        vmin=min(solar_df['potential']),
        vmax=max(solar_df['potential']),
        caption='Solar Potential (kWh/m²/year)'
    )
    colormap.add_to(m)
    
    return m

def main():
    geojson_path = 'india-soi.geojson'
    solar_data_path = 'india_solar_data.csv'
    
    try:
        solar_map = create_improved_solar_map(geojson_path, solar_data_path)
        solar_map.save('improved_india_solar_potential.html')
        print("Improved map has been generated and saved as 'improved_india_solar_potential.html'")
        
    except Exception as e:
        print(f"An error occurred: {e}")

if __name__ == "__main__":
    main()
