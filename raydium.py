import numpy as np
import pandas as pd
import geopandas as gpd
import folium
import requests
from scipy.interpolate import griddata
import time
import branca.colormap as cm
import json

def fetch_nasa_power_data(lat, lon):
    """
    Fetch solar radiation data from NASA POWER API
    """
    base_url = "https://power.larc.nasa.gov/api/temporal/daily/point"
    params = {
        'parameters': 'ALLSKY_SFC_SW_DWN',
        'community': 'RE',
        'longitude': str(lon),
        'latitude': str(lat),
        'start': '20230101',
        'end': '20231231',
        'format': 'JSON'
    }
    
    try:
        response = requests.get(base_url, params=params)
        response.raise_for_status()
        return response.json()
    except Exception as e:
        print(f"Error: {e}")
        return None

def calculate_solar_potential(power_data, panel_efficiency=0.2):
    """
    Calculate solar potential from radiation data
    """
    if not power_data:
        return None
    
    try:
        daily_data = power_data['properties']['parameter']['ALLSKY_SFC_SW_DWN']
        radiation_values = [v * 0.277778 for v in daily_data.values() 
                          if isinstance(v, (int, float)) and v != -999]
        
        if not radiation_values:
            return None
            
        return np.mean(radiation_values) * 365 * panel_efficiency
    except:
        return None

def create_india_solar_map(geojson_path):
    """
    Create an interactive map of India's solar potential with improved visualization
    """
    # Read India GeoJSON
    india = gpd.read_file(geojson_path)
    
    # Create base map
    m = folium.Map(location=[20.5937, 78.9629], zoom_start=5, 
                  tiles='CartoDB positron')
    
    # Create a coarser grid for initial sampling (2-degree resolution)
    lat_range = np.arange(8.4, 37.6, 2.0)
    lon_range = np.arange(68.7, 97.25, 2.0)
    
    solar_data = []
    total_points = len(lat_range) * len(lon_range)
    
    # Fetch data for grid points
    for i, lat in enumerate(lat_range):
        for lon in lon_range:
            point = gpd.points_from_xy([lon], [lat])[0]
            
            if any(india.geometry.contains(point)):
                power_data = fetch_nasa_power_data(lat, lon)
                potential = calculate_solar_potential(power_data)
                
                if potential is not None:
                    solar_data.append({
                        'latitude': lat,
                        'longitude': lon,
                        'potential': potential
                    })
                
                time.sleep(1)  # Rate limiting
    
    solar_df = pd.DataFrame(solar_data)
    
    if solar_df.empty:
        raise ValueError("No solar data collected")
    
    # Create a finer grid for smooth interpolation
    grid_size = 80  # Reduced from previous version
    grid_lat = np.linspace(solar_df['latitude'].min(), solar_df['latitude'].max(), grid_size)
    grid_lon = np.linspace(solar_df['longitude'].min(), solar_df['longitude'].max(), grid_size)
    grid_lon, grid_lat = np.meshgrid(grid_lon, grid_lat)
    
    # Interpolate using cubic method for smoothness
    points = solar_df[['longitude', 'latitude']].values
    values = solar_df['potential'].values
    grid_z = griddata(points, values, (grid_lon, grid_lat), method='cubic')
    
    # Create a more distinguishable colormap
    colors = [
        '#313695',  # Deep blue
        '#4575b4',  # Blue
        '#74add1',  # Light blue
        '#abd9e9',  # Very light blue
        '#fee090',  # Light yellow
        '#fdae61',  # Orange
        '#f46d43',  # Dark orange
        '#d73027',  # Red
        '#a50026'   # Deep red
    ]
    
    # Create the interpolated raster layer
    img_array = np.zeros((grid_size, grid_size, 4), dtype=np.uint8)
    norm_z = (grid_z - np.nanmin(grid_z)) / (np.nanmax(grid_z) - np.nanmin(grid_z))
    
    # Convert interpolated values to RGBA
    for i in range(grid_size):
        for j in range(grid_size):
            if not np.isnan(norm_z[i, j]):
                # Get color index
                idx = int(norm_z[i, j] * (len(colors) - 1))
                color = colors[idx]
                # Convert hex to RGB
                r = int(color[1:3], 16)
                g = int(color[3:5], 16)
                b = int(color[5:7], 16)
                img_array[i, j] = [r, g, b, 180]  # Semi-transparent
    
    # Add the interpolated layer
    image_bounds = [[solar_df['latitude'].min(), solar_df['longitude'].min()],
                   [solar_df['latitude'].max(), solar_df['longitude'].max()]]
    
    folium.raster_layers.ImageOverlay(
        img_array,
        bounds=image_bounds,
        opacity=0.8,
        name='Solar Potential'
    ).add_to(m)
    
    # Add India boundary
    folium.GeoJson(
        india,
        style_function=lambda x: {
            'fillColor': 'none',
            'color': 'black',
            'weight': 1.5,
            'fillOpacity': 0
        },
        name='India Boundary'
    ).add_to(m)
    
    # Add colormap legend
    colormap = cm.LinearColormap(
        colors=colors,
        vmin=np.nanmin(grid_z),
        vmax=np.nanmax(grid_z)
    )
    colormap.add_to(m)
    colormap.caption = 'Solar Potential (kWh/m²/year)'
    
    # Add layer control
    folium.LayerControl().add_to(m)
    
    # Save data
    solar_df.to_csv('india_solar_data.csv', index=False)
    
    return m, solar_df

def main():
    """
    Main function to run the application
    """
    geojson_path = 'india-soi.geojson'
    
    try:
        print("Creating India's solar potential map...")
        solar_map, solar_data = create_india_solar_map(geojson_path)
        solar_map.save('india_solar_potential.html')
        print("\nMap has been generated and saved as 'india_solar_potential.html'")
        print("Raw data has been saved as 'india_solar_data.csv'")
        
    except Exception as e:
        print(f"An error occurred: {e}")

if __name__ == "__main__":
    main()