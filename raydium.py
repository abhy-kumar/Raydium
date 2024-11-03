import numpy as np
import pandas as pd
import geopandas as gpd
import folium
from folium import plugins
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
        data = response.json()
        return data
    except requests.exceptions.RequestException as e:
        print(f"Error making request: {e}")
        return None
    except ValueError as e:
        print(f"Error parsing JSON: {e}")
        return None

def calculate_solar_potential(power_data, panel_efficiency=0.2):
    """
    Calculate solar potential based on radiation data
    """
    if power_data is None:
        return None
    
    try:
        daily_data = power_data.get('properties', {}).get('parameter', {}).get('ALLSKY_SFC_SW_DWN', {})
        
        if not daily_data:
            print("No daily data found in response")
            return None
        
        radiation_values = []
        for value in daily_data.values():
            if isinstance(value, (int, float)) and value != -999:
                radiation_values.append(value * 0.277778)  # Convert MJ to kWh
        
        if not radiation_values:
            print("No valid radiation values found")
            return None
        
        annual_potential = np.mean(radiation_values) * 365 * panel_efficiency
        return annual_potential
        
    except (KeyError, TypeError) as e:
        print(f"Error processing data: {e}")
        return None

def create_india_solar_map(geojson_path):
    """
    Create an interactive map of India's solar potential with optimized rendering
    """
    # Read India GeoJSON
    india = gpd.read_file(geojson_path)
    
    # Create a base map centered on India
    m = folium.Map(location=[20.5937, 78.9629], zoom_start=5)
    
    # Create a coarser grid for sampling points (1-degree resolution)
    lat_range = np.arange(8.4, 37.6, 1.0)  # Increased step size
    lon_range = np.arange(68.7, 97.25, 1.0)  # Increased step size
    
    # Store solar potential data
    solar_data = []
    
    # Fetch data for each grid point
    total_points = len(lat_range) * len(lon_range)
    current_point = 0
    
    for lat in lat_range:
        for lon in lon_range:
            current_point += 1
            print(f"\nProcessing point {current_point}/{total_points}")
            
            point = gpd.points_from_xy([lon], [lat])[0]
            
            # Check if point is within India
            if any(india.geometry.contains(point)):
                power_data = fetch_nasa_power_data(lat, lon)
                time.sleep(2)  # Rate limiting
                
                if power_data:
                    potential = calculate_solar_potential(power_data)
                    if potential is not None:
                        solar_data.append({
                            'latitude': lat,
                            'longitude': lon,
                            'potential': potential
                        })
    
    # Create DataFrame from collected data
    solar_df = pd.DataFrame(solar_data)
    
    if solar_df.empty:
        raise ValueError("No solar data could be collected. Please check API access and coordinates.")
    
    # Create a moderately sized grid for interpolation (reduced from 500x500)
    grid_lat = np.linspace(solar_df['latitude'].min(), solar_df['latitude'].max(), 100)  # Reduced points
    grid_lon = np.linspace(solar_df['longitude'].min(), solar_df['longitude'].max(), 100)  # Reduced points
    grid_lon, grid_lat = np.meshgrid(grid_lon, grid_lat)
    
    # Interpolate the data
    points = solar_df[['longitude', 'latitude']].values
    values = solar_df['potential'].values
    grid_z = griddata(points, values, (grid_lon, grid_lat), method='cubic')
    
    # Create custom colormap
    min_val = np.nanmin(grid_z)
    max_val = np.nanmax(grid_z)
    step = (max_val - min_val) / 4  # 5 colors = 4 steps
    
    colors = ['#ffffb2', '#fecc5c', '#fd8d3c', '#f03b20', '#bd0026']
    colormap = cm.LinearColormap(
        colors=colors,
        vmin=min_val,
        vmax=max_val
    )
    
    # Create a heatmap layer instead of ImageOverlay
    heat_data = []
    for i in range(len(grid_lat)):
        for j in range(len(grid_lon)):
            if not np.isnan(grid_z[i, j]):
                heat_data.append([
                    float(grid_lat[i, j]),
                    float(grid_lon[i, j]),
                    float(grid_z[i, j])
                ])
    
    # Add heatmap layer
    plugins.HeatMap(
        heat_data,
        min_opacity=0.5,
        max_zoom=4,
        radius=25,
        blur=15,
        max_val=max_val,
        gradient={0.4: '#ffffb2', 0.6: '#fecc5c', 0.7: '#fd8d3c', 0.8: '#f03b20', 1: '#bd0026'},
        name='Solar Potential'
    ).add_to(m)
    
    # Add India boundary
    folium.GeoJson(
        india,
        style_function=lambda x: {
            'fillColor': 'none',
            'color': 'black',
            'weight': 1.5
        },
        name='India Boundary'
    ).add_to(m)
    
    # Add colormap to the map
    colormap.add_to(m)
    colormap.caption = 'Solar Potential (kWh/m²/year)'
    
    # Add layer control
    folium.LayerControl().add_to(m)
    
    # Save the solar potential data
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