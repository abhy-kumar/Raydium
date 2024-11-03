import numpy as np
import pandas as pd
import geopandas as gpd
import folium
import requests
from scipy.interpolate import griddata
import time
import branca.colormap as cm

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
    Create an interactive map of India's solar potential with optimized visualization and performance
    """
    # Read India GeoJSON and simplify the geometry
    india = gpd.read_file(geojson_path)
    india_simplified = india.geometry.simplify(0.1)
    
    # Get India's bounds
    bounds = india_simplified.total_bounds
    
    # Create base map centered on India
    m = folium.Map(location=[20.5937, 78.9629], zoom_start=5, tiles='CartoDB positron')
    
    # Create a grid with a 1.5-degree resolution
    lat_range = np.arange(bounds[1], bounds[3], 1.5)
    lon_range = np.arange(bounds[0], bounds[2], 1.5)
    
    solar_data = []
    
    # Fetch data for grid points within India's boundary
    for lat in lat_range:
        for lon in lon_range:
            point = gpd.points_from_xy([lon], [lat])[0]
            
            # Check if the point lies within India's boundary
            if any(india_simplified.contains(point)):
                power_data = fetch_nasa_power_data(lat, lon)
                potential = calculate_solar_potential(power_data)
                
                if potential is not None:
                    solar_data.append({
                        'latitude': lat,
                        'longitude': lon,
                        'potential': potential
                    })
                
                time.sleep(0.5)  # Slight delay to avoid hitting rate limits
    
    solar_df = pd.DataFrame(solar_data)
    
    if solar_df.empty:
        raise ValueError("No solar data collected")
    
    # Create interpolation grid
    grid_size = 100  # Finer grid for better detail
    grid_lat = np.linspace(bounds[1], bounds[3], grid_size)
    grid_lon = np.linspace(bounds[0], bounds[2], grid_size)
    grid_lon, grid_lat = np.meshgrid(grid_lon, grid_lat)
    
    # Interpolate using linear method
    points = solar_df[['longitude', 'latitude']].values
    values = solar_df['potential'].values
    grid_z = griddata(points, values, (grid_lon, grid_lat), method='linear')
    
    # Mask points outside India using simplified geometry
    mask = np.zeros_like(grid_z, dtype=bool)
    grid_points = gpd.GeoDataFrame(geometry=gpd.points_from_xy(grid_lon.flatten(), grid_lat.flatten()))
    mask_points = grid_points.geometry.apply(lambda pt: any(india_simplified.contains(pt))).values
    mask = mask_points.reshape(grid_z.shape)
    grid_z[~mask] = np.nan
    
    # Define colormap for solar potential
    colormap = cm.LinearColormap(
        colors=['#313695', '#4575b4', '#74add1', '#abd9e9', '#fee090', '#fdae61', '#f46d43', '#d73027', '#a50026'],
        vmin=np.nanmin(values),
        vmax=np.nanmax(values)
    )
    
    # Convert the interpolated layer to RGBA using colormap
    img_array = colormap(grid_z, alpha=0.8)
    
    # Add the interpolated layer to the map
    image_bounds = [[bounds[1], bounds[0]], [bounds[3], bounds[2]]]
    folium.raster_layers.ImageOverlay(
        image=img_array,
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
    geojson_path = '/mnt/data/image.png'
    
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
