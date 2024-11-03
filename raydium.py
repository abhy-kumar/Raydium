import numpy as np
import pandas as pd
import geopandas as gpd
import folium
import requests
from shapely.geometry import Point
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
        print(f"Error fetching data for lat={lat}, lon={lon}: {e}")
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

def create_india_solar_map(geojson_path='india-soi.geojson'):
    # Read India GeoJSON
    print(f"Reading GeoJSON file from: {geojson_path}")
    india = gpd.read_file(geojson_path)
    
    # Simplify the geometry for better performance
    india_simplified = india.geometry.simplify(0.1)
    
    # Combine all geometries into a single geometry
    india_union = india_simplified.unary_union
    
    # Get India's bounds
    bounds = india_union.bounds  # (minx, miny, maxx, maxy)
    print(f"Map bounds: {bounds}")
    
    # Create base map centered on India
    m = folium.Map(location=[20.5937, 78.9629], zoom_start=5, tiles='CartoDB positron')
    
    # Create a grid with a 2-degree resolution (can be adjusted)
    lat_range = np.arange(bounds[1], bounds[3], 2)
    lon_range = np.arange(bounds[0], bounds[2], 2)
    
    solar_data = []
    total_points = len(lat_range) * len(lon_range)
    processed_points = 0
    
    print(f"Starting data collection for {total_points} potential grid points...")
    
    # Fetch data for grid points within India's boundary
    for lat in lat_range:
        for lon in lon_range:
            point = Point(lon, lat)
            
            # Check if the point lies within India's boundary
            if india_union.contains(point):
                processed_points += 1
                print(f"Processing point {processed_points}/{total_points} at lat={lat:.2f}, lon={lon:.2f}")
                
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
    
    print("Creating interpolation grid...")
    # Create interpolation grid
    grid_size = 100  # Finer grid for better visualization
    grid_lat = np.linspace(bounds[1], bounds[3], grid_size)
    grid_lon = np.linspace(bounds[0], bounds[2], grid_size)
    grid_lon, grid_lat = np.meshgrid(grid_lon, grid_lat)
    
    # Interpolate using linear method
    points = solar_df[['longitude', 'latitude']].values
    values = solar_df['potential'].values
    grid_z = griddata(points, values, (grid_lon, grid_lat), method='linear')
    
    # Create GeoDataFrame for grid points
    grid_points = gpd.GeoDataFrame(
        geometry=gpd.points_from_xy(grid_lon.flatten(), grid_lat.flatten()),
        crs=india.crs
    )
    
    # Create mask using the unified geometry
    mask = np.array([india_union.contains(point) for point in grid_points.geometry])
    mask = mask.reshape(grid_z.shape)
    grid_z[~mask] = np.nan
    
    print("Creating visualization...")
    # Define colormap for solar potential with new color scheme
    colormap = cm.LinearColormap(
        colors=['#1a237e',  # Dark blue (lowest)
                '#4fc3f7',   # Light blue
                '#81c784',   # Light green
                '#2e7d32',   # Medium green
                '#1b5e20'],  # Dark green (highest)
        vmin=np.nanmin(values),
        vmax=np.nanmax(values)
    )
    
    # Apply colormap
    img = colormap(grid_z)
    image_bounds = [[bounds[1], bounds[0]], [bounds[3], bounds[2]]]
    
    folium.raster_layers.ImageOverlay(
        image=img,
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
    output_csv = 'india_solar_data.csv'
    output_html = 'india_solar_potential.html'
    
    print(f"Saving data to {output_csv}")
    solar_df.to_csv(output_csv, index=False)
    
    print(f"Saving map to {output_html}")
    m.save(output_html)
    
    return m, solar_df

def main():
    """
    Main function to run the application
    """
    try:
        print("Starting solar potential map generation...")
        solar_map, solar_data = create_india_solar_map()
        print("\nProcess completed successfully!")
        print("Map has been saved as 'india_solar_potential.html'")
        print("Raw data has been saved as 'india_solar_data.csv'")
        
    except Exception as e:
        print(f"An error occurred: {e}")

if __name__ == "__main__":
    main()
