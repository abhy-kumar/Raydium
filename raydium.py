import numpy as np
import pandas as pd
import geopandas as gpd
import folium
from folium import plugins
import requests
from scipy.interpolate import griddata
import time
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import branca.colormap as bcm

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
    Create an interactive map of India's solar potential with interpolated contours
    """
    # Read India GeoJSON
    india = gpd.read_file(geojson_path)
    
    # Create a base map centered on India
    m = folium.Map(location=[20.5937, 78.9629], zoom_start=5)

    # Create a grid for sampling points
    lat_range = np.arange(8.4, 37.6, 1.0)  # Using 1.0 degree step for faster processing
    lon_range = np.arange(68.7, 97.25, 1.0)

    # Store solar potential data
    solar_data = []

    # Fetch data for each grid point
    total_points = len(lat_range) * len(lon_range)
    current_point = 0

    for lat in lat_range:
        for lon in lon_range:
            current_point += 1
            print(f"Processing point {current_point}/{total_points}")

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

    # Create a finer mesh grid for interpolation
    grid_lat = np.linspace(solar_df['latitude'].min(), solar_df['latitude'].max(), 500)
    grid_lon = np.linspace(solar_df['longitude'].min(), solar_df['longitude'].max(), 500)
    grid_lon, grid_lat = np.meshgrid(grid_lon, grid_lat)

    # Interpolate the data
    points = solar_df[['longitude', 'latitude']].values
    values = solar_df['potential'].values
    grid_z = griddata(points, values, (grid_lon, grid_lat), method='cubic')

    # Mask points outside India using GeoPandas
    grid_points = gpd.GeoDataFrame(geometry=gpd.points_from_xy(grid_lon.flatten(), grid_lat.flatten()))
    grid_points['z'] = grid_z.flatten()
    grid_points = grid_points.dropna(subset=['z'])
    grid_points = grid_points[grid_points['z'] != -999]

    # Optionally, you can clip the interpolated data to India's boundaries
    grid_clip = gpd.clip(grid_points, india.unary_union)

    # Add the heatmap layer (optional: you can remove this if you prefer only contours)
    locations = [[lat, lon] for lat, lon in zip(solar_df['latitude'], solar_df['longitude'])]
    heatmap_values = solar_df['potential'].tolist()

    hm = plugins.HeatMap(
        locations,
        heatmap_values,
        min_opacity=0.4,
        radius=15,
        blur=15,
        max_zoom=7,
        gradient={
            0.4: '#ffffb2',
            0.6: '#fecc5c',
            0.7: '#fd8d3c',
            0.8: '#f03b20',
            1.0: '#bd0026'
        }
    )
    hm.add_to(m)

    # Create a contour plot using matplotlib
    fig, ax = plt.subplots(figsize=(8, 6))
    contour = ax.contourf(grid_lon, grid_lat, grid_z, levels=15, cmap='viridis', alpha=0.6)

    # Save the contour as a PNG image with transparent background
    plt.axis('off')
    contour_cb = fig.colorbar(contour, ax=ax, orientation='vertical', shrink=0.5)
    plt.savefig('contour.png', bbox_inches='tight', pad_inches=0, transparent=True)
    plt.close()

    # Get the bounds of the grid
    bounds = [grid_lat.min(), grid_lon.min(), grid_lat.max(), grid_lon.max()]

    # Add the contour image as an overlay
    img_path = 'contour.png'
    folium.raster_layers.ImageOverlay(
        name='Solar Potential Contours',
        image=img_path,
        bounds=[[bounds[0], bounds[1]], [bounds[2], bounds[3]]],
        opacity=0.6,
        interactive=True,
        cross_origin=False,
        zindex=1,
    ).add_to(m)

    # Add GeoJSON boundary
    folium.GeoJson(
        india,
        style_function=lambda x: {
            'fillColor': 'none',
            'color': 'black',
            'weight': 1.5
        }
    ).add_to(m)

    # Add layer control
    folium.LayerControl().add_to(m)

    # Add legend using branca
    min_val = solar_df['potential'].min()
    max_val = solar_df['potential'].max()

    colormap = bcm.linear.Viridis_09.scale(min_val, max_val)
    colormap.caption = 'Solar Potential (kWh/m²/year)'
    colormap.add_to(m)

    # Save the solar potential data
    solar_df.to_csv('india_solar_data.csv', index=False)

    return m, solar_df

def main():
    """
    Main function to run the application
    """
    geojson_path = 'india-soi.geojson'  # Ensure this path is correct and the file exists

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
