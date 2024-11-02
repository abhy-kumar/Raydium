import numpy as np
import pandas as pd
import geopandas as gpd
import folium
from folium import plugins
import requests
from scipy.interpolate import griddata
import time

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

    # Create a finer mesh grid for interpolation
    grid_lat = np.linspace(solar_df['latitude'].min(), solar_df['latitude'].max(), 200)
    grid_lon = np.linspace(solar_df['longitude'].min(), solar_df['longitude'].max(), 200)
    grid_lon, grid_lat = np.meshgrid(grid_lon, grid_lat)

    # Interpolate the data
    points = solar_df[['longitude', 'latitude']].values
    values = solar_df['potential'].values
    grid_z = griddata(points, values, (grid_lon, grid_lat), method='cubic')

    # Add the heatmap layer
    locations = [[lat, lon] for lat, lon in zip(solar_df['latitude'], solar_df['longitude'])]
    values = solar_df['potential'].tolist()

    hm = plugins.HeatMap(
        locations,
        values,
        min_opacity=0.4,
        radius=25,
        blur=15,
        gradient={
            0.4: '#ffffb2',
            0.6: '#fecc5c',
            0.7: '#fd8d3c',
            0.8: '#f03b20',
            1.0: '#bd0026'
        }
    )
    hm.add_to(m)

    # Add GeoJSON boundary
    folium.GeoJson(
        india,
        style_function=lambda x: {
            'fillColor': 'none',
            'color': 'black',
            'weight': 1.5
        }
    ).add_to(m)

    # Add legend
    legend_html = f'''
        <div style="position: fixed;
                    bottom: 50px; right: 50px; width: 250px;
                    border:2px solid grey; z-index:9999; font-size:14px;
                    background-color:white;
                    padding: 10px;
                    border-radius: 5px;">
        <p style="margin: 0 0 5px 0;"><strong>Solar Potential (kWh/m²/year)</strong></p>
        <table style="width:100%; border-collapse: collapse;">
    '''

    # Add color gradient and value ranges
    min_val = min(values)
    max_val = max(values)
    colors = ['#ffffb2', '#fecc5c', '#fd8d3c', '#f03b20', '#bd0026']
    step = (max_val - min_val) / len(colors)
    
    for i, color in enumerate(colors):
        value = min_val + i * step
        legend_html += f'''
            <tr>
                <td style="width:30px; background-color:{color}; height:20px;"></td>
                <td style="padding-left:10px;">{value:.0f} - {value + step:.0f}</td>
            </tr>
        '''

    legend_html += '''
        </table>
        </div>
    '''

    m.get_root().html.add_child(folium.Element(legend_html))

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
