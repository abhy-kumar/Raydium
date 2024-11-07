import pandas as pd
import geopandas as gpd
import folium
from shapely.geometry import Point
from scipy.interpolate import griddata
import numpy as np
import matplotlib.pyplot as plt
import branca.colormap as cm
from matplotlib.colors import ListedColormap
import logging

# Logging setup
logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(message)s')
logger = logging.getLogger(__name__)

def create_visualization(data_csv='india_solar_data.csv', 
                       geojson_path='india-soi.geojson',
                       output_html='india_solar_potential.html', 
                       output_image='solar_potential_high_res.png'):
    
    logger.info("Reading GeoJSON file")
    india = gpd.read_file(geojson_path)
    
    # Ensure the GeoJSON is in the correct projection (WGS84)
    india = india.to_crs('EPSG:4326')
    bounds = india.total_bounds
    logger.info(f"Map bounds: {bounds}")

    logger.info(f"Reading solar data from {data_csv}")
    solar_df = pd.read_csv(data_csv)
    if solar_df.empty:
        raise ValueError("No solar data to visualize")

    logger.info("Creating visualization...")
    
    # Adjust grid size and aspect ratio
    height = bounds[3] - bounds[1]
    width = bounds[2] - bounds[0]
    aspect_ratio = width / height
    
    # Calculate grid sizes maintaining the aspect ratio
    base_grid_size = 350
    grid_size_y = base_grid_size
    grid_size_x = int(base_grid_size * aspect_ratio)
    
    # Create properly spaced grid
    grid_lat = np.linspace(bounds[1], bounds[3], grid_size_y)
    grid_lon = np.linspace(bounds[0], bounds[2], grid_size_x)
    grid_lon, grid_lat = np.meshgrid(grid_lon, grid_lat)

    # Interpolating solar potential data
    points = solar_df[['longitude', 'latitude']].values
    values = solar_df['potential'].values
    
    # Use 'cubic' interpolation with adjusted minimum points
    grid_z = griddata(points, values, (grid_lon, grid_lat), method='cubic', rescale=True)

    # Create more precise mask using shapely
    india_union = india.unary_union
    grid_points = [Point(lon, lat) for lon, lat in zip(grid_lon.flatten(), grid_lat.flatten())]
    mask = np.array([india_union.contains(point) for point in grid_points]).reshape(grid_z.shape)
    grid_z = np.ma.masked_array(grid_z, ~mask)

    # Custom colormap with improved range handling
    colormap = cm.LinearColormap(
        ['#440154', '#482878', '#3E4989', '#31688E', '#26828E', '#1F9E89', '#35B779', '#6DCD59', '#B4DE2C', '#FDE725'],
        vmin=np.nanmin(values), vmax=np.nanmax(values),
        caption='Solar Potential (kWh/m²/year)'
    )

    # Create high-res static image with proper aspect ratio
    fig = plt.figure(figsize=(20, 20/aspect_ratio))
    ax = fig.add_subplot(111)
    ax.set_axis_off()
    
    # Use proper extent and aspect
    img = ax.imshow(grid_z, 
                   cmap=ListedColormap(colormap.colors),
                   extent=[bounds[0], bounds[2], bounds[1], bounds[3]],
                   origin='lower',
                   aspect='auto')
    
    # Add boundary with higher resolution
    india.boundary.plot(ax=ax, color='black', linewidth=1)
    
    # Adjust layout to remove padding
    plt.subplots_adjust(left=0, right=1, bottom=0, top=1)
    logger.info(f"Saving high-resolution image to {output_image}")
    plt.savefig(output_image, bbox_inches='tight', pad_inches=0, dpi=300)
    plt.close(fig)

    # Create Folium map with corrected overlay
    logger.info("Creating Folium map with overlay")
    center_lat = (bounds[1] + bounds[3]) / 2
    center_lon = (bounds[0] + bounds[2]) / 2
    
    m = folium.Map(
        location=[center_lat, center_lon],
        zoom_start=5,
        tiles='CartoDB positron'
    )

    # Add image overlay with precise bounds
    folium.raster_layers.ImageOverlay(
        image=output_image,
        bounds=[[bounds[1], bounds[0]], [bounds[3], bounds[2]]],
        opacity=0.7,
        interactive=True,
        cross_origin=False,
        zindex=1
    ).add_to(m)

    # Add GeoJSON boundary with higher precision
    folium.GeoJson(
        india,
        style_function=lambda x: {
            'color': 'black',
            'weight': 1.5,
            'fillOpacity': 0
        }
    ).add_to(m)

    colormap.add_to(m)
    folium.LayerControl().add_to(m)
    
    logger.info(f"Saving interactive map to {output_html}")
    m.save(output_html)

def main():
    create_visualization()

if __name__ == "__main__":
    main()
