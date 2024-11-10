import geopandas as gpd
import pandas as pd
import numpy as np
import folium
import branca.colormap as cm
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from rasterio.transform import from_bounds
from rasterio.features import geometry_mask
from shapely.geometry import Point
import logging
from scipy.ndimage import gaussian_filter

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def create_solar_map(solar_data_path='india_solar_data.csv', geojson_path='india-soi.geojson'):
    # Load and prepare data
    logger.info("Loading data...")
    solar_df = pd.read_csv(solar_data_path)
    india = gpd.read_file(geojson_path)
    
    # Ensure input data is in WGS84
    if india.crs != 'EPSG:4326':
        india = india.to_crs('EPSG:4326')
    
    # Create GeoDataFrame from solar data
    solar_gdf = gpd.GeoDataFrame(
        solar_df,
        geometry=[Point(xy) for xy in zip(solar_df.longitude, solar_df.latitude)],
        crs='EPSG:4326'
    )
    
    # Use Asia South Albers Equal Area projection instead of Web Mercator
    # This preserves area relationships better for India's latitude
    india_proj = india.to_crs('ESRI:102028')
    solar_gdf_proj = solar_gdf.to_crs('ESRI:102028')
    
    # Calculate bounds and create raster with higher resolution
    bounds = india_proj.total_bounds
    resolution = 2000  # 2km resolution
    width = int((bounds[2] - bounds[0]) / resolution)
    height = int((bounds[3] - bounds[1]) / resolution)
    transform = from_bounds(bounds[0], bounds[1], bounds[2], bounds[3], width, height)
    
    # Initialize raster
    raster_data = np.zeros((height, width), dtype=np.float32)
    weight_matrix = np.zeros((height, width), dtype=np.float32)
    
    # Populate raster with improved interpolation
    for idx, row in solar_gdf_proj.iterrows():
        x, y = row.geometry.x, row.geometry.y
        col, row_idx = ~transform * (x, y)
        col, row_idx = int(col), int(row_idx)
        
        if 0 <= row_idx < height and 0 <= col < width:
            # Use inverse distance weighting for better interpolation
            raster_data[row_idx, col] += row['potential']
            weight_matrix[row_idx, col] += 1
    
    # Average values where multiple points exist
    mask = weight_matrix > 0
    raster_data[mask] = raster_data[mask] / weight_matrix[mask]
    
    # Create and apply mask for India's boundaries
    geometry = [feature['geometry'] for feature in india_proj.__geo_interface__['features']]
    mask_raster = geometry_mask(geometry, out_shape=(height, width), transform=transform, invert=True)
    
    # Apply mask
    raster_data = np.ma.masked_array(raster_data, ~mask_raster)
    
    # Apply gentler smoothing
    filled_data = raster_data.filled(np.nan)
    smoothed_data = gaussian_filter(filled_data, sigma=1)
    smoothed_masked = np.ma.masked_array(smoothed_data, ~mask_raster)
    
    # Create visualization
    fig, ax = plt.subplots(figsize=(20, 20), dpi=300)
    ax.set_axis_off()
    
    # Calculate proper value ranges using percentiles
    valid_data = smoothed_masked.compressed()
    vmin = float(np.percentile(valid_data, 2))
    vmax = float(np.percentile(valid_data, 98))
    
    # Use a more appropriate colormap for solar radiation
    colors = ['#313695', '#4575b4', '#74add1', '#abd9e9', '#e0f3f8',
              '#fee090', '#fdae61', '#f46d43', '#d73027', '#a50026']
    colormap = cm.LinearColormap(
        colors=colors,
        vmin=vmin,
        vmax=vmax,
        caption='Solar Potential (kWh/m²/year)'
    )
    
    # Plot with correct orientation
    img = ax.imshow(
        smoothed_masked,
        extent=india_proj.total_bounds,
        cmap=ListedColormap(colormap.colors),
        origin='upper',  # Changed from 'lower' to 'upper'
        interpolation='nearest',
        vmin=vmin,
        vmax=vmax
    )
    
    # Add boundary outline
    india_proj.boundary.plot(ax=ax, color='black', linewidth=1)
    
    # Add colorbar
    plt.colorbar(img, ax=ax, label='Solar Potential (kWh/m²/year)')
    
    # Save high-resolution image
    plt.savefig('solar_potential_high_res.png', 
                bbox_inches='tight', 
                pad_inches=0.1, 
                dpi=300,
                facecolor='white',
                edgecolor='none')
    plt.close(fig)
    
    # Create interactive Folium map with correct projection
    india_wgs84 = india_proj.to_crs('EPSG:4326')
    bounds_latlon = india_wgs84.total_bounds
    center_lat = (bounds_latlon[1] + bounds_latlon[3]) / 2
    center_lon = (bounds_latlon[0] + bounds_latlon[2]) / 2
    
    m = folium.Map(
        location=[center_lat, center_lon],
        zoom_start=5,
        tiles='CartoDB positron'
    )
    
    # Add image overlay with correct coordinates
    img_bounds = [
        [bounds_latlon[1], bounds_latlon[0]],
        [bounds_latlon[3], bounds_latlon[2]]
    ]
    
    folium.raster_layers.ImageOverlay(
        name='Solar Potential',
        image='solar_potential_high_res.png',
        bounds=img_bounds,
        opacity=0.7,
        interactive=True,
        cross_origin=False,
        zindex=1,
    ).add_to(m)
    
    # Add boundary overlay
    folium.GeoJson(
        india_wgs84.__geo_interface__,
        style_function=lambda x: {
            'fillColor': 'none',
            'color': 'black',
            'weight': 1.5,
            'fillOpacity': 0
        },
        name='India Boundary'
    ).add_to(m)
    
    # Add colormap and layer control
    colormap.add_to(m)
    folium.LayerControl().add_to(m)
    
    # Save interactive map
    m.save('india_solar_potential.html')
    logger.info("Map creation completed!")

if __name__ == "__main__":
    create_solar_map()