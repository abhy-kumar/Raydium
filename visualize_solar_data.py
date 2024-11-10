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
    logger.info(f"Solar Data Rows: {len(solar_df)}")
    india = gpd.read_file(geojson_path)
    logger.info(f"India GeoJSON CRS: {india.crs}")
    logger.info(f"Number of Geometries: {len(india)}")
    
    # Ensure input data is in WGS84
    if india.crs != 'EPSG:4326':
        india = india.to_crs('EPSG:4326')
    
    # Create GeoDataFrame from solar data
    solar_gdf = gpd.GeoDataFrame(
        solar_df,
        geometry=[Point(xy) for xy in zip(solar_df.longitude, solar_df.latitude)],
        crs='EPSG:4326'
    )
    logger.info(f"Solar GeoDataFrame CRS: {solar_gdf.crs}")
    logger.info(f"Solar Points Total Bounds: {solar_gdf.total_bounds}")
    
    # Project to Web Mercator for broader coverage
    india_proj = india.to_crs('EPSG:3857')
    solar_gdf_proj = solar_gdf.to_crs('EPSG:3857')
    
    # Calculate bounds and create raster
    bounds = india_proj.total_bounds
    resolution = 2500  # 2.5km resolution for higher detail
    width = int((bounds[2] - bounds[0]) / resolution)
    height = int((bounds[3] - bounds[1]) / resolution)
    transform = from_bounds(bounds[0], bounds[1], bounds[2], bounds[3], width, height)
    
    logger.info(f"Raster Dimensions: {width}x{height}")
    
    # Initialize raster with zeros instead of NaN
    raster_data = np.zeros((height, width), dtype=np.float32)
    
    # Create weight matrix for averaging multiple points in same cell
    weight_matrix = np.zeros((height, width), dtype=np.float32)
    
    # Populate raster with data points using weighted averaging
    populated = 0
    for idx, row in solar_gdf_proj.iterrows():
        col_float, row_float = ~transform * (row.geometry.x, row.geometry.y)
        col, row_idx = int(col_float), int(row_float)
        if 0 <= row_idx < height and 0 <= col < width:
            raster_data[row_idx, col] += row['potential']
            weight_matrix[row_idx, col] += 1
            populated += 1
    
    # Average values where multiple points exist
    mask = weight_matrix > 0
    raster_data[mask] = raster_data[mask] / weight_matrix[mask]
    
    logger.info(f"Populated Pixels: {populated} out of {len(solar_gdf_proj)}")
    
    if populated == 0:
        logger.error("No data points were mapped to the raster. Check input data and projections.")
        return
    
    # Create and apply mask for India's boundaries
    geometry = [feature['geometry'] for feature in india_proj.__geo_interface__['features']]
    mask_raster = geometry_mask(geometry, out_shape=(height, width), transform=transform, invert=True)
    logger.info(f"Mask Applied. Number of True Pixels: {np.sum(mask_raster)}")
    
    # Apply mask
    raster_data = np.ma.masked_array(raster_data, ~mask_raster)
    
    # Apply Gaussian smoothing to fill gaps
    # Fill masked values with nearest neighbor before smoothing
    filled_data = raster_data.filled(0)  # Fill with 0 instead of NaN
    smoothed_data = gaussian_filter(filled_data, sigma=2)
    
    # Reapply the mask after smoothing
    smoothed_masked = np.ma.masked_array(smoothed_data, ~mask_raster)
    
    if np.all(smoothed_masked.mask):
        logger.error("Smoothed data contains only masked values. Cannot create visualization.")
        return
    
    # Create high-resolution plot
    logger.info("Creating visualization...")
    fig, ax = plt.subplots(figsize=(20, 20), dpi=300)
    ax.set_axis_off()
    
    # Calculate min and max values properly
    valid_data = smoothed_masked.compressed()  # Get only valid (non-masked) data
    if len(valid_data) == 0:
        logger.error("No valid data points after masking.")
        return
        
    vmin = float(np.percentile(valid_data, 1))  # Use 1st percentile instead of minimum
    vmax = float(np.percentile(valid_data, 99))  # Use 99th percentile instead of maximum
    logger.info(f"Visualization Vmin: {vmin}, Vmax: {vmax}")
    
    # Create colormap with properly sorted values
    colormap = cm.LinearColormap(
        colors=['#440154', '#482878', '#3E4989', '#31688E', '#26828E',
                '#1F9E89', '#35B779', '#6DCD59', '#B4DE2C', '#FDE725'],
        vmin=vmin,
        vmax=vmax,
        caption='Solar Potential (kWh/m²/year)'
    )
    
    # Plot raster data
    img = ax.imshow(
        smoothed_masked,
        extent=india_proj.total_bounds,
        cmap=ListedColormap(colormap.colors),
        origin='lower',
        interpolation='bilinear',
        vmin=vmin,
        vmax=vmax
    )
    
    # Add boundary outline
    india_proj.boundary.plot(ax=ax, color='black', linewidth=1)
    
    # Save high-resolution image
    plt.savefig('solar_potential_high_res.png', 
                bbox_inches='tight', 
                pad_inches=0, 
                dpi=300,
                facecolor='white',
                edgecolor='none')
    plt.close(fig)
    
    # Create interactive Folium map
    logger.info("Creating interactive map...")
    
    # Convert bounds back to latitude and longitude for Folium
    india_bounds_latlon = india.to_crs('EPSG:4326').total_bounds
    center_lat = (india_bounds_latlon[1] + india_bounds_latlon[3]) / 2
    center_lon = (india_bounds_latlon[0] + india_bounds_latlon[2]) / 2
    
    m = folium.Map(
        location=[center_lat, center_lon],
        zoom_start=5,
        tiles='CartoDB positron'
    )
    
    # Add image overlay
    img_bounds = [
        [india_bounds_latlon[1], india_bounds_latlon[0]],
        [india_bounds_latlon[3], india_bounds_latlon[2]]
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
        india.__geo_interface__,
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
