import pandas as pd
import geopandas as gpd
import folium
from shapely.geometry import Point
from scipy.interpolate import griddata
import numpy as np
import matplotlib.pyplot as plt
import branca.colormap as cm
from matplotlib.colors import ListedColormap
import os
import logging
import diskcache as dc
from tqdm import tqdm

# Logging setup (same as in raydium.py)
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    handlers=[
        logging.FileHandler("visualization.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

def create_visualization(data_csv='india_solar_data.csv', geojson_path='india-soi.geojson',
                        output_html='india_solar_potential.html',
                        output_image='solar_potential_high_res.png'):
    logger.info("Reading GeoJSON file")
    india = gpd.read_file(geojson_path)
    bounds = india.total_bounds
    logger.info(f"Map bounds: {bounds}")
    
    logger.info(f"Reading solar data from {data_csv}")
    solar_df = pd.read_csv(data_csv)
    if solar_df.empty:
        raise ValueError("No solar data to visualize")
    
    logger.info("Creating visualization...")

    grid_size = 500  # Adjust as necessary
    grid_lat = np.linspace(bounds[1], bounds[3], grid_size)
    grid_lon = np.linspace(bounds[0], bounds[2], grid_size)
    grid_lon, grid_lat = np.meshgrid(grid_lon, grid_lat)

    points = solar_df[['longitude', 'latitude']].values
    values = solar_df['potential'].values
    grid_z = griddata(points, values, (grid_lon, grid_lat), method='linear')

    india_union = india.unary_union
    india_gdf = gpd.GeoSeries([Point(lon, lat) for lon, lat in zip(grid_lon.flatten(), grid_lat.flatten())])
    mask = india_gdf.within(india_union).values.reshape(grid_z.shape)
    grid_z = np.ma.masked_array(grid_z, ~mask)

    colormap = cm.LinearColormap(
        ['#440154', '#482878', '#3E4989', '#31688E', '#26828E',
         '#1F9E89', '#35B779', '#6DCD59', '#B4DE2C', '#FDE725'],
        vmin=np.nanmin(values),
        vmax=np.nanmax(values),
        caption='Solar Potential (kWh/m²/year)'
    )

    fig, ax = plt.subplots(figsize=(20, 24))
    ax.set_axis_off()

    img = ax.imshow(
        grid_z,
        cmap=ListedColormap(colormap.colors),
        extent=[bounds[0], bounds[2], bounds[1], bounds[3]],
        origin='lower',
        aspect='auto'
    )

    india.boundary.plot(ax=ax, color='black', linewidth=1)

    logger.info(f"Saving high-resolution image to {output_image}")
    plt.savefig(output_image, bbox_inches='tight', pad_inches=0, dpi=300)
    plt.close(fig)

    logger.info(f"Creating Folium map and adding image overlay")
    m = folium.Map(location=[(bounds[1] + bounds[3]) / 2, (bounds[0] + bounds[2]) / 2],
                   zoom_start=5, tiles='CartoDB positron')

    folium.raster_layers.ImageOverlay(
        name='Solar Potential',
        image=output_image,
        bounds=[[bounds[1], bounds[0]], [bounds[3], bounds[2]]],
        opacity=0.7,
        interactive=True,
        cross_origin=False,
        zindex=1,
    ).add_to(m)

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

    colormap.add_to(m)
    folium.LayerControl().add_to(m)

    logger.info(f"Saving map to {output_html}")
    m.save(output_html)

def main():
    create_visualization()

if __name__ == "__main__":
    main()