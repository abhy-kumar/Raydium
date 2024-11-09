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

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def create_solar_map(solar_data_path='india_solar_data.csv', geojson_path='india-soi.geojson'):
    solar_df = pd.read_csv(solar_data_path)
    india = gpd.read_file(geojson_path)
    india_proj = india.to_crs('EPSG:32644')

    solar_gdf = gpd.GeoDataFrame(
        solar_df,
        geometry=[Point(xy) for xy in zip(solar_df.longitude, solar_df.latitude)],
        crs='EPSG:4326'
    )
    solar_gdf_proj = solar_gdf.to_crs(india_proj.crs)

    bounds = india_proj.total_bounds
    resolution = 5000
    width = int((bounds[2] - bounds[0]) / resolution)
    height = int((bounds[3] - bounds[1]) / resolution)
    transform = from_bounds(bounds[0], bounds[1], bounds[2], bounds[3], width, height)

    raster_data = np.zeros((height, width), dtype=np.float32)
    raster_data[:] = np.nan

    for idx, row in solar_gdf_proj.iterrows():
        col_float, row_float = ~transform * (row.geometry.x, row.geometry.y)
        col, row_idx = int(col_float), int(row_float)
        if 0 <= row_idx < height and 0 <= col < width:
            raster_data[row_idx, col] = row['potential']

    geometry = [feature['geometry'] for feature in india_proj.__geo_interface__['features']]
    mask_raster = geometry_mask(geometry, out_shape=(height, width), transform=transform, invert=True)
    raster_data = np.ma.masked_array(raster_data, ~mask_raster)

    fig, ax = plt.subplots(figsize=(20, 20))
    ax.set_axis_off()

    colormap = cm.LinearColormap(
        ['#440154', '#482878', '#3E4989', '#31688E', '#26828E',
         '#1F9E89', '#35B779', '#6DCD59', '#B4DE2C', '#FDE725'],
        vmin=np.nanmin(raster_data),
        vmax=np.nanmax(raster_data),
        caption='Solar Potential (kWh/m²/year)'
    )

    img = ax.imshow(
        raster_data,
        extent=india_proj.total_bounds,
        cmap=ListedColormap(colormap.colors),
        origin='lower'
    )

    india_proj.boundary.plot(ax=ax, color='black', linewidth=1)
    plt.savefig('solar_potential_high_res.png', bbox_inches='tight', pad_inches=0, dpi=300)
    plt.close(fig)

    center_lat = (india.total_bounds[1] + india.total_bounds[3]) / 2
    center_lon = (india.total_bounds[0] + india.total_bounds[2]) / 2
    m = folium.Map(location=[center_lat, center_lon], zoom_start=5, tiles='CartoDB positron')

    img_bounds = [[india.total_bounds[1], india.total_bounds[0]], 
                  [india.total_bounds[3], india.total_bounds[2]]]

    folium.raster_layers.ImageOverlay(
        name='Solar Potential',
        image='solar_potential_high_res.png',
        bounds=img_bounds,
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
    m.save('india_solar_potential.html')

if __name__ == "__main__":
    create_solar_map()