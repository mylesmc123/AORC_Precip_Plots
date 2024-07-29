'''
Plot AORC data for multiple events
Each event will have multiple .nc files
Aggregate the data for each event and plot the data at a specific location(s).
Two plots will be created:
1. Incremental
2. Cumulative
'''
# %%
import os
import xarray as xr
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
import glob
import geopandas as gpd
from rasterio.warp import calculate_default_transform, reproject, Resampling
from rasterio import mask as msk
import shapely
import rasterstats as rstats

# %%
# Define the location of the data, assumes the data will be in a directory of the form:
# /data_dir/event/*.nc
data_dir = r"V:\projects\p00659_dec_glo_phase3\00_collection\Precip"
events = [
    'Harvey',
    'Ike',
    'Rita',
    'March 2022'
]

# Define the coordinates for each location to use for the plots.
# The coordinates are in the form (lat, lon)
# using the USGS Gage for GLO (29.6756, -94.6661)
locations = [{
    'name': 'USGS Gage 08042558',
    'lat': 29.6756,
    'lon': -94.6661
}]
# %%
files = glob.glob(f'{data_dir}/{events[0]}/*.nc4')
data = xr.open_mfdataset(files)
data
# %%
# Clip the data to the extent of the model domain
# Use the model domain geojson to clip the data

# Read in the model domain geojson
domain = gpd.read_file('GLO RAS Domain.geojson')
# %%
# get extent of the domain
extent = domain.total_bounds
w, s, e, n = extent[0]-1, extent[1]-2, extent[2]+1, extent[3]+1
# extent = {
#     'north': n,
#     'south': s,
#     'east': e,
#     'west': w
# }
# %%
data = data.sel(latitude=slice(s, n), longitude=slice(w, e))
data
# %%

# sum and convert to inches
data['precip_cumul'] = data['APCP_surface'].sum(dim='time') * 0.0393701
# plot
data['precip_cumul'].plot()
# %%
# export as raster geoTIFF using xarray
data['precip_cumul'].rio.to_raster(f'{events[0]}_cumulative_precip_inches.tif')
# %%
# clip the raster to the extent of the model domain
import rasterio as rio
import rasterio.crs
from rasterio.plot import show
from rasterio.mask import mask

# open the raster
raster = rio.open(f'{events[0]}_cumulative_precip_inches.tif')

# %%
# get raster extent
raster_extent = rio.plot.plotting_extent(raster)
raster_extent
# %%
domain.geometry.bounds
# %%

def projectRaster(raster_fn, raster_src, projected_dir, dst_crs):
    transform, width, height = calculate_default_transform(
            raster_src.crs, dst_crs, raster_src.width, raster_src.height, *raster_src.bounds)
    kwargs = raster_src.meta.copy()
    kwargs.update({
        'crs': dst_crs,
        'transform': transform,
        'width': width,
        'height': height
    })

    projected_raster = projected_dir + '\\' + raster_fn.split("\\")[-1]
    with rasterio.open(projected_raster, 'w', **kwargs) as dst:
        for i in range(1, raster_src.count + 1):
            reproject(
                source=rasterio.band(raster_src, i),
                destination=rasterio.band(dst, i),
                src_transform=raster_src.transform,
                src_crs=raster_src.crs,
                dst_transform=transform,
                dst_crs=dst_crs,
                resampling=Resampling.nearest)
    return projected_raster, kwargs

def cropRasterByMask(src_projected, crop_shp, cropped_raster_fn):
    geom = []
    coord = shapely.geometry.mapping(crop_shp)["features"][0]["geometry"]
    geom.append(coord)
    out_image, out_transform = msk.mask(src_projected, 
                                        geom, 
                                        crop=True,
                                        nodata=0)
    out_meta = src_projected.meta
    out_meta.update({"driver": "GTiff",
                    "height": out_image.shape[1],
                    "width": out_image.shape[2],
                    "transform": out_transform})

    with rasterio.open(os.path.join(cropped_raster_fn), "w", **out_meta) as dest:
        dest.write(out_image)
# %%
dst_crs = 'EPSG:4326'
projected_dir = './projected'
cropRasterByMask = './cropped'
raster_fn = f'{events[0]}_cumulative_precip_inches.tif'
projected_raster, kwargs = projectRaster(raster_fn, raster, projected_dir, dst_crs)

# %%
src_projected = rasterio.open(projected_raster)
cropped_raster_fn = f'{events[0]}_cumulative_precip_inches_cropped.tif'

# %%
# cropRasterByMask(src_projected, domain, cropped_raster_fn)
geom = []
coord = shapely.geometry.mapping(domain)["features"][0]["geometry"]
geom.append(coord)
out_image, out_transform = msk.mask(src_projected, 
                                        geom, 
                                        crop=True,
                                        nodata=0)
out_meta = src_projected.meta
out_meta.update({"driver": "GTiff",
                "height": out_image.shape[1],
                "width": out_image.shape[2],
                "transform": out_transform})
# %%
with rasterio.open(os.path.join(cropped_raster_fn), "w", **out_meta) as dest:
        dest.write(out_image)
# %%
# plot the cropped raster with  colorbar and 0 as nodata.
cropped_src = rasterio.open(cropped_raster_fn)
profile = cropped_src.profile
array = cropped_src.read(1)
array[array==0] = np.nan    
array[array==9999] = np.nan
firstArrayShape = array.shape
raster.close()
src_projected.close()
cropped_src.close()
# %%
# plot cropped raster
plt.figure(figsize=(10, 10))
plt.imshow(array, cmap='turbo')
plt.colorbar()
# add title
plt.title(f'{events[0]} Cumulative Precipitation (inches)')
# add colorbar title of the unit
# plt.colorbar.set_label('Inches')
# hide x and y axis
plt.axis('off')
plt.show()

# %%
# Create a basin average timeseries for the event 
# using APCP_surface clipped to the basin domain
# get the average precipitation foor each timestep
import rasterstats as rstats
# for each timestep, get the average precipitation
r_stats = []
for i in range(data['APCP_surface'].shape[0]):
    # get the average precipitation for the timestep
    r = rstats.zonal_stats(domain, data['APCP_surface'][i].values, affine=profile['transform'], stats='mean')
    r_stats.append(r[0]['mean'])
# %%
# combine r_stats and data.time to create a pandas dataframe
df = pd.DataFrame(r_stats, columns=['mean_precip'])
df['Time'] = data.time.values
df.set_index('Time', inplace=True)
# convert mean_precip to inches from mm
df['mean_precip'] = df['mean_precip'] * 0.0393701
df.plot()

# %%
# accumulate the mean precipitation
df['cumulative_precip'] = df['mean_precip'].cumsum()
df['cumulative_precip'].plot()
# %%
# Create subplots for the each of the basin average Incremental and Cumulative precipitation plots
fig, ax = plt.subplots(2, 1, figsize=(8, 8))
# Incremental plot
df['mean_precip'].plot(ax=ax[0])
ax[0].set_title(f'{events[0]} Incremental Precipitation')
ax[0].set_ylabel('Precipitation (inches)')
# remove the x-axis label
ax[0].set_xlabel('')
# remove the x-axis values
ax[0].set_xticks([])
# remove the x-axis tick values
ax[0].tick_params(axis='x', which='both', bottom=False, top=False, labelbottom=False)
# Cumulative plot
df['cumulative_precip'].plot(ax=ax[1])
ax[1].set_title(f'{events[0]} Cumulative Precipitation')
ax[1].set_ylabel('Precipitation (inches)')
# add a buffer between subplots
plt.tight_layout()
plt.savefig(f'{events[0]}_Avg_Precip.png', dpi=300, bbox_inches='tight')
plt.show()
# %%
# Save the plot to disk

# %%
