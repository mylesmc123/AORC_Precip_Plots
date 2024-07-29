# %%
import os
import xarray as xr
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import glob
import geopandas as gpd
import rasterio as rio
import rasterio.crs
from rasterio.warp import calculate_default_transform, reproject, Resampling
from rasterio import mask as msk
import shapely
import rasterstats as rstats

# Define the location of the data, assumes the data will be in a directory of the form:
# /data_dir/event/*.nc
data_dir = r"V:\projects\p00659_dec_glo_phase3\00_collection\Precip"
events = [
    'Rita',
]
domain = gpd.read_file('GLO RAS Domain.geojson')
# get extent of the domain
extent = domain.total_bounds
w, s, e, n = extent[0]-1, extent[1]-2, extent[2]+1, extent[3]+1

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

# %%
files = glob.glob(f'{data_dir}/{events[0]}/*.nc4')
data = xr.open_mfdataset(files)

#%%
data
# %%
# crop the data to the extent of the domain
data = data.sel(latitude=slice(s, n), longitude=slice(w, e))

# %%
# len time
len(data.time)

# %% convert to inches
data['APCP_surface'] = data['APCP_surface'] * 0.0393701
# set units to inches
data['APCP_surface'].attrs['units'] = 'inches'
# %%
data['APCP_surface'].isel(time=350).plot()
# %%
# overlay the domain on the plot with the data
fig, ax = plt.subplots()
data['APCP_surface'].isel(time=350).plot(ax=ax)
domain.boundary.plot(ax=ax, edgecolor='red')
plt.show()
# %%
# accumulate the precipitation data through time
data['precip_cumul'] = data['APCP_surface'].sum(dim='time')

# # export as raster geoTIFF using xarray
# data['precip_cumul'].rio.to_raster(f'{events[0]}_cumulative_precip_inches.tif')
# # open the raster
# raster = rio.open(f'{events[0]}_cumulative_precip_inches.tif')
# %%
# plot the accumulated precipitation over the domain
fig, ax = plt.subplots()
data['precip_cumul'].plot(ax=ax)
domain.boundary.plot(ax=ax, edgecolor='red')
plt.show()
# %%
# set the crs for the data
data.rio.set_crs('EPSG:4326')

# save the cumulative data to a raster
data['precip_cumul'].rio.to_raster(f'{events[0]}_test.tif')

dst_crs = 'EPSG:4326'
projected_dir = './projected'
cropRasterByMask = './cropped'
raster_fn = f'{events[0]}_test.tif'
cropped_raster_fn = f'{events[0]}_test_cropped.tif'
geom = []
coord = shapely.geometry.mapping(domain)["features"][0]["geometry"]
geom.append(coord)
with rasterio.open(raster_fn) as raster:    
    projected_raster, kwargs = projectRaster(raster_fn, raster, projected_dir, dst_crs)
with rasterio.open(projected_raster) as projected_raster:
    out_image, out_transform = msk.mask(projected_raster, 
                                        geom, 
                                        crop=True,
                                        nodata=0)
    out_meta = projected_raster.meta
    out_meta.update({"driver": "GTiff",
                "height": out_image.shape[1],
                "width": out_image.shape[2],
                "transform": out_transform})

    # write the cropped raster to disk
    with rasterio.open(os.path.join(cropped_raster_fn), "w", **out_meta) as dest:
        dest.write(out_image)

with rasterio.open(cropped_raster_fn) as cropped_src:
    profile = cropped_src.profile

# %%
r_stats = []
for i in range(data['APCP_surface'].shape[0]):
    # get the average precipitation for the timestep
    r = rstats.zonal_stats(domain, data['APCP_surface'][i].values, affine=profile['transform'], stats='mean')
    r_stats.append(r[0]['mean'])
# combine r_stats and data.time to create a pandas dataframe
df = pd.DataFrame(r_stats, columns=['mean_precip'])
df['Time'] = data.time.values
df.set_index('Time', inplace=True)
df.plot()

# %%
# accumulate the mean precipitation
df['cumulative_precip'] = df['mean_precip'].cumsum()
df['cumulative_precip'].plot()
# %%
# get the range of Time in df
df.index.min(), df.index.max()

# %%
# save the dataframe to a csv
df.to_csv(f'{events[0]}_test.csv')