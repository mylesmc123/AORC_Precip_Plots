'''
Plot AORC data for multiple events
Each event will have multiple .nc files
Aggregate the data for each event and plot the data at a specific location(s).
Two plots will be created:
1. Incremental
2. Cumulative
'''
# %%
import xarray as xr
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import glob
import geopandas as gpd

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
# set raster crs to domain crs
import rioxarray as rxr

crs = rasterio.crs.CRS({"init": "epsg:4326"})
raster = raster.rio.reproject(crs)

#%%
# clip to extent
clipped_raster, transform = mask(raster, [domain.geometry], crop=True)
# plot the clipped raster
show(clipped_raster[0])
# %%
