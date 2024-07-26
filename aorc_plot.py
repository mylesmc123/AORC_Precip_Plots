'''
Plot AORC data for multiple events
Each event will have multiple .nc files
Aggregate the data for each event and plot the data at a specific location(s).
Two plots will be created:
1. Incremental
2. Cumulative
'''

import xarray as xr
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import glob


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

for point in locations:
    # For each event, read in the data and aggregate the data for each location
    for event in events:
        # Get the list of files for the event
        files = glob.glob(f'{data_dir}/{event}/*.nc4')
        # Read in the data
        data = xr.open_mfdataset(files)
        # Get the data for the point location
        data_loc = data.sel(latitude=point['lat'], longitude=point['lon'], method='nearest')
        # Calculate the incremental and cumulative precipitation
        data_loc['precip_inc'] = data_loc['precipitationCal'].diff(dim='time', label='lower')
        data_loc['precip_cum'] = data_loc['precipitationCal'].cumsum(dim='time')
        # Plot the data
        fig, ax = plt.subplots(2, 1, figsize=(10, 10))
        # Incremental plot
        data_loc['precip_inc'].plot(ax=ax[0])
        ax[0].set_title(f'{event} Incremental Precipitation')
        ax[0].set_ylabel('Precipitation (mm)')
        # Cumulative plot
        data_loc['precip_cum'].plot(ax=ax[1])
        ax[1].set_title(f'{event} Cumulative Precipitation')
        ax[1].set_ylabel('Precipitation (mm)')
        plt.show()
        # Save the plot
        plt.savefig(f'{event}_{lat}_{lon}.png')

