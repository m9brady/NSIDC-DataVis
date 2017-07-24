# -*- coding: utf-8 -*-
import os
import tempfile
from glob import glob
from datetime import datetime
import numpy as np
import numpy.ma as ma
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.colors as colors

#import statsmodels.api as sm

from mpl_toolkits.basemap import Basemap
from matplotlib import patches

### [Southmost, Northmost, Westmost, Eastmost]
bounds = None
# >60N
bounds = [60.0, 90.0, -180.0, 180.0]
# beaufort
#bounds = [67.0, 80.0, -160.0, -115.0]
# beaufort tighter
#bounds = [68.0, 78.0, -150.0, -123.0]
# svalbard
#bounds = [75.0, 81.0, 5.0, 25.0]
# fram
bounds = [73.0, 82.0, -22.0, 13.0]
# high arctic canada
#bounds = [81.0, 88.0, -110.0, -50.0]
# top 10 lat
#bounds = [80.0, 90.0, -180.0, 180.0]
# top 5 lat
#bounds = [85.0, 90.0, -180.0, 180.0]
# high arctic MYI zone
#bounds = [81.0, 87.0, -90.0, -30.0]


# first week of Sept is 35/36 depending on year
# last week of March is ~12
week = 36
# get today's (local time) weeknum
#week = datetime.today().isocalendar()[1]

# first year in time series is 1984
start_year = 1984
# 1997 to line up with Radarsat1
start_year = 1997


# prelimary workspace setup, assumes the binaries and grid-dats are
# already downloaded and ready to go
# http://dx.doi.org/10.5067/PFSVFZA9Y85G
data_dir = os.path.join(tempfile.gettempdir(), 'NSIDC_SeaIceAge')
os.chdir(data_dir)
lats = os.path.join(data_dir, 'data', 'Na12500-CF_latitude.dat')
lons = os.path.join(data_dir, 'data', 'Na12500-CF_longitude.dat')
lat_arr = np.fromfile(lats, dtype='float32')
lon_arr = np.fromfile(lons, dtype='float32')
age_files = sorted(glob(os.path.join(data_dir, 'data', 'bins','*','iceage.*.bin')))

# probably not necessary, but makes the pandas dataframes look a litte nicer in ipython
frame_cols = np.empty((0,), dtype=[('year',np.uint8), 
                            ('week_num',np.uint8), 
                            ('FYI',np.float64), 
                            ('SYI',np.float64), 
                            ('MYI_3',np.float64), 
                            ('MYI_4',np.float64), 
                            ('MYI_5+',np.float64)])
                            #('OW',np.float64)])
pixcount_frame = pd.DataFrame(frame_cols)
cumulativepct_frame = pd.DataFrame(frame_cols)

# generate a mask if we want to restrict our data to a certain geographic boundary
if bounds:
    mask = (lat_arr >= bounds[0]) & (lat_arr <= bounds[1]) & (lon_arr >= bounds[2]) & (lon_arr <= bounds[3])
# otherwise, keep all 521,284 pixels
else:
    mask = np.array([True for x in xrange(len(lat_arr))])

# iterate through binary files, pulling pixel counts (not used) as well as 
# cumulative percentages for ice age types
for idx, age_file in enumerate(age_files):
    year = int(os.path.basename(age_file).split(".")[3])
    week_num = int(os.path.basename(age_file).split(".")[4])
    file_arr = np.fromfile(age_file, dtype='uint8')
    age_arr = file_arr[mask]
    
    # pixel counts for each ice age class including open-water
    fyi_count = len(age_arr[age_arr == 5])
    syi_count = len(age_arr[age_arr == 10])
    myi3_count = len(age_arr[age_arr == 15])
    myi4_count = len(age_arr[age_arr == 20])
    myi5_count = len(age_arr[np.logical_and(age_arr >= 25, age_arr < 254)])
    #ow_count = len(age_arr[age_arr == 0])
    total_count = fyi_count + syi_count + myi3_count + myi4_count + myi5_count# + ow_count
    
    # percentages of all ice age class pixels (better-looking for arctic aoi)
    myi5_pct = myi5_count*1./total_count
    myi4_pct = myi4_count*1./total_count
    myi3_pct = myi3_count*1./total_count
    syi_pct = syi_count*1./total_count
    fyi_pct = fyi_count*1./total_count
    #ow_pct = ow_count*1./total_count
    
    # cumulative percentages for plotting cumulative plots
    myi5_cp = myi5_pct
    myi4_cp = myi5_cp + myi4_pct
    myi3_cp = myi4_cp + myi3_pct
    syi_cp = myi3_cp + syi_pct
    fyi_cp = syi_cp + fyi_pct
    #ow_cp = fyi_cp + ow_pct
    
    cumulativepct_frame.loc[idx] = [year, week_num, fyi_cp, syi_cp, myi3_cp, myi4_cp, myi5_cp]#, ow_cp]
    pixcount_frame.loc[idx] = [year, week_num, fyi_count, syi_count, myi3_count, myi4_count, myi5_count]#, ow_count]

# generate figure with two horizontally-stacked subplots
fig = plt.figure(figsize=(15,7))
ax = fig.add_subplot(121)
ax_map = fig.add_subplot(122)

# setup the subplot that will contain the EASE-grid map
ll_lat = lat_arr.reshape((722,722))[721,0] 
ll_lon = lon_arr.reshape((722,722))[721,0]
ur_lat = lat_arr.reshape((722,722))[0,721] 
ur_lon = lon_arr.reshape((722,722))[0,721]
b = Basemap(llcrnrlat=ll_lat, 
            llcrnrlon=ll_lon, 
            urcrnrlat=ur_lat, 
            urcrnrlon=ur_lon, 
            area_thresh=5000.,  # skip drawing of features with areas < 5,000sqkm
            resolution='l',  # low-res (nicer than crude-res)
            projection='laea', 
            lat_0=90., 
            lon_0=0., 
            ax=ax_map)
b.drawcoastlines()
b.fillcontinents(color='0.90', lake_color=None) # grey continents with major lakes
# latitude gridlines (does not plot higher than 80N for some reason)
b.drawparallels(np.arange(40,90,10), color='0.25', linewidth=0.5, labels=[1,1,0,0])
# longitude gridlines
b.drawmeridians(np.arange(-180,180,10), color='0.25', linewidth=0.5, labels=[0,0,1,1])
# convert lats/lons to map units
x,y = b(lon_arr, lat_arr)
# mask for the current area of interest
aoi_label = "Area of Interest"
#if not bounds:
#    mask = np.logical_and(file_arr >= 0, file_arr < 254)
#    aoi_label = "Ice Extent"
x2 = x[mask]
y2 = y[mask]


# plot the boundary AOI with 55% transparency to show coastline/graticule underneath
#x2[x2 == 0] = np.nan
#y2[y2 == 0] = np.nan
#b.plot(x2, y2, alpha=0.45, color='#EFAF40', label='_nolegend_')
if not (bounds[3] == 180 and bounds[2] == -180):
    xbnds, ybnds = b([bounds[2], bounds[2], bounds[3], bounds[3]], [bounds[0], bounds[1], bounds[1], bounds[0]]) 
    poly = patches.Polygon(zip(xbnds,ybnds), facecolor='#EFAF40', alpha=0.4)
else:
    pass
    
ax_map.add_patch(poly)
# manually set legend entry for area of interest
pat = patches.Patch(color='#F5CF8C', label=aoi_label)
# add legend to subplot in upper left corner
ax_map.legend(handles=[pat], loc='upper left', framealpha=1.)


#most_current = age_files.index()
f = file_arr.copy().astype(float)
#f = ma.masked_array(f, mask=~mask, fill_value=np.nan)
# pixels older than 5 years will be coloured the same as myi5
f[np.logical_and(f <> 255, f > 30)] = 30
# mask out the land and out-of-current-bounds pixels
f[np.where(f == 255)] *= np.nan
f[~mask] *= np.nan

# get unique pixel values, for setting up the colour mapping
uni = np.unique(f[~np.isnan(f)]).tolist()
# hex values
clist = ['#FFFFFF', '#2400F5', '#00F6FF', '#15FF00', '#FFC700','#FF0000']
ccmap = colors.ListedColormap(clist)
cnorm = colors.BoundaryNorm(uni, ccmap.N)
lon_range = lon_arr.reshape((722,722))
lat_range = lat_arr.reshape((722,722))
x_range, y_range = b(lon_range, lat_range)
b.pcolormesh(x_range,y_range,f.reshape((722,722)), cmap=ccmap, norm=cnorm)




# subset our dataframe to a specific week and starting year defined at top of script
cpct_frame = cumulativepct_frame.loc[cumulativepct_frame.week_num == week]
cpct_frame = cpct_frame.loc[cpct_frame.year >= start_year]

# plot each ice age class using the pre-calculated cumulative percentages
for c in cpct_frame.columns[2:]:
    cpct_frame.plot('year', c, ax=ax, label='_nolegend_', color='k')

# ColourMap LUT for the cumulative percentage plot
cmap = {'OW':'#78B4FF', 'FYI':'#2400F5', 'SYI':'#00F6FF', 'MYI_3':'#15FF00', 'MYI_4':'#FFC700', 'MYI_5+':'#FF0000'}
# filling the spaces between lines on the plot with the proper colours
#ax.fill_between(cpct_frame.year, cpct_frame['OW'], cpct_frame['FYI'], label='Open Water or <15% Ice', color=cmap['OW'])
ax.fill_between(cpct_frame.year, cpct_frame['FYI'], cpct_frame['SYI'], label='First-Year Ice', color=cmap['FYI'])
ax.fill_between(cpct_frame.year, cpct_frame['SYI'], cpct_frame['MYI_3'], label='Second-Year Ice', color=cmap['SYI'])
ax.fill_between(cpct_frame.year, cpct_frame['MYI_3'], cpct_frame['MYI_4'], label='Third-Year Ice', color=cmap['MYI_3'])
ax.fill_between(cpct_frame.year, cpct_frame['MYI_4'], cpct_frame['MYI_5+'], label='Fourth-Year Ice', color=cmap['MYI_4'])
ax.fill_between(cpct_frame.year, cpct_frame['MYI_5+'], 0, label='Fifth-Year Ice and Older', color=cmap['MYI_5+'])

# general matplotlib figure/subplot formatting
ax.set_xlabel('Year')
x_axis_maj_range = range(int(cpct_frame.year.min()), int(cpct_frame.year.max()+1), 2)
x_axis_min_range = range(int(cpct_frame.year.min()), int(cpct_frame.year.max()+1))
ax.set_xticks(x_axis_maj_range)
ax.set_xticks(x_axis_min_range, minor=True)
ax.set_ylabel('Cumulative Percentage')
ax.set_ybound(0,1)
ax.invert_yaxis()
ax.set_yticks([i/10. for i in xrange(11)])
y_ticklabels = ['{}%'.format(i*10) for i in range(0,11)]
y_ticklabels.reverse()
ax.set_yticklabels(y_ticklabels)
ax.legend(framealpha=1.)
ax.set_title('{}-{} Sea Ice Age Composition (Week {}/52)'.format(min(x_axis_min_range), max(x_axis_min_range), week))
fig.text(0.59, 0.02, 'EASE-Grid Sea Ice Age from Tschudi et al. (2016): dx.doi.org/10.5067/PFSVFZA9Y85G')
fig.tight_layout()
plt.show()