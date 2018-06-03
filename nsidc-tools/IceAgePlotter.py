import os
import sys
from ConfigParser import ConfigParser, NoOptionError, NoSectionError
from glob import glob
from datetime import datetime

# suppress the UserWarning about _nolegend_ plotting with basemap
import warnings
warnings.simplefilter(action='ignore', category=UserWarning)
# suppress the many, many MatplotlibDeprecationWarnings from basemap
from matplotlib import warnings as mpl_warn
mpl_warn.default_action = 'ignore'

import matplotlib.colors as colors
import matplotlib.patches as patches
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import requests
from bs4 import BeautifulSoup
from mpl_toolkits.basemap import Basemap

# constants to change
AGE_CONST = 'nsidc0611_seaice_age_v3'
AGE_FILESIZE = 510
GEO_BOUNDS = [65.0, 90.0, -180.0, 180.0]
WEEK = int(datetime.today().isocalendar()[1])


def parse_cfg(cfgfile):
    '''
    Parse the required `data_dir` and `plot_dir` variables from a config file
    -----------------------
    `cfgfile` (str): path to config file, formatted for python's ConfigParser
    '''
    parser = ConfigParser()
    parser.read(cfgfile)
    try:
        cfg = {'data_dir':parser.get('NSIDC_Tools', 'data_dir'),
               'plot_dir':parser.get('NSIDC_Tools', 'plot_dir')}
    except NoSectionError:
        raise Exception('Error reading config file. Ensure it has the proper Section heading --> [NSIDC_Tools]')
    except NoOptionError:
        raise Exception('Error reading config file. Ensure sure it has the proper Options set --> data_dir, plot_dir')
    return cfg

def download_bin_data(download_dir):
    '''
    Function to grab all available ice age binary data from NSIDC DAAC
    User must have the proper .netrc configuration for NASA Earthdata access
    -----------------------
    `download_dir` (str): directory to store the data files
    '''
    if not os.path.exists(download_dir): 
        os.makedirs(download_dir)
    # Establish how many data files are available locally
    data_local = os.path.join(download_dir, 'data')
    if not os.path.isdir(data_local):
        os.makedirs(data_local)
    local_listing = sorted(glob(os.path.join(data_local, '*.bin')))
    url = 'https://daacdata.apps.nsidc.org/pub/DATASETS/' + AGE_CONST
    try:
        r = requests.get(url)
    except requests.exceptions.ConnectionError:
        raise Exception("Error - Cannot connect to NASA EarthData catalog. Check your internet settings")
    if r.ok:
        rsoup = BeautifulSoup(r.text, 'html.parser')
        listing = [row.findAll('a')[1].text for row in rsoup.findAll('tr')]
        try:
            lat_remote = url + '/' + [l for l in listing if 'latitude' in l][0]
        except IndexError:
            raise Exception("Warning - Cannot find Na12500-CF_latitude.dat on remote server")
        lat_local = os.path.join(download_dir, os.path.basename(lat_remote))
        if not os.path.isfile(lat_local):
            lat = requests.get(lat_remote)
            if lat.ok:
                with open(lat_local, 'wb') as latbin:
                    latbin.write(lat.content)
            else:
                raise Exception("{} {}".format(lat.status_code, lat.reason))
            lat.close()
        try:
            lon_remote = url + '/' + [l for l in listing if 'longitude' in l][0]
        except IndexError:
            raise Exception("Warning - Cannot find Na12500-CF_longitude.dat on remote server")
        lon_local = os.path.join(download_dir, os.path.basename(lon_remote))
        if not os.path.isfile(lon_local):
            lon = requests.get(lon_remote)
            if lon.ok:
                with open(lon_local, 'wb') as lonbin:
                    lonbin.write(lon.content)
            else:
                raise Exception("{} {}".format(lon.status_code, lon.reason))
            lon.close()
    r.close()
    del rsoup
    # traverse data subfolder on remote server
    url_list = []
    try:
        data_remote = url + '/' + [l for l in listing if 'data' in l][0]
    except IndexError:
        raise Exception("Error - Cannot find 'data' subfolder in remote directory listing")
    d = requests.get(data_remote)
    if d.ok:
        dsoup = BeautifulSoup(d.text, 'html.parser')
        year_list = [row.findAll('a')[1].text for row in dsoup.findAll('tr') if len(row.findAll('a')[1].text) == 5]
        print "Searching for bin data on NASA EarthData catalogue..."
        for year in year_list:
            curryear_remote = data_remote + year
            y = requests.get(curryear_remote)
            if y.ok:
                ysoup = BeautifulSoup(y.text, 'html.parser')
                bin_list = [row.findAll('a')[1].text for row in ysoup.findAll('tr') if  '.bin' in row.findAll('a')[1].text]
                for binfile in bin_list:
                    url_list.append(curryear_remote + binfile)
            else:
                raise Exception("{} {}".format(y.status_code, y.reason))
            y.close()
            del ysoup
    else:
        raise Exception("{} {}".format(d.status_code, d.reason))
    d.close()
    del dsoup
    print "{} binary files found on remote server".format(len(url_list))
    print "{} local files present".format(len(local_listing))
    if len(url_list) > len(local_listing):
        for local_file in local_listing:
            bname = os.path.basename(local_file)
            year = bname.split(".")[3]
            burl = os.path.dirname(os.path.dirname(url_list[0]))
            url_list.remove(burl+"/{}/{}".format(year,bname))
        choice = raw_input("Total download size for new ice age data is ~{:.2f}MB, proceed to download? (y/[n])".format(len(url_list)*AGE_FILESIZE/1000.))
        if choice == 'y':
            for remote_file in url_list:
                local_file = os.path.join(data_local, os.path.basename(remote_file))
                if not os.path.isfile(local_file):
                    f = requests.get(remote_file)
                    if f.ok:
                        with open(local_file, 'wb') as localbin:
                            localbin.write(f.content)
                        print "Downloaded {} to {}".format(os.path.basename(local_file), os.path.abspath(data_local))
                    else:
                        raise Exception("{} {}".format(f.status_code, f.reason))
                    f.close()
                else:
                    continue
        else:
            return
    else:
        return


def plot_stacked_fill(week=35, bounds=[65.0, 90.0, -180.0, 180.0], start_year=1984, end_year=2018, data_dir='./data/'+AGE_CONST):
    '''
    Load binary sea ice datasets into memory, and plot the stacked fill charts
    for the selected year range and geographic boundaries
    -------------------------------
    `week` (int): the week number to plot with. `Default 35`
    `bounds` (list): 4-element list of floats [minLat, maxLat, minLon, maxLon]. `Default [65.0, 90.0, -180.0, 180.0]`
    `start_year` (int): year to start the plot. `Default 1984`
    `end_year` (int): year to end the plot. `Default 2018`
    `data_dir` (str): path to where source binary data is kept. `Default ./data/nsidc0611_seaice_age_v3`

    TODO: Target for refactoring, split up the dataframe prep into a different function
    '''
    lats = os.path.join(data_dir, 'Na12500-CF_latitude.dat')
    if not os.path.isfile(lats):
        raise Exception("Error: Cannot locate 'Na12500-CF_latitude.dat'")
    lat_arr = np.fromfile(lats, dtype='float32')
    lons = os.path.join(data_dir, 'Na12500-CF_longitude.dat')
    if not os.path.isfile(lons):
        raise Exception("Error: Cannot locate 'Na12500-CF_longitude.dat'")
    lon_arr = np.fromfile(lons, dtype='float32')
    age_files = sorted(glob(os.path.join(data_dir, 'data', 'iceage.*.bin')))
    if len(age_files) == 0:
        raise Exception("Error: No ice age binary data found")
    # filter between start_year and end_year
    years = [str(y) for y in range(start_year, end_year+1)]
    age_files = [f for f in age_files if os.path.basename(f).split(".")[3] in years]
    # pre-formatted dataframe columns
    frame_cols = np.empty((0,), dtype=[('YEAR','uint16'),
                            ('WEEK_NUM','uint8'), 
                            ('FYI','float'), 
                            ('SYI','float'), 
                            ('MYI_3','float'), 
                            ('MYI_4','float'), 
                            ('MYI_5+','float')])#,
                            #('OW','float')])
    # One dataframe for pixel counts, another dataframe for cumulative percentages
    pixcount_frame = pd.DataFrame(frame_cols)
    cumulativepct_frame = pd.DataFrame(frame_cols)
    # generate a mask if we want to restrict our data to a certain geographic boundary
    if bounds:
        mask = (lat_arr >= bounds[0]) & \
               (lat_arr <= bounds[1]) & \
               (lon_arr >= bounds[2]) & \
               (lon_arr <= bounds[3])
    # otherwise, keep all 521,284 pixels
    else:
        mask = np.full(shape=lat_arr.shape, fill_value=True)
    # Iterate through age files, apply mask, extract pixel counts and cumulative percentages
    print "Loading ice age data into dataframes..."
    for idx, age_file in enumerate(age_files):
        year = int(os.path.basename(age_file).split(".")[3])
        week_num = int(os.path.basename(age_file).split(".")[4])
        file_arr = np.fromfile(age_file, dtype='uint8')
        age_arr = file_arr[mask]
        # pixel counts for each ice age class including open water
        fyi_count = len(age_arr[age_arr == 5])
        syi_count = len(age_arr[age_arr == 10])
        myi3_count = len(age_arr[age_arr == 15])
        myi4_count = len(age_arr[age_arr == 20])
        myi5_count = len(age_arr[np.logical_and(age_arr >= 25, age_arr < 254)])
        #ow_count = len(age_arr[age_arr == 0])
        #coastline_count = len(age_arr[age_arr == 254])
        #land_count = len(age_arr[age_arr == 255])
        ice_count = fyi_count + syi_count + myi3_count + myi4_count + myi5_count
        #sea_count = ice_count + ow_count
        # percentages of all ocean-related class pixels (better-looking for arctic aoi)
        myi5_pct = myi5_count * 1. / ice_count
        myi4_pct = myi4_count * 1. / ice_count
        myi3_pct = myi3_count * 1. / ice_count
        syi_pct = syi_count * 1. / ice_count
        fyi_pct = fyi_count * 1. / ice_count
        #ow_pct = ow_count * 1. / sea_count
        # cumulative percentages for plotting cumulative plots
        myi5_cp = myi5_pct
        myi4_cp = myi5_cp + myi4_pct
        myi3_cp = myi4_cp + myi3_pct
        syi_cp = myi3_cp + syi_pct
        fyi_cp = syi_cp + fyi_pct
        #ow_cp = fyi_cp + ow_pct
        # insert into dataframes
        cumulativepct_frame.loc[idx] = [year, week_num, fyi_cp, syi_cp, myi3_cp, myi4_cp, myi5_cp]#, ow_cp]
        pixcount_frame.loc[idx] = [year, week_num, fyi_count, syi_count, myi3_count, myi4_count, myi5_count]#, ow_count]
        # clear some memory
        del file_arr
        del age_arr
    # downsample where possible
    cumulativepct_frame['YEAR'] = pd.to_numeric(cumulativepct_frame['YEAR'], downcast='integer')
    cumulativepct_frame['WEEK_NUM'] = pd.to_numeric(cumulativepct_frame['WEEK_NUM'], downcast='integer')
    for column in pixcount_frame.columns:
        pixcount_frame[column] = pd.to_numeric(pixcount_frame[column], downcast='integer')
    # setup the plots (2 axes side by side)
    print "Setting up plot figure..."
    fig = plt.figure(figsize=(15, 7))
    ax_chart = fig.add_subplot(121)
    ax_map = fig.add_subplot(122)
    ## AX_MAP STUFF
    # setup the subplot that will contain the EASE-grid map
    lon_range = lon_arr.reshape((722,722))
    lat_range = lat_arr.reshape((722,722))
    ll_lat = lat_range[721,0] 
    ll_lon = lon_range[721,0]
    ur_lat = lat_range[0,721] 
    ur_lon = lon_range[0,721]
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
    b.drawmeridians(np.arange(-180,180,20), color='0.25', linewidth=0.5, labels=[0,0,1,1])
    # convert lats/lons to map units
    x,y = b(lon_arr, lat_arr)
    # mask for the current area of interest
    aoi_label = "Area of Interest"
    # mask out the lat/lon data that is out of bounds
    x2 = x[mask]
    y2 = y[mask]
    # plot the boundary AOI with 55% transparency to show coastline/graticule underneath
    x2[x2 == 0] = np.nan
    y2[y2 == 0] = np.nan
    # this creates the transparent "AOI" view to show where the current bounds are
    b.plot(x2, y2, alpha=0.45, color='#EFAF40', label='_nolegend_')
    # manually set legend entry for area of interest
    pat = patches.Patch(color='#F5CF8C', label=aoi_label)
    # add legend to subplot in upper left corner
    ax_map.legend(handles=[pat], loc='upper left', framealpha=1., edgecolor='k')
    # grab the latest age array for plotting on the basemap
    map_file = sorted([f for f in age_files if str(week).zfill(2) in os.path.basename(f).split(".")[4]])[-1]
    map_arr = np.fromfile(map_file, dtype='uint8')
    f = map_arr.copy().astype(float)
    # pixels older than 5 years will be coloured the same as myi5
    f[np.logical_and(f <> 255, f > 30)] = 30
    # mask out the land and out-of-current-bounds pixels
    f[np.where(f == 255)] *= np.nan
    f[~mask] *= np.nan
    # add text to denote which age dataset is being displayed
    text_content = "Dataset shown: {}".format(os.path.basename(map_file))
    props = dict(boxstyle='round', facecolor='white', alpha=1.)
    ax_map.text(200000, 200000, text_content, bbox=props)
    # get unique pixel values, for setting up the colour mapping
    uni = np.unique(f[~np.isnan(f)]).tolist()
    # hex values for colormap
    clist = ['#FFFFFF', '#2400F5', '#00F6FF', '#15FF00', '#FFC700','#FF0000']
    ccmap = colors.ListedColormap(clist)
    cnorm = colors.BoundaryNorm(uni, ccmap.N)
    x_range, y_range = b(lon_range, lat_range)
    b.pcolormesh(x_range, y_range, f.reshape((722,722)), cmap=ccmap, norm=cnorm)
    ## AX_CHART STUFF
    # subset our dataframe to a specific week and starting year
    cpct_frame = cumulativepct_frame.loc[cumulativepct_frame['WEEK_NUM'] == week]
    cpct_frame = cpct_frame.loc[cpct_frame['YEAR'] >= start_year]
    for c in cpct_frame.columns[2:]:
        cpct_frame.plot('YEAR', c, ax=ax_chart, label='_nolegend_', color='k')
    # ColourMap LUT for the cumulative percentage plot
    cmap = {'OW':'#78B4FF', 'FYI':'#2400F5', 'SYI':'#00F6FF', 'MYI_3':'#15FF00', 'MYI_4':'#FFC700', 'MYI_5+':'#FF0000'}
    # filling the spaces between lines on the plot with the proper colours
    #ax_chart.fill_between(cpct_frame['YEAR'], cpct_frame['OW'], cpct_frame['FYI'], label='Open Water or <15% Ice', color=cmap['OW'])
    ax_chart.fill_between(cpct_frame['YEAR'], cpct_frame['FYI'], cpct_frame['SYI'], label='First-Year Ice', color=cmap['FYI'])
    ax_chart.fill_between(cpct_frame['YEAR'], cpct_frame['SYI'], cpct_frame['MYI_3'], label='Second-Year Ice', color=cmap['SYI'])
    ax_chart.fill_between(cpct_frame['YEAR'], cpct_frame['MYI_3'], cpct_frame['MYI_4'], label='Third-Year Ice', color=cmap['MYI_3'])
    ax_chart.fill_between(cpct_frame['YEAR'], cpct_frame['MYI_4'], cpct_frame['MYI_5+'], label='Fourth-Year Ice', color=cmap['MYI_4'])
    ax_chart.fill_between(cpct_frame['YEAR'], cpct_frame['MYI_5+'], 0, label='Fifth-Year Ice and Older', color=cmap['MYI_5+'])
    ax_chart.set_xlabel('Year')
    min_year = cpct_frame['YEAR'].min()
    max_year = cpct_frame['YEAR'].max()
    x_axis_maj_range = range(min_year, max_year+1, 2)
    x_axis_min_range = range(min_year, max_year+1)
    ax_chart.set_xticks(x_axis_maj_range)
    ax_chart.set_xticks(x_axis_min_range, minor=True)
    ax_chart.set_ylabel('Cumulative Percentage')
    ax_chart.set_xbound(min_year, max_year)
    ax_chart.set_ybound(0, 1)
    ax_chart.invert_yaxis()
    ax_chart.set_yticks([i / 10. for i in xrange(11)])
    y_ticklabels = ['{}%'.format(i * 10) for i in range(0,11)]
    y_ticklabels.reverse()
    ax_chart.set_yticklabels(y_ticklabels)
    ax_chart.legend(framealpha=1., edgecolor='k')
    ax_chart.set_title('{}-{} Sea Ice Age Composition (Week {}/52)'.format(min(x_axis_min_range), max(x_axis_min_range), week))
    # General Figure Formatting
    fig.text(0.59, 0.02, 'EASE-Grid Sea Ice Age from Tschudi et al. (2016): dx.doi.org/10.5067/PFSVFZA9Y85G')
    fig.tight_layout()
    return fig


def main(cfg):
    '''
    Call the other functions in this script
    '''
    data_dir = os.path.join(cfg['data_dir'], AGE_CONST)
    plot_dir = cfg['plot_dir']

    # prep
    download_bin_data(data_dir)

    # plot
    age_plot = plot_stacked_fill(week=WEEK, bounds=GEO_BOUNDS, data_dir=data_dir)

    # save
    print "Saving plot to", os.path.abspath(plot_dir)
    age_plot.savefig(os.path.join(plot_dir, 'NSIDC_WeeklySeaIceAge.png'))
    
    plt.close('all')
    return


if __name__ == "__main__":
    os.chdir(os.path.dirname(os.path.dirname(__file__)))
    # python nsidc-tools/MonthlyIceIndexPlotter.py config.cfg
    if len(sys.argv) == 2:
        CFG_FILE = sys.argv[1]
        print "Attempting to parse config file: ", CFG_FILE
        CFG = parse_cfg(CFG_FILE)
    # python nsidc-tools/MonthlyIceIndexPlotter.py
    else:
        print "Processing with default config options"
        CFG = {'data_dir': './data',
               'plot_dir': './plots'}
    main(CFG)
