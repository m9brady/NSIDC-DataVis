import os
import sys
import warnings
# suppress the FutureWarning about pandas.core.datetools
warnings.simplefilter(action='ignore', category=FutureWarning)
from ConfigParser import ConfigParser, NoOptionError, NoSectionError
from datetime import datetime
from glob import glob
from shutil import copyfileobj
from urllib2 import URLError, urlopen

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import statsmodels.api as sm
from dateutil.relativedelta import relativedelta
from matplotlib.offsetbox import AnchoredText


# constants to change
PLOT_MONTH = int(datetime.today().strftime('%m'))


def parse_cfg(cfgfile):
    parser = ConfigParser()
    parser.read(cfgfile)
    try:
        creds = {'data_dir':parser.get('NSIDC_Tools', 'data_dir'),
                 'plot_dir':parser.get('NSIDC_Tools', 'plot_dir')}
    except NoSectionError:
        raise Exception('Error reading config file. Ensure it has the proper Section heading --> [NSIDC_Tools]')
    except NoOptionError:
        raise Exception('Error reading config file. Ensure sure it has the proper Options set --> data_dir, plot_dir')
    return creds


def download_monthly_data(download_dir, hemisphere='N'):
    if not os.path.isdir(download_dir): os.makedirs(download_dir)
    today = datetime.today()
    if not os.path.isdir(download_dir): os.makedirs(download_dir)
    if hemisphere.upper() in ['SOUTH', 'S']:
        csv_url = 'ftp://sidads.colorado.edu/DATASETS/NOAA/G02135/south/monthly/data/'
        #shp_url = 'ftp://sidads.colorado.edu/DATASETS/NOAA/G02135/south/monthly/shapefiles/shp_extent/'
    else:
        csv_url = 'ftp://sidads.colorado.edu/DATASETS/NOAA/G02135/north/monthly/data/'
        #shp_url = 'ftp://sidads.colorado.edu/DATASETS/NOAA/G02135/north/monthly/shapefiles/shp_extent/'
    try:
        csv_listing = urlopen(csv_url).read().splitlines()
    except URLError:
        print "Cannot connect to NSIDC FTP, check your internet settings"
        return None
    csv_files = [csv_url+f.split()[-1] for f in csv_listing[2:] if f.endswith('.csv')]
    if len(csv_files) == 0:
        print "No files found at remote FTP:", csv_url
        return None
    for src_file in csv_files:
        curr_month = datetime(1900, int(os.path.basename(src_file).split("_")[1]), 1)
        dst_file = os.path.join(download_dir, os.path.basename(src_file))
        you_gotta_download = False
        if not os.path.isfile(dst_file):
            you_gotta_download = True
        else:
            # Re-download the file if it is >15 days older relative to current day at start of script execution
            modtime = datetime.fromtimestamp(os.path.getmtime(dst_file))
            if modtime <= today - relativedelta(days=15):
                you_gotta_download = True
        if you_gotta_download: # then go get it!
            print "Retrieving {}-hemisphere monthly data ({}) from NSIDC FTP...".format(hemisphere, curr_month.strftime("%b"))
            remote_data = urlopen(src_file)
            with open(dst_file, 'wb') as local_data:
                copyfileobj(remote_data, local_data)
            remote_data.close()
        else:
            print "Existing local monthly ({}) dataset for {}-hemisphere is fresh enough...".format(curr_month.strftime("%b"), hemisphere)
    return None


def prep_dframe(csv_dir, hemisphere='N'):
    '''
    Load the source .csv files into a pandas dataframe
    Parse NSIDC dataset version number
    Remove extra whitespace from column names
    Replace -9999 values (missing data) with np.NaN
    '''
    csv_files = glob(os.path.join(csv_dir, '{}_??_extent_*.csv'.format(hemisphere)))
    if len(csv_files) == 0:
        print "Cannot find any csv files in dir:", csv_dir
        return None, None
    data_version = os.path.splitext(os.path.basename(csv_files[0]))[0].split("v")[-1]
    dframe = pd.DataFrame()
    for csv in csv_files:
        dframe = dframe.append(pd.read_csv(csv))
    dframe.columns = ['year', 'month', 'data_type', 'region', 'extent', 'area']
    # default sort by year ascending, then month ascending
    dframe = dframe.sort_values(['year', 'month'])
    # replace -9999 values (missing data) with NaN
    dframe.replace(-9999, np.nan, inplace=True)
    # reset the integer index to ensure key uniqueness
    dframe.reset_index(drop=True, inplace=True)
    return dframe, data_version


def plot_dframe(dframe, month=None, version='3.0', summary=False):
    # rudimentary parameter checking
    if month is not None and month in range(1,13):
        # subset input dataframe to the month of interest
        subframe = dframe.loc[dframe['month'] == month]   
    else:
        raise ValueError("Must supply an integer argument for month (e.g. 12 for December)")
    # define fontdicts
    title_font = {'fontsize': 20, 'fontweight': 'regular'}
    axes_label_font = {'fontsize': 16, 'fontweight': 'regular'}
    # determine trend using ordinary least-squares regression
    y = subframe.extent
    X = subframe.year
    X = sm.add_constant(X)
    est = sm.OLS(y, X, missing='drop') # ignore NaN's by setting 'missing' arg
    est = est.fit()
    trend = est.predict(X)
    # add the trend data to the subset of dframe
    subframe = subframe.assign(trend=pd.Series(data=trend, index=subframe.index))
    # produce the figure 
    screen_dpi = 100. # default windows 7 screen dpi is 96
    fig, ax = plt.subplots(figsize=(10, 8), facecolor='white', dpi=screen_dpi)
    # if there are any NaN values in subframe, we have to plot twice
    if len(subframe.loc[pd.isnull(subframe['extent'])]) > 0:
        # we plot the NaN (black dashed line)
        subframe.dropna().plot(x='year', y='extent', ax=ax, legend=True, style='k--', dashes=[2.,1.5], linewidth=2.)
    # plot the non-NaN time series (black solid line)
    subframe.plot(x='year', y='extent', ax=ax, legend=False, style='k-', linewidth=2.5)
    # plot the trendline (blue solid line)
    subframe.plot(x='year', y='trend', ax=ax, legend=False, style='b-', linewidth=2.5)
    # possible bug in matplotlib where the legend acts strangely, despite setting as False for some plots
    if len(subframe.loc[pd.isnull(subframe['extent'])]) > 0:
        ax.legend(['Missing Data'], loc='upper right', fontsize='large', edgecolor='k', fancybox=False, shadow=False)
    # pull month string from int using datetime
    monthname = datetime(1900, int(month), 1).strftime("%B")
    # figure formatting
    minyear = subframe['year'].min()
    maxyear = subframe['year'].max()
    if subframe.iloc[0]['region'].strip() == 'N':
        region = 'Arctic'
    elif subframe.iloc[0]['region'].strip() == 'S':
        region = 'Antarctic'
    else:
        region = 'Global'
    ax.set_title("Average Monthly {} Sea Ice Extent\n{} {} - {}".format(region, monthname, minyear, maxyear), title_font)
    ax.set_ylabel("Extent (million square kilometers)", axes_label_font)
    ax.set_xlabel("Year", axes_label_font)
    ax.set_xbound(lower=minyear-1, upper=maxyear+1)
    ax.set_ybound(lower=subframe['extent'].min()-0.5, upper=subframe['extent'].max()+0.5)
    ax.grid(b=True, color='#D8D8D8', which='major', alpha=0.5)
    # major ticks will be in increments of 4
    ax.set_xticks(np.arange(minyear, maxyear, 4), minor=False)
    # minor ticks will be for every year (add 1 to the end to retain the most recent year)
    ax.set_xticks(np.arange(minyear, maxyear+1, 1), minor=True)
    fig.text(0.010, 0.016, "Source: NSIDC Sea Ice Index v{} monthly data files (doi:10.7265/N5736NV7)".format(version))
    # some fun statistics and an estimate of what the NSIDC Monthly Sea Ice News and Analysis will be
    if summary:
        print est.summary2()
    # get the climatological (30-year) mean extent for 1981-2010
    climo_1981_2010 = subframe[subframe.loc[subframe.year == 1981].index.tolist()[0]:subframe.loc[subframe.year == 2011].index.tolist()[0]].extent.mean()
    # linear rate of change in million sq km
    change_rate = est.params.year
    frame_text = AnchoredText("Linear rate of change: {} km$^2$ per year\nPercent per decade (relative to 1981-2010 mean): {}%".format(int(round(change_rate * 1000000, -2)),round(change_rate / climo_1981_2010 * 10 * 100, 2)),
                              loc=3, prop=dict(size=11))
    ax.add_artist(frame_text)
    # console prints
    print "\nEstimated stats for the Monthly {} Sea Ice News\n-------------------------------------".format(region.capitalize())
    print "Linear rate of change: {} square kilometers".format(int(round(change_rate * 1000000, -2)))
    print "Percent per decade (relative to 1981-2010 climo): {}%".format(round(change_rate / climo_1981_2010 * 10 * 100, 2))
    return fig


def plot_anomaly(dframe, month=None, version='3.0', summary=False):
    # rudimentary parameter checking
    if month is not None and month in range(1,13):
        # subset input dataframe to the month of interest
        subframe = dframe.loc[dframe['month'] == month]
    else:
        raise ValueError("Must supply an integer argument for month (e.g. 12 for December)")
    # define fontdicts
    title_font = {'fontsize': 20, 'fontweight': 'regular'}
    axes_label_font = {'fontsize': 16, 'fontweight': 'regular'}
    # get the climatological (30-year) mean extent for 1981-2010
    climo_1981_2010 = subframe[subframe.loc[subframe.year == 1981].index.tolist()[0]:subframe.loc[subframe.year == 2011].index.tolist()[0]].extent.mean()
    # calculate the anomalies relative to 30-year mean
    extent_anomaly = (subframe.extent - climo_1981_2010) / climo_1981_2010 * 100.
    extent_anomaly.name = 'extent_anomaly'
    subframe = pd.concat([subframe, extent_anomaly], axis=1)
    # determine trend using ordinary least-squares regression
    y = subframe.extent_anomaly
    X = subframe.year
    X = sm.add_constant(X)
    est = sm.OLS(y, X, missing='drop') # ignore NaN's by setting 'missing' arg
    est = est.fit()
    trend = est.predict(X)
    # fun stats
    pct_decade = est.params.year * 10
    pct_decade_std = est.bse.year * 10
    # append the trend data points to the subset of dframe
    subframe = subframe.assign(extent_anomaly_trend = pd.Series(data=trend, index=subframe.index))
    # produce the figure 
    screen_dpi = 100. # default windows 7 screen dpi is 96
    fig, ax = plt.subplots(figsize=(10, 8), facecolor='white', dpi=screen_dpi)
    if len(subframe.loc[pd.isnull(subframe['extent_anomaly'])]) > 0:
        # we plot with the NaN showing (black dashed line)
        subframe.dropna().plot(x='year', y='extent_anomaly', ax=ax, legend=True, style='k--', dashes=[2.,1.5], linewidth=2.)
    # plot the time series (black solid line with + markers)
    subframe.plot(x='year', y='extent_anomaly',  ax=ax, legend=False, style='k-P', markersize=8, linewidth=2.0)
    # plot the trendline (grey dashed line)
    subframe.plot(x='year', y='extent_anomaly_trend', ax=ax, legend=False, style='--', color='grey', dashes=[4,4], linewidth=1.5)
    # possible bug in matplotlib where the legend acts strangely, despite setting as False for some plots
    if len(subframe.loc[pd.isnull(subframe['extent'])]) > 0:
        ax.legend(['Missing Data'], loc='upper right', fontsize='large', edgecolor='k', fancybox=False, shadow=False)
    # pull month string from int using datetime
    monthname = datetime(1900, int(month), 1).strftime("%B")
    minyear = subframe['year'].min()
    maxyear = subframe['year'].max()
    maxanom = max([abs(subframe['extent_anomaly'].min()), abs(subframe['extent_anomaly'].max())])
    maxanom = int(np.ceil(maxanom/5.)*5.)
    if subframe.iloc[0]['region'].strip() == 'N':
        region = 'Arctic'
    elif subframe.iloc[0]['region'].strip() == 'S':
        region = 'Antarctic'
    else:
        region = 'Global'
    # figure formatting
    ax.set_title("Monthly {} Sea Ice Extent Anomalies\n{} {} - {}".format(region, monthname, minyear, maxyear), title_font)
    ax.set_ylabel("Anomaly (% difference from 30-year mean)", axes_label_font)
    ax.set_xlabel("Year", axes_label_font)
    ax.set_xbound(lower=minyear-1, upper=maxyear+1)
    ax.set_ybound(lower=-maxanom, upper=maxanom)
    ax.grid(b=True, color='#D8D8D8', which='major', alpha=0.5)
    # major ticks will be in increments of 4
    ax.set_xticks(np.arange(minyear, maxyear, 4), minor=False)
    # minor ticks will be for every year (add 1 to the end to retain the most recent year)
    ax.set_xticks(np.arange(minyear, maxyear+1, 1), minor=True)
    frame_text = AnchoredText(u"1981-2010 mean ice extent: {} million km$^2$\nLinear rate of change: {}% (\u00B1{}%) per decade".format(
                              round(climo_1981_2010, 1), round(pct_decade, 1), round(pct_decade_std, 1)), loc=3, prop=dict(size=11))
    ax.add_artist(frame_text)
    fig.text(0.010, 0.016, "Source: NSIDC Sea Ice Index v{} monthly data files (doi:10.7265/N5736NV7)".format(version))
    if summary:
        print est.summary2()
    print "\nEstimated anomaly stats for the Monthly {} Sea Ice News\n-------------------------------------".format(region.capitalize())
    print u"Anomaly slope = {} \u00B1{} percent per decade".format(round(pct_decade,1), round(pct_decade_std,1))
    return fig


def main(cfg):
    data_dir = cfg['data_dir'] + '/monthly'
    plot_dir = cfg['plot_dir']

    # Get data from NSIDC FTP
    download_monthly_data(data_dir, 'N')
    download_monthly_data(data_dir, 'S')

    # Prep the pandas dataframes
    n_dframe, n_version = prep_dframe(data_dir, 'N')
    s_dframe, s_version = prep_dframe(data_dir, 'S')

    # Generate figure object instances
    n_monthly_fig = plot_dframe(n_dframe, month=PLOT_MONTH, version=n_version)
    s_monthly_fig = plot_dframe(s_dframe, month=PLOT_MONTH, version=s_version)
    n_anomaly_fig = plot_anomaly(n_dframe, month=PLOT_MONTH, version=n_version)
    s_anomaly_fig = plot_anomaly(s_dframe, month=PLOT_MONTH, version=s_version)

    # Save figures to local PNG
    print "\nSaving plots to", os.path.abspath(plot_dir)
    n_monthly_fig.savefig(os.path.join(plot_dir, 'NSIDC_MonthlyIceIndex_N-Hemisphere.png'))
    s_monthly_fig.savefig(os.path.join(plot_dir, 'NSIDC_MonthlyIceIndex_S-Hemisphere.png'))
    n_anomaly_fig.savefig(os.path.join(plot_dir, 'NSIDC_MonthlyIceAnomaly_N-Hemisphere.png'))
    s_anomaly_fig.savefig(os.path.join(plot_dir, 'NSIDC_MonthlyIceAnomaly_S-Hemisphere.png'))
    plt.close('all')


if __name__ == "__main__":
    os.chdir(os.path.dirname(os.path.dirname(__file__)))
    # python MonthlyIceIndexPlotter.py config.cfg
    if len(sys.argv) == 2:
        CFGFILE = sys.argv[1]
        print "Attempting to parse config file: ", CFGFILE
        CFG = parse_cfg(CFGFILE)
    # python MonthlyIceIndexPlotter.py
    else:
        print "Processing with default config options"
        CFG = {'data_dir': './data',
               'plot_dir': './plots'}
    main(CFG)
