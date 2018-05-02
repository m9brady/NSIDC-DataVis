import os
import sys
from ConfigParser import ConfigParser, NoSectionError, NoOptionError
from datetime import datetime, timedelta
from shutil import copyfileobj
from urllib2 import urlopen, URLError

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

LABEL_FONTPARAMS = {'family': 'sans-serif',
                    'color': 'black',
                    'weight': 'normal',
                    'size': '14'}
TITLE_FONTPARAMS = {'family': 'sans-serif',
                    'color': 'black',
                    'weight': 'normal',
                    'size': '18'}


def parse_cfg(cfgfile):
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


def download_daily_data(download_dir, hemisphere='N'):
    '''
    Required:   <download_dir>    directory to store the data files
    Optional:   <hemisphere>      one of 'N'/'North' or 'S'/'South', default is 'N'
    '''
    today = datetime.today()
    if hemisphere.upper() in ['SOUTH', 'S']:
        url = 'ftp://sidads.colorado.edu/DATASETS/NOAA/G02135/south/daily/data/'
    else:
        url = 'ftp://sidads.colorado.edu/DATASETS/NOAA/G02135/north/daily/data/'
    try:
        ftp_listing = urlopen(url).read().splitlines()
    except URLError:
        raise URLError("Cannot connect to NSIDC FTP url: {}".format(url))
    data_files = [f.split()[-1] for f in ftp_listing[:4] if f.endswith('.csv')]
    #  climatology file (.csv)
    try:
        climatology = [url+"/"+d for d in data_files if 'climatology' in d][0]
    except IndexError:
        raise Exception("Cannot locate climatology dataset at FTP: {}".format(url))
    target_climo = os.path.join(download_dir, os.path.basename(climatology))
    you_gotta_download = False
    if not os.path.isfile(target_climo):
        you_gotta_download = True
    else:
        # Re-download the file if it is >=12 hours older relative to "today" at start of script execution
        modtime = datetime.fromtimestamp(os.path.getmtime(target_climo))
        if modtime <= today - timedelta(0.5):
            you_gotta_download = True
    if you_gotta_download: # then go get it!
        print "Retrieving {}-hemisphere climatology data from NSIDC FTP...".format(hemisphere)
        remote_data = urlopen(climatology)
        with open(target_climo, 'wb') as local_data:
            copyfileobj(remote_data, local_data)
        remote_data.close()
    else:
        print "Existing local climatology files for {}-hemisphere are fresh enough...".format(hemisphere)
    # daily data file (.csv)
    try:
        daily = [url+"/"+d for d in data_files if 'daily' in d][0]
    except IndexError:
        raise Exception("Cannot locate daily dataset at FTP: {}".format(url))
    target_daily = os.path.join(download_dir, os.path.basename(daily))
    you_gotta_download = False
    if not os.path.isfile(target_daily):
        you_gotta_download = True
    else:
        # Re-download the file if it is >=12 hours older relative to current day at start of script execution
        modtime = datetime.fromtimestamp(os.path.getmtime(target_daily))
        if modtime <= today - timedelta(0.5):
            you_gotta_download = True
    if you_gotta_download: # then download!
        print "Retrieving {}-hemisphere daily data from NSIDC FTP...".format(hemisphere)
        remote_data = urlopen(daily)
        with open(target_daily, 'wb') as local_data:
            copyfileobj(remote_data, local_data)
        remote_data.close()
    else:
        print "Existing local daily files for {}-hemisphere are fresh enough...".format(hemisphere)
    # return references to data files for later use in plotting
    return target_daily, target_climo


def prep_dataframes(daily, climo):
    '''
    Due to some issues with data gaps, it is necessary to do some interpolation and data-sanitizing before plotting.
    Future work can also include some smoothing
    '''
    df_daily = pd.read_csv(daily, skiprows=2, usecols=[0, 1, 2, 3, 4],
                           names=['Year', 'Month', 'Day', 'Extent', 'Missing'],
                           dtype={'Year':'uint32', 'Month':'uint32', 'Day':'uint32', 'Extent':float, 'Missing':float})
    # Add flag-field for whether the data has been interpolated
    df_daily['Interpolated'] = False
    # Replace the numerical index with a datetime-based one
    current_index = pd.DatetimeIndex(data=pd.to_datetime(df_daily.Year*10000+df_daily.Month*100+df_daily.Day, format='%Y%m%d'))
    df_daily.index = current_index
    df_daily = df_daily.asfreq('1D') # Expands time series and introduces NaN's, thus changing a few column dtypes
    # Fill the YYYY-MM-DD column NaN data using the datetime64[ns] index
    df_daily.loc[pd.isnull(df_daily['Year']), 'Year'] = df_daily.loc[pd.isnull(df_daily['Year'])].index.to_series().apply(lambda x: int(x.strftime('%Y')))
    df_daily.loc[pd.isnull(df_daily['Month']), 'Month'] = df_daily.loc[pd.isnull(df_daily['Month'])].index.to_series().apply(lambda x: int(x.strftime('%m')))
    df_daily.loc[pd.isnull(df_daily['Day']), 'Day'] = df_daily.loc[pd.isnull(df_daily['Day'])].index.to_series().apply(lambda x: int(x.strftime('%d')))
    # Interpolate the ice extent NaN data linearly (never seems to be more than 1-day lag)
    df_daily['Extent'] = df_daily['Extent'].interpolate(method='linear')
    df_daily.loc[pd.isnull(df_daily['Missing']), 'Interpolated'] = True
    # Downcast YYYY-MM-DD columns to reduce mem footprint
    df_daily[['Year', 'Month', 'Day']] = df_daily[['Year', 'Month', 'Day']].apply(pd.to_numeric, downcast='unsigned')
    # Drop Feb 29 from daily because it is the worst
    mask = np.logical_and(df_daily['Month'] == 2, df_daily['Day'] == 29) # pylint: disable=E1101
    df_daily = df_daily[~mask]
    # climo data might be slightly flawed now due to the lack of including interpolated source data....
    df_climo = pd.read_csv(climo, skiprows=2,
                           names=['DOY', 'Average_Extent', 'STD', '10th', '25th', '50th', '75th', '90th'],
                           dtype={'DOY':'uint16', 'Average_Extent':float, 'STD':float, '10th':float,
                                  '25th':float, '50th':float, '75th':float, '90th':float})
    # Drop Feb 29 from climo because it is the worst
    df_climo = df_climo.loc[df_climo.DOY != 60]
    df_climo.loc[:, 'DOY'] = np.linspace(1, 365, 365, dtype='uint16')
    return df_daily, df_climo


def prep_global_dataframes(n_d, n_c, s_d, s_c):
    df_daily = pd.DataFrame(data={'Year': n_d.Year,
                                  'Month': n_d.Month,
                                  'Day': n_d.Day,
                                  'Extent': n_d.Extent + s_d.Extent,
                                  'Missing': n_d.Missing + s_d.Missing,
                                  'Interpolated': n_d.Interpolated & s_d.Interpolated
                                 },
                            columns=['Year', 'Month', 'Day', 'Extent',
                                     'Missing', 'Interpolated'])
    df_daily['Interpolated'] = df_daily['Interpolated'].astype(bool)
    df_climo = pd.DataFrame(data={'DOY': n_c.DOY,
                                  'Average_Extent':n_c.Average_Extent + s_c.Average_Extent,
                                  'STD':(n_c.STD + s_c.STD) / 2., 'p10':n_c['10th'] + s_c['10th'],
                                  'p25':n_c['25th'] + s_c['25th'], 'p50':n_c['50th'] + s_c['50th'],
                                  'p75':n_c['75th'] + s_c['75th'], 'p90':n_c['90th'] + s_c['90th']
                                 },
                            columns=['DOY', 'Average_Extent', 'STD', 'p10',
                                     'p25', 'p50', 'p75', 'p90'])
    return df_daily, df_climo


def plot_timeseries(daily_df, climo_df, aoi='Arctic', sigma=2.):
    '''
    daily_df --> sanitized pandas dataframe of daily ice extent
    climo_df --> sanitized pandas dataframe of daily climatological means (1981-2010)
    aoi --> which hemisphere to plot ['Arctic', 'Antarctic', 'Global']
    sigma --> how many standard deviations to include in the shaded climatology range
    '''
    fig, ax = plt.subplots(figsize=(12, 7))
    # To overlay the years rather than plot sequentially, we can force the data to all be in the same common year
    common_index = pd.date_range(start=datetime(2010, 1, 1), end=datetime(2010, 12, 31), freq='1D')
    min_year = daily_df.Year.min()
    max_year = daily_df.Year.max()
    plt.title('Daily {0} Sea Ice Extent {1}-{2}'.format(aoi.capitalize(), min_year, max_year), fontdict=TITLE_FONTPARAMS)
    for year in daily_df.Year.unique():
        lineweight = 0.8
        linetype = 'solid'
        # optional highlight extreme years
        if year in [2012, 2016]:
            lineweight = 3.0
        if year == max_year:
            lineweight = 3.0
            linetype = 'dashed'
        subframe = daily_df.loc[daily_df.Year == year]
        if subframe.shape[0] < 365:
            expanded_index = pd.date_range(start=datetime(year, 1, 1), end=datetime(year, 12, 31), freq='1D')
            subframe = subframe.reindex(expanded_index)
        subframe.index = common_index
        # experimenting with pandas rolling means
        extent_rolling = subframe['Extent'].rolling(min_periods=1, window=5, center=False).mean()
        extent_rolling.plot(ax=ax, label=year, linewidth=lineweight, linestyle=linetype, zorder=2)
        #subframe.plot(ax=ax, x=common_index, y='Extent', label=year, linewidth=lineweight, linestyle=linetype, zorder=2)

    # Fill in the shaded plot to show which years fall within <sigma> standard deviations from the 1981-2010 mean
    climo_df.index = common_index
    ax.fill_between(common_index,
                    (climo_df.Average_Extent - sigma * climo_df.STD),
                    (climo_df.Average_Extent + sigma * climo_df.STD),
                    facecolor='grey', alpha=0.4, zorder=2,
                    label='$\pm${}$\sigma$ range'.format(int(sigma))) # pylint: disable=W1401
    # Draw a dotted line showing the actual 1981-2010 mean ice extent
    climo_df.plot(ax=ax, x=common_index, y='Average_Extent', label='1981-2010 Mean',
                  linewidth=1.5, linestyle='dotted', color='black', zorder=3)
    # Figure formatting
    ax.grid(which='major', axis='both', color='#ECECEC', linestyle='solid', linewidth=1., zorder=2)
    xticks = [d for d in common_index if d.day == 1]
    xlabels = [datetime(2010, m, 1).strftime('%b') for m in range(1, 13)]
    ax.set_xlim(xticks[0], xticks[-1]+timedelta(30))
    ax.set_xticks(xticks)
    ax.set_xticklabels(xlabels)
    ax.set_xlabel('Month', fontdict=LABEL_FONTPARAMS)
    ax.set_ylabel('Extent (million square kilometers)', fontdict=LABEL_FONTPARAMS)
    ax.minorticks_off()
    legend_cols = int(np.ceil((daily_df.Year.unique().shape[0] + 1) / 10.))
    ax.legend(ncol=legend_cols, loc='best', labelspacing=0.3, columnspacing=0.5)
    fig.tight_layout()
    fig.text(0.010, 0.016, "Source: NSIDC Sea Ice Index v3.0 daily data files (doi:10.7265/N5736NV7)")
    latest_row = subframe.loc[~pd.isnull(subframe.Extent)].iloc[-1] # pylint: disable=E1130
    y, m, d = list(latest_row[:3].astype(int))
    latest_date = datetime(y, m, d).strftime("%Y-%m-%d")
    fig.text(0.645, 0.016, "Latest daily value ({}) may be subject to change".format(latest_date))
    return fig


def main(cfg):
    data_dir = cfg['data_dir'] + '/daily'
    plot_dir = cfg['plot_dir']

    # prep
    if not os.path.exists(data_dir):
        os.makedirs(data_dir)
    N_daily_csv, N_daily_climo = download_daily_data(data_dir, hemisphere='N')
    S_daily_csv, S_daily_climo = download_daily_data(data_dir, hemisphere='S')
    N_daily_df, N_climo_df = prep_dataframes(N_daily_csv, N_daily_climo)
    S_daily_df, S_climo_df = prep_dataframes(S_daily_csv, S_daily_climo)
    G_daily_df, G_climo_df = prep_global_dataframes(N_daily_df, N_climo_df, S_daily_df, S_climo_df)

    # plot
    if not os.path.exists(plot_dir): 
        os.makedirs(plot_dir)
    n_fig = plot_timeseries(N_daily_df, N_climo_df, 'Arctic')
    s_fig = plot_timeseries(S_daily_df, S_climo_df, 'Antarctic')
    g_fig = plot_timeseries(G_daily_df, G_climo_df, 'Global')

    print "Saving plots to", os.path.abspath(plot_dir)
    n_fig.savefig(os.path.join(plot_dir, 'NSIDC_DailyIceIndex_N-Hemisphere.png'))
    s_fig.savefig(os.path.join(plot_dir, 'NSIDC_DailyIceIndex_S-Hemisphere.png'))
    g_fig.savefig(os.path.join(plot_dir, 'NSIDC_DailyIceIndex_Global.png'))
    plt.close('all')

    #TODO: Produce console printouts of daily decline trends, etc.

if __name__ == "__main__":
    os.chdir(os.path.dirname(os.path.dirname(__file__)))
    # python DailyIceIndexPlotter.py config.cfg
    if len(sys.argv) == 2:
        CFGFILE = sys.argv[1]
        print "Attempting to parse config file: ", CFGFILE
        CFG = parse_cfg(CFGFILE)
    # python DailyIceIndexPlotter.py
    else:
        print "Processing with default config options"
        CFG = {'data_dir': './data',
               'plot_dir': './plots'}
    main(CFG)
