# -*- coding: utf-8 -*-
import os
import tempfile
from ftplib import FTP, all_errors
from datetime import datetime 
import pandas as pd
import matplotlib.pyplot as plt
import statsmodels.api as sm

data_dir = os.path.join(tempfile.gettempdir(),'NSIDC_DailySeaIceIndex')
if not os.path.exists(data_dir): 
    print "Creating working directory under: {}".format(data_dir)
    os.mkdir(data_dir)

N_daily_csv = os.path.join(data_dir, 'N_seaice_extent_daily_v2.1.csv')
S_daily_csv = os.path.join(data_dir, 'S_seaice_extent_daily_v2.1.csv')
N_climo_csv = os.path.join(data_dir, 'N_seaice_extent_climatology_1981-2010_v2.1.csv')
S_climo_csv = os.path.join(data_dir, 'S_seaice_extent_climatology_1981-2010_v2.1.csv')
   
def pull_datafiles(hemisphere, outdir, climo=False):
    if hemisphere.upper() in ["SOUTH", "S"]:
        data_dir = 'DATASETS/NOAA/G02135/south/daily/data'
        data_file = 'S_seaice_extent_daily_v2.1.csv'
        climo_file = 'S_seaice_extent_climatology_1981-2010_v2.1.csv' if climo else None
    elif hemisphere.upper() in ["NORTH", "N"]:
        data_dir = 'DATASETS/NOAA/G02135/north/daily/data'
        data_file = 'N_seaice_extent_daily_v2.1.csv'
        climo_file = 'N_seaice_extent_climatology_1981-2010_v2.1.csv' if climo else None
    else:
        print "Error in retrieving remote data files (hemisphere arg must be one of ['north', 'n', 'south', 's'])"
        return None
    
    try:
        ftp = FTP('sidads.colorado.edu')
        ftp.login()
        ftp.cwd(data_dir)
        outfile_data = os.path.join(outdir, data_file)
        ftp.retrbinary('RETR {}'.format(data_file), open(outfile_data, 'wb').write)
        if climo:
            outfile_climo = os.path.join(outdir, climo_file)
            ftp.retrbinary('RETR {}'.format(climo_file), open(outfile_climo, 'wb').write)
        ftp.close()
        return 'FTP-PULL OK'
    except all_errors as ftp_err:
        return "Failure to pull NSIDC data files:\n{}".format(ftp_err.message)
        
    

def check_datafiles():
    today = datetime.today().strftime("%Y-%m-%d")
    # check if the files exist, and if so, check their modified times
    # Arctic
    if os.path.isfile(N_daily_csv):
        N_daily_csv_modtime = datetime.fromtimestamp(os.path.getmtime(N_daily_csv)).strftime("%Y-%m-%d")
        if N_daily_csv_modtime < today:
            if os.path.isfile(N_climo_csv):
                print "Pulling most recent Arctic dailies from NSIDC FTP"
                pull_datafiles('north', data_dir)
            else:
                print "Pulling most recent Arctic dailies and climatology from NSIDC FTP"
                pull_datafiles('north', data_dir, True)
    else:
        if os.path.isfile(N_climo_csv):
            print "Pulling most recent Arctic dailies from NSIDC FTP"
            pull_datafiles('north', data_dir)
        else:
            print "Pulling most recent Arctic dailies and climatology from NSIDC FTP"
            pull_datafiles('north', data_dir, True)
    # Antarctic            
    if os.path.isfile(S_daily_csv):
        S_daily_csv_modtime = datetime.fromtimestamp(os.path.getmtime(S_daily_csv)).strftime("%Y-%m-%d")
        if S_daily_csv_modtime < today:
            if os.path.isfile(S_climo_csv):
                print "Pulling most recent Antarctic dailies from NSIDC FTP"
                pull_datafiles('south', data_dir)
            else:
                print "Pulling most recent Antarctic dailies and climatology from NSIDC FTP"
                pull_datafiles('south', data_dir, True)
    else:
        if os.path.isfile(S_climo_csv):
            print "Pulling most recent Antarctic dailies from NSIDC FTP"
            pull_datafiles('south', data_dir)
        else:
            print "Pulling most recent Antarctic dailies and climatology from NSIDC FTP"
            pull_datafiles('south', data_dir, True)

# http://stackoverflow.com/a/34966632
def is_leap_and_29Feb(s):
    return (s.Year % 4 == 0) & \
           ((s.Year % 100 != 0) | (s.Year % 400 == 0)) & \
           (s.Month == 2) & (s.Day == 29)            
    
# WIP
def gen_my_own_stats(daily):
    sdate = daily[:1]
    edate = daily[-1:]
    sdate = datetime.strptime("{}{}{}".format(int(sdate.Year), int(sdate.Month), int(sdate.Day)), '%Y%m%d')
    edate = datetime.strptime("{}{}{}".format(int(edate.Year), int(edate.Month), int(edate.Day)), '%Y%m%d')
    date_index = pd.date_range(sdate, edate, freq='D')
    
    datetime_df = pd.DataFrame(data=pd.to_datetime((daily.Year*10000+daily.Month*100+daily.Day).apply(str), format="%Y%m%d"), 
                               index=daily.index, columns=['Date'])
    oday_df = pd.DataFrame(data=pd.DatetimeIndex(data=datetime_df.Date).dayofyear, 
                           index=daily.index, columns=['Ordinal_date'])
    daily = pd.concat([daily, datetime_df, oday_df], axis=1)
    daily.index = daily.Date
    daily = daily.reindex(date_index, columns=['Year','Month','Day','Extent','Missing','Ordinal_date'])
    # interpolate NaN's using linear interpolation (default)
    #daily.Extent.interpolate(method='linear', inplace=True)
        
    # for 1981-2010
    climo_df = daily.loc[(daily.Year >= 1981) & (daily.Year <= 2010)].copy()
    group = climo_df.groupby(['Month','Day'])
    my_climo = pd.DataFrame(data={'Extent_Mean':group.Extent.mean(), 'Extent_Std':group.Extent.std()},
                            columns=['Extent_Mean', 'Extent_Std'])
    
    daily = daily.join(my_climo, ['Month','Day'])
    
    
    
def plot_it_up(aoi):
    if aoi == 'Global':
        dframe = G_daily_df.copy()
        climo_df = G_climo_df.copy()
    elif aoi == 'Arctic':
        dframe = N_daily_df.copy()
        climo_df = N_climo_df.copy()
    elif aoi == 'Antarctic':
        dframe = S_daily_df.copy()
        climo_df = S_climo_df.copy()
    else:
        print "Error: Area of Interest must be one of ['Global', 'Arctic', 'Antarctic']"
        return None
    curr_year = datetime.today().year
    fig = plt.figure(figsize=(12,7))
    ax = fig.add_subplot(111)
    
    #dframe = dframe.loc[dframe.Year > 1996]

    for year in dframe.Year.unique():
        weight = 0.8
        linetype = 'solid'
        # optional: highlight recent extreme years
        if year in [2012, 2016, 2017]:
            weight = 3.0
        if year == curr_year:
            linetype = 'dashed'
        subframe = dframe.loc[dframe.Year == year]
        # drop any Feb 29 data (helps with plotting)
        mask = is_leap_and_29Feb(subframe)
        subframe = subframe.loc[~mask]
        # generate the ordinal dates for subframe
        # use 1900 for the year since we don't really care about leap year compatibility
        datetime_df = pd.DataFrame(data=pd.to_datetime((1900*10000+subframe.Month*100+subframe.Day).apply(str),format='%Y%m%d'), 
                                   index=subframe.index, columns=['Date'])
        as_index = pd.DatetimeIndex(data=datetime_df.Date)
        oday_df = pd.DataFrame(data=as_index.dayofyear.values, index=subframe.index, columns=['Ordinal_date'])
        subframe = pd.concat([subframe, oday_df], axis=1)
        #subframe['Extent'] = subframe['Extent'][:].rolling(window=3, center=True).mean()
        subframe.plot.line(x='Ordinal_date', y='Extent', ax=ax, label=year, linewidth=weight, linestyle=linetype, zorder=2)

    # drop Feb 29 from climo too
    if climo_df.DOY.max() == 366:
        climo_df = climo_df.loc[climo_df.DOY <> 60]
        climo_df.DOY = climo_df.index # reformat the DOY column (badly)
    # plot climo stdev range
    sigma_mult = 2.
    ax.fill_between(climo_df.DOY, 
                    (climo_df.Average_Extent - sigma_mult*climo_df.STD), 
                    (climo_df.Average_Extent + sigma_mult*climo_df.STD), 
                    facecolor='grey', alpha=0.4, zorder=2, label='$\pm${}$\sigma$ range'.format(int(sigma_mult)))
    # plot climo mean
    climo_df.plot(x='DOY', y='Average_Extent', 
                  ax=ax, label='1981-2010 Mean', 
                  linewidth=1.5, linestyle='dotted', color='black')
    
    title_font = {'family': 'sans-serif',
                  'color': 'black',
                  'weight': 'normal',
                  'size': '18'}
    label_font = {'family': 'sans-serif',
                  'color': 'black',
                  'weight': 'normal',
                  'size': '14'}
    
    minyear = dframe.Year.min()
    maxyear = dframe.Year.max()
    plt.title('Daily {0} Sea Ice Extent  {1}-{2}'.format(aoi, minyear, maxyear), fontdict=title_font)
    #ax.set_xlabel('Ordinal Date', fontdict=label_font) #don't need this now that we re-ticked da X-axis
    ax.grid(which='major', axis='both', color='#ECECEC', linestyle='solid', linewidth=1., zorder=1)    
    # tick intervals based on a non-leap year
    ax.set_xticks([1, 32, 60, 91, 121, 152, 182, 213, 244, 274, 305, 335])
    ax.set_xticklabels(['Jan','Feb','Mar','Apr','May','Jun','Jul','Aug','Sep','Oct','Nov','Dec'])
    ax.set_xlabel('Month', fontdict=label_font)
    ax.set_ylabel('Extent (million square kilometers)', fontdict=label_font)
    ax.legend(ncol=int(pd.np.ceil((len(N_daily_df.Year.unique())+1)/10.)), loc='best', labelspacing=0.3, columnspacing=0.5)
    fig.tight_layout()
    
    # data source message at bottom-left of figure
    fig.text(0.010, 0.016, "Source: NSIDC Sea Ice Index v2.1 daily data files (doi:10.7265/N5736NV7)")
    
    # "Latest-update" message at bottom-right of figure
    edate = subframe[-1:]
    edate = datetime.strptime("{}{}{}".format(int(edate.Year), int(edate.Month), int(edate.Day)), '%Y%m%d')
    fig.text(0.645,0.016,"Latest daily value ({}) may be subject to change".format(edate.strftime("%Y-%m-%d")))

def dump_month_daily_decline(aoi):
    if aoi == 'Global':
        dframe = G_daily_df.copy()
    elif aoi == 'Arctic':
        dframe = N_daily_df.copy()
    elif aoi == 'Antarctic':
        dframe = S_daily_df.copy()
    else:
        print "Error: Area of Interest must be one of ['Global', 'Arctic', 'Antarctic']"
        return None
    
    today = datetime.today()
    curr_year = today.year
    last_month = today.month - 1 if today.month > 1 else 12
    
    subframe = dframe.loc[dframe.Year == curr_year].loc[dframe.Month == last_month]
    y = subframe.Extent
    X = subframe.Day
    X = sm.add_constant(X)
    est = sm.OLS(y, X, missing='drop') # ignore NaN's by setting 'missing' arg
    est = est.fit()
    #trend = est.predict(X)
    # add the trend data points to the subset of dframe
    #subframe = subframe.assign(extent_trend = pd.Series(data=trend, index=subframe.index))
    print est.summary2()
    
if __name__ == "__main__":
    check_datafiles()
    
    N_daily_df = pd.read_csv(N_daily_csv, skiprows=2, names=['Year', 'Month', 'Day','Extent','Missing'], usecols=[0,1,2,3,4])
    S_daily_df = pd.read_csv(S_daily_csv, skiprows=2, names=['Year', 'Month', 'Day','Extent','Missing'], usecols=[0,1,2,3,4])
    G_daily_df = pd.DataFrame(data={"Year":N_daily_df.Year, "Month":N_daily_df.Month, "Day":N_daily_df.Day, 
                                    "Extent":N_daily_df.Extent+S_daily_df.Extent, "Missing":N_daily_df.Missing+S_daily_df.Missing}, 
                                    columns=['Year','Month','Day','Extent','Missing'])
    
    #smoothing
    #N_daily_df['Extent'] = N_daily_df['Extent'][:].rolling(window=5, center=True, win_type='hamming', min_periods=3).mean()
    #S_daily_df['Extent'] = S_daily_df['Extent'][:].rolling(window=5, center=True, win_type='hamming', min_periods=3).mean()
    #G_daily_df['Extent'] = G_daily_df['Extent'][:].rolling(window=5, center=True, win_type='hamming', min_periods=3).mean()
    
    N_climo_df = pd.read_csv(N_climo_csv, skiprows=2, names=['DOY', 'Average_Extent', 'STD', '10th', '25th', '50th', '75th', '90th'])
    S_climo_df = pd.read_csv(S_climo_csv, skiprows=2, names=['DOY', 'Average_Extent', 'STD', '10th', '25th', '50th', '75th', '90th'])
    G_climo_df = pd.DataFrame(data={'DOY': N_climo_df.DOY, 'Average_Extent':N_climo_df.Average_Extent+S_climo_df.Average_Extent, 
                                    'STD':(N_climo_df.STD+S_climo_df.STD)/2., 'p10':N_climo_df['10th']+S_climo_df['10th'], 
                                    'p25':N_climo_df['25th']+S_climo_df['25th'], 'p50':N_climo_df['50th']+S_climo_df['50th'],
                                    'p75':N_climo_df['75th']+S_climo_df['75th'], 'p90':N_climo_df['90th']+S_climo_df['90th']
                                    }, columns=['DOY', 'Average_Extent', 'STD', 'p10', 'p25', 'p50', 'p75', 'p90'])
    
    for aoi in ['Global', 'Arctic', 'Antarctic']:
        plot_it_up(aoi)
        dump_month_daily_decline(aoi)

    # plot all 3 figures
    plt.show()