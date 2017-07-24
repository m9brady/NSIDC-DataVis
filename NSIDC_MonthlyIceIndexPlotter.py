# -*- coding: utf-8 -*-
import os # filesystem directory manipulation
from glob import glob # generating list of file paths
from datetime import datetime # managing dates
from matplotlib.offsetbox import AnchoredText # statistics annotation on plot
import matplotlib.pyplot as plt # creating the figure and axes
import numpy as np # array manipulation
import pandas as pd # data frame formatting and data ingestion
import statsmodels.api as sm # Ordinary-least squares regression
from ftplib import FTP, all_errors # Downloading data from NSIDC servers
import tempfile # finding the temp directory on filesystem

# setup the working directories
######## global variables are usually not recommended ###########
data_dir = os.path.join(tempfile.gettempdir(), "NSIDC_MonthlySeaIceIndex")
csv_dir = os.path.join(data_dir, 'csv')
if not os.path.exists(data_dir): 
    print "Creating working directory under: {}".format(data_dir)
    os.makedirs(csv_dir)
else:
    if not os.path.exists(csv_dir):
        os.mkdir(csv_dir)

# retrieve all the source data via FTP, without any form of graceful error-catching
def pull_nsidc_data(hemisphere='N'):
    
    nsidc_ftp_host = 'sidads.colorado.edu'
    # different directories based on hemisphere
    nsidc_ftp_dir = '/DATASETS/NOAA/G02135/north/monthly/data/' if hemisphere.lower() == 'n' else '/DATASETS/NOAA/G02135/south/monthly/data/'
    try:
        print "Retrieving {} hemisphere monthlies from NSIDC FTP...".format(hemisphere)
        ftp = FTP(nsidc_ftp_host)
        # anonymous login
        ftp.login() 
        ftp.cwd(nsidc_ftp_dir)
        # Get list of source data files. Assumes the vX.X naming convention...
        csv_files = ftp.nlst('{}_??_extent_v?.?.csv'.format(hemisphere))
        new_file_count = 0
        for remote_csv in csv_files:
            local_csv = os.path.join(csv_dir, os.path.basename(remote_csv))
            # check if local csv file exists
            if os.path.exists(local_csv):
                # compare last-modified times of local file with remote file
                local_mtime = datetime.fromtimestamp(os.path.getmtime(local_csv))
                remote_mtime = datetime.strptime(ftp.sendcmd('MDTM ' + remote_csv)[4:], '%Y%m%d%H%M%S')
                # if the remote file is not newer, we don't overwrite local
                if local_mtime >= remote_mtime:
                    continue
                # otherwise we replace the local file with the remote file
                else:
                    os.remove(local_csv)
                    ftp.retrbinary('RETR {}'.format(remote_csv), open(local_csv, 'wb').write)
                    new_file_count += 1
            # if no local file exists, go get it
            else:
                ftp.retrbinary('RETR {}'.format(remote_csv), open(local_csv, 'wb').write)
                new_file_count += 1
        ftp.close()
        if new_file_count == 0:
            print "Newest NSIDC data files already present"
        else:
            print "FTP-RETR of {} newer NSIDC data files OK".format(new_file_count)
        return 0
    except all_errors as ftp_err:
        print "Failure to pull NSIDC data files:\n{}".format(ftp_err)
        return 1


# convert the retrieved csv datafiles into a pandas dataframe, replacing -9999 values with NaN's
def format_dframe(hemisphere='N'):
    csv_files = glob(os.path.join(csv_dir, '{}_??_extent_v?.?.csv'.format(hemisphere)))
    # generate an empty pandas DataFrame for appending our source data
    dframe = pd.DataFrame()
    for csv in csv_files:
        # I <3 pandas.read_csv
        dframe = dframe.append(pd.read_csv(csv))
    # reformat column titles to strip whitespace 
    # assumes the header will never change...
    dframe.columns = ['year', 'month', 'data_type', 'region', 'extent', 'area']
    # default sort by year ascending, then month ascending
    dframe = dframe.sort_values(['year','month'])
    # not really necessary, but may be nice to have later
    #dframe['timeslice'] = dframe.apply(lambda x: datetime.datetime.strptime("{}{}".format(x['year'],x['month']), "%Y%m"), axis=1)
    # account for -9999 values a.k.a. missing data
    dframe.loc[dframe['extent'] == -9999, ['extent', 'area', 'data_type']] = np.nan
    return dframe


# subset and plot the dataframe depending on the passed month value  
def plot_dframe(dframe, month=None, summary=False):
    # rudimentary parameter checking
    if month is not None and month in xrange(1,13):
        # subset input dataframe to the month of interest
        newframe = dframe.loc[dframe['month'] == month]        
        # determine trend using ordinary least-squares regression
        y = newframe.extent
        X = newframe.year
        X = sm.add_constant(X)
        est = sm.OLS(y, X, missing='drop') # ignore NaN's by setting 'missing' arg
        est = est.fit()
        trend = est.predict(X)
        # add the trend data to the subset of dframe
        newframe = newframe.assign(trend = pd.Series(data=trend, index=newframe.index))

        screen_dpi = 100. # default windows 7 screen dpi is 96
        fig = plt.figure(figsize=(1024./screen_dpi,768./screen_dpi), facecolor='white', dpi=screen_dpi)
        fig.set_size_inches(w=1024./screen_dpi,h=768./screen_dpi,forward=True)
        ax = fig.add_subplot(111)
        # if there are any NaN values, we have to plot twice
        if len(newframe.loc[pd.isnull(newframe['extent'])]) > 0:
            # we plot with the NaN showing (dashed line)
            newframe.dropna().plot(x='year', y='extent', ax=ax, legend=True, style='k--', dashes=[2.,1.5], linewidth=2.)
        # plot the regular time series
        newframe.plot(x='year', y='extent',  ax=ax, legend=False, style='k-', linewidth=2.5)
        # plot the trendline
        newframe.plot(x='year', y='trend', ax=ax, legend=False, style='b-', linewidth=2.5)
        # possible bug in matplotlib where the legend acts strangely, despite setting as False for some plots
        if len(newframe.loc[pd.isnull(newframe['extent'])]) > 0:
            ax.legend(['Missing Data'], loc='upper right', fontsize='large', edgecolor='k', fancybox=False, shadow=False)
        # pull month string from int using datetime
        monthname = datetime(1900, int(month), 1).strftime("%B")
        # define fontdicts
        title_font = {'fontsize': 20, 'fontweight': 'regular'}
        axes_label_font = {'fontsize': 16, 'fontweight': 'regular'}
        #axes_ticklabel_font = {'fontsize': 10, 'fontweight': 'regular'}
        # figure formatting
        if newframe.region[0].strip() == 'N':
            ax.set_title("Average Monthly Arctic Sea Ice Extent\n{} {} - {}".format(monthname, newframe['year'].min(), newframe['year'].max()), title_font)
        elif newframe.region[0].strip() == 'S':
            ax.set_title("Average Monthly Antarctic Sea Ice Extent\n{} {} - {}".format(monthname, newframe['year'].min(), newframe['year'].max()), title_font)
        else:
            ax.set_title("Average Monthly Global Sea Ice Extent\n{} {} - {}".format(monthname, newframe['year'].min(), newframe['year'].max()), title_font)
        ax.set_ylabel("Extent (million square kilometers)", axes_label_font)
        ax.set_xlabel("Year", axes_label_font)
        
        ax.set_xbound(lower=newframe['year'].min()-1, upper=newframe['year'].max()+1)
        ax.set_ybound(lower=newframe['extent'].min()-0.5, upper=newframe['extent'].max()+0.5)
        #ax.set_ybound(lower=np.floor(newframe['extent'].min()), upper=np.ceil(newframe['extent'].max()))
        ax.grid(b=True, color='#D8D8D8', which='major', alpha=0.5)
        # major ticks will be in increments of 4
        ax.set_xticks(np.arange(newframe['year'].min(), newframe['year'].max(), 4), minor=False)
        # minor ticks will be for every year (add 1 to the end to retain the most recent year)
        ax.set_xticks(np.arange(newframe['year'].min(), newframe['year'].max()+1, 1), minor=True)
        #ax.set_xticklabels(ax.get_xticklabels(), axes_ticklabel_font)
        #ax.set_yticklabels(ax.get_yticklabels(), axes_ticklabel_font)
        #ax.grid(color='#D8D8D8', which='minor', alpha=0.4)
        
        fig.text(0.010, 0.016, "Source: NSIDC Sea Ice Index v2.1 monthly data files (doi:10.7265/N5736NV7)")
        # some fun statistics and an estimate of what the NSIDC Monthly Sea Ice News and Analysis will be
        if summary:
            print est.summary2()
        # get the climatological (30-year) mean extent for 1981-2010
        climo_1981_2010 = newframe[newframe.loc[newframe.year == 1981].index.tolist()[0]:newframe.loc[newframe.year == 2011].index.tolist()[0]].extent.mean()
        # linear rate of change in million sq km
        change_rate = est.params.year
        # add info to plot
        #ax.text(1978.5, newframe['extent'].min()-0.3, "Linear rate of change: {} km$^2$ per year".format(int(round(change_rate * 1000000, -2))))
        #ax.text(1978.5, newframe['extent'].min()-0.4, "Percent per decade (relative to 1981-2010 mean): {}%".format(round(change_rate / climo_1981_2010 * 10 * 100, 2)))
        #bbox_props = dict(boxstyle='square', fc='w', ec='0.5', alpha=1.0)
        frame_text = "Linear rate of change: {} km$^2$ per year\nPercent per decade (relative to 1981-2010 mean): {}%".format(int(round(change_rate * 1000000, -2)),round(change_rate / climo_1981_2010 * 10 * 100, 2))
        frame_text = AnchoredText("Linear rate of change: {} km$^2$ per year\nPercent per decade (relative to 1981-2010 mean): {}%".format(int(round(change_rate * 1000000, -2)),round(change_rate / climo_1981_2010 * 10 * 100, 2)),
                                  loc=3, prop=dict(size=11))
        #ax.annotate(frame_text, loc='lower left', bbox=bbox_props)
        ax.add_artist(frame_text)
        
        print "\nEstimated stats for the Monthly Sea Ice News\n-------------------------------------\n"
        print "Linear rate of change: {} square kilometers".format(int(round(change_rate * 1000000, -2)))
        print "Percent per decade (relative to 1981-2010 climo): {}%".format(round(change_rate / climo_1981_2010 * 10 * 100, 2))
        #plt.show()
    else:
        raise ValueError("Must supply an integer argument for month (e.g. 12 for December)")


def plot_pct_anomaly(dframe, month=None, summary=False):
    if month is not None and month in xrange(1,13):
        # subset input dataframe to the month of interest
        newframe = dframe.loc[dframe['month'] == month]
        # get the climatological (30-year) mean extent for 1981-2010
        climo_1981_2010 = newframe[newframe.loc[newframe.year == 1981].index.tolist()[0]:newframe.loc[newframe.year == 2011].index.tolist()[0]].extent.mean()
        # calculate the anomalies relative to 30-year mean
        extent_anomaly = (newframe.extent - climo_1981_2010)/climo_1981_2010*100.
        extent_anomaly.name = 'extent_anomaly'
        # slap the two dframes together
        newframe = pd.concat([newframe, extent_anomaly], axis=1)
        # determine trend using ordinary least-squares regression
        y = newframe.extent_anomaly
        X = newframe.year
        X = sm.add_constant(X)
        est = sm.OLS(y, X, missing='drop') # ignore NaN's by setting 'missing' arg
        est = est.fit()
        trend = est.predict(X)
        # add the trend data points to the subset of dframe
        newframe = newframe.assign(extent_anomaly_trend = pd.Series(data=trend, index=newframe.index))
        # add a column for plotting the zero-line
        #newframe = newframe.assign(zeroline = pd.Series(data=np.zeros(len(trend)), index=newframe.index))

        # plot time
        screen_dpi = 100. # default windows 7 screen dpi is 96
        fig = plt.figure(figsize=(1024./screen_dpi,768./screen_dpi), facecolor='white', dpi=screen_dpi)
        fig.set_size_inches(w=1024./screen_dpi,h=768./screen_dpi,forward=True)
        ax = fig.add_subplot(111)
        # plot the x=0 line
        #newframe.plot(x='year', y='zeroline', ax=ax, legend=False, style='r-', linewidth=1.5, alpha=0.4)
        # if there are any NaN values, we have to plot twice
        if len(newframe.loc[pd.isnull(newframe['extent_anomaly'])]) > 0:
            # we plot with the NaN showing (dashed line)
            newframe.dropna().plot(x='year', y='extent_anomaly', ax=ax, legend=True, style='k--', dashes=[2.,1.5], linewidth=2.)
        # plot the regular time series
        newframe.plot(x='year', y='extent_anomaly',  ax=ax, legend=False, style='k-P', markersize=8, linewidth=2.0)
        # plot the trendline
        newframe.plot(x='year', y='extent_anomaly_trend', ax=ax, legend=False, style='--', color='grey', dashes=[4,4], linewidth=1.5)
        # possible bug in matplotlib where the legend acts strangely, despite setting as False for some plots
        if len(newframe.loc[pd.isnull(newframe['extent'])]) > 0:
            ax.legend(['Missing Data'], loc='upper right', fontsize='large', edgecolor='k', fancybox=False, shadow=False)
        # pull month string from int using datetime
        monthname = datetime(1900, int(month), 1).strftime("%B")
        
        # define fontdicts
        title_font = {'fontsize': 20, 'fontweight': 'regular'}
        axes_label_font = {'fontsize': 16, 'fontweight': 'regular'}
        
        # figure formatting
        if newframe.region[0].strip() == 'N':
            ax.set_title("Monthly Arctic Sea Ice Extent Anomalies\n{} {} - {}".format(monthname, newframe['year'].min(), newframe['year'].max()), title_font)
        elif newframe.region[0].strip() == 'S':
            ax.set_title("Monthly Antarctic Sea Ice Extent Anomalies\n{} {} - {}".format(monthname, newframe['year'].min(), newframe['year'].max()), title_font)
        else:
            ax.set_title("Monthly Global Sea Ice Extent Anomalies\n{} {} - {}".format(monthname, newframe['year'].min(), newframe['year'].max()), title_font)
        ax.set_ylabel("Anomaly (% difference from 30-year mean)", axes_label_font)
        ax.set_xlabel("Year", axes_label_font)
        #ax.set_ybound(lower=np.floor(newframe['extent'].min()), upper=np.ceil(newframe['extent'].max()))
        ax.set_xbound(lower=newframe['year'].min()-1, upper=newframe['year'].max()+1)
        ax.set_ybound(lower=newframe['extent_anomaly'].min()-5, upper=newframe['extent_anomaly'].max()+5)
        ax.grid(b=True, color='#D8D8D8', which='major', alpha=0.5)
        # major ticks will be in increments of 4
        ax.set_xticks(np.arange(newframe['year'].min(), newframe['year'].max(), 4), minor=False)
        # minor ticks will be for every year (add 1 to the end to retain the most recent year)
        ax.set_xticks(np.arange(newframe['year'].min(), newframe['year'].max()+1, 1), minor=True)
        #ax.set_xticklabels(ax.get_xticklabels(), axes_ticklabel_font)
        #ax.set_yticklabels(ax.get_yticklabels(), axes_ticklabel_font)
        #ax.grid(color='#D8D8D8', which='minor', alpha=0.4)
        #ax.text(1978.5,newframe['extent_anomaly'].min()-4.3, u"1981-2010 mean: {} million km$^2$".format(round(climo_1981_2010,1)))
        frame_text = AnchoredText(u"1981-2010 mean: {} million km$^2$".format(round(climo_1981_2010,1)),
                                  loc=3, prop=dict(size=11))
        #ax.annotate(frame_text, loc='lower left', bbox=bbox_props)
        ax.add_artist(frame_text)
        
        fig.text(0.010, 0.016, "Source: NSIDC Sea Ice Index v2.1 monthly data files (doi:10.7265/N5736NV7)")
        # some fun "statistics" and an estimate of what the NSIDC Monthly Sea Ice News and Analysis will mention
        pct_decade = est.params.year * 10
        pct_decade_std = est.bse.year * 10
        if summary:
            print est.summary2()
        print u"\nAnomaly slope = {} \u00B1{} percent per decade".format(round(pct_decade,1), round(pct_decade_std,1))
        

def plot_master(month=datetime.today().month):
    #pull_nsidc_data('N')
    #pull_nsidc_data('S')
    north_ice_extent = format_dframe('N')
    plot_dframe(north_ice_extent, month)
    plot_pct_anomaly(north_ice_extent, month)



if __name__ == "__main__":
    plot_master(month=7)
    plt.show()