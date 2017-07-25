# -*- coding: utf-8 -*-
import os
import tempfile
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as pat
from matplotlib.colors import ListedColormap, BoundaryNorm
import mpl_toolkits.basemap as bm
from glob import glob

import requests
from bs4 import BeautifulSoup

# requests method
# assumes your .netrc is setup beforehand with NASA EarthData credentials
def get_iceage_data(daac_url='https://daacdata.apps.nsidc.org/pub/DATASETS/nsidc0611_seaice_age_v3'):
    # setup local base directory (relative to current working directory)
    base_dir = os.path.basename(daac_url)
    if not os.path.exists(base_dir): os.mkdir(base_dir)
    listing = []
    # first get the directory listing and lat/lon array files
    r = requests.get(daac_url)
    if r.status_code == 200:
        rsoup = BeautifulSoup(r.text, 'lxml')
        listing = [row.findAll('a')[1].text for row in rsoup.findAll('tr')]
        lat_remote = daac_url+'/Na12500-CF_latitude.dat' # never changes?
        lat_local = os.path.join(base_dir, os.path.basename(lat_remote))
        if os.path.basename(lat_remote) in listing:
            if not os.path.isfile(lat_local):
                lat = requests.get(lat_remote)
                if lat.status_code == 200:
                    with open(lat_local, 'wb') as latbin:
                        latbin.write(lat.content)
        lon_remote = daac_url+'/Na12500-CF_longitude.dat' # never changes?
        lon_local = os.path.join(base_dir, os.path.basename(lon_remote))
        if os.path.basename(lon_remote) in listing:
            if not os.path.isfile(lon_local):
                lon = requests.get(lon_remote)
                if lon.status_code == 200:
                    with open(lon_local, 'wb') as lonbin:
                        lonbin.write(lon.content)
    else:
        print r.status_code, r.reason
        return -1
    r.close()
    del rsoup
    # check to ensure we didn't get an empty directory listing
    if len(listing) == 0:
        print 'Error occurred during root tree listing of {}'.format(daac_url)
        return -1
    # establish a list of subfolders (should be a bunch of years)
    url_list = []
    if 'data/' in listing:
        new_url = daac_url + '/data/'
        d = requests.get(new_url)
        if d.status_code == 200:
            dsoup = BeautifulSoup(d.text, 'lxml')
            year_list = [row.findAll('a')[1].text for row in dsoup.findAll('tr') if len(row.findAll('a')[1].text) == 5]
            url_list = [new_url+ '{}'.format(y) for y in year_list]
            del year_list
        else:
            print d.status_code, d.reason
            return -1
        d.close()
        del dsoup
    # check to ensure we didn't get an empty directory listing
    if len(url_list) == 0:
        print 'Error occurred during subdirectory listing of {}'.format(daac_url)
        return -1
    bin_urls = []
    for url in url_list:
        u = requests.get(url)
        if u.status_code == 200:
            # too lazy to check for empty year-directories here, will probably come back to bite me
            usoup = BeautifulSoup(u.text, 'lxml')
            bin_list = [row.findAll('a')[1].text for row in usoup.findAll('tr') if '.bin' in row.findAll('a')[1].text]
            bin_urls.append([url + '{}'.format(b) for b in bin_list])
        u.close()
        del usoup
    if len(bin_urls) == 0:
        print 'Error occurred during binfile discovery of {}'.format(daac_url)
        return -1
    final_urls = [item for sublist in bin_urls for item in sublist]
    del bin_urls
    # finally, download the binaries
    for bin_remote in final_urls:
        bin_local = os.path.join(base_dir, bin_remote.split(base_dir)[-1][1:].replace('/', os.sep))
        bin_localdir = os.path.dirname(bin_local)
        if not os.path.exists(bin_localdir): os.makedirs(bin_localdir)
        if not os.path.isfile(bin_local):
            f = requests.get(bin_remote)
            if f.status_code == 200:
                with open(bin_local, 'wb') as outbin:
                    outbin.write(f.content)
                    print "Downloaded {} to {}".format(os.path.basename(bin_local), os.path.abspath(bin_localdir))
            f.close()
        else:
            print "File exists for {}. Skipping...".format(os.path.basename(bin_local))
    
    
    
    
    
    


#http://stackoverflow.com/a/10824420
# a function to create a 1-d list from a list containing lists of lists of lists of lists of lists
def flatten(container):
    for i in container:
        if isinstance(i, (list,tuple)):
            for j in flatten(i):
                yield j
        else:
            yield i


data_dir = os.path.join(tempfile.gettempdir(), 'NSIDC_SeaIceAge')
if not os.path.exists(data_dir): os.mkdir(data_dir)
outdir = os.path.join(data_dir, "out_jpeg")
os.chdir(data_dir)

lats = os.path.join(data_dir, 'data', 'Na12500-CF_latitude.dat')
lons = os.path.join(data_dir, 'data', 'Na12500-CF_longitude.dat')
lat_arr = np.fromfile(lats,dtype='float32').reshape((722,722))
lon_arr = np.fromfile(lons,dtype='float32').reshape((722,722))

age_list = glob(os.path.join(data_dir, 'data', 'bins', '*', 'iceage.*.bin'))

for infile in age_list:
    fdir, fname = os.path.split(infile)
    version = fname.split(".n.")[-1].rstrip(".bin").replace("v", "version: ")
    outfile = os.path.join(outdir, fname.replace(".bin",".jpg"))
    if not os.path.exists(outfile):

        img_arr = np.fromfile(infile,dtype='uint8').reshape((722,722))
     
        # ENHANCE!
        #img_arr = img_arr[111:611, 111:611]
        #lat_arr = lat_arr[111:611, 111:611]
        #lon_arr = lon_arr[111:611, 111:611]
        # ENHANCE! but not as much!
        #img_arr = img_arr[30:692, 30:692]
        #lat_arr = lat_arr[30:692, 30:692]
        #lon_arr = lon_arr[30:692, 30:692]
        
        year, week = fname.split(".")[-5:-3]
        plot_arr = np.copy(img_arr)
        plot_arr[plot_arr == 255] = 0
        plot_arr[plot_arr > 30] = 30
        #plot_bounds = np.unique(plot_arr).tolist()
        #plot_cmap = ListedColormap(["#FFFFFF","#3F51A3","#6FCCDC","#6FBF44","#FCB117","#F02511"],"indexed")
        #plot_norm = BoundaryNorm(plot_bounds,plot_cmap.N)
        
        bmap = bm.maskoceans(lon_arr, lat_arr, plot_arr, inlands=False, resolution='i')
        land = np.invert(bmap.mask.astype('uint8')*255)
        
        combined = land+plot_arr
        combined[np.where(combined%5 <> 0)]+=1
        
        bounds = np.unique(combined).tolist()
        # find the amount of 5+ year MYI classes
        myi5 = len(bounds[bounds.index(25):bounds.index(bounds[-1])])
        # nsidc ice age v3.0 cmap, fills in as many 5+ myi classes as necessary
        cmap = ListedColormap(list(flatten(["#FFFFFF","#3F51A3","#6FCCDC","#6FBF44","#FCB117",["#F02511"] * myi5,"#CCCCCC"])),"indexed")
        #norm = BoundaryNorm(bounds,cmap.N)
        norm = BoundaryNorm([0,5,10,15,20,25,254,255], cmap.N)
        
        fig = plt.figure(figsize=(6,6))
        #ax = fig.add_subplot(111)
        ax = fig.add_axes([0., 0., 1., 1.])
        ax.axis('off')
        #m=Basemap(projection='npaeqd', boundinglat=lat_arr.min(), lon_0=0.)
        #m.drawcoastlines()
        #img = axes.imshow(img_arr,cmap=cmap,norm=norm)
        img = ax.imshow(combined, cmap=cmap, norm=norm)
        #plt.colorbar(img,cmap=cmap,norm=norm,boundaries=bounds,ticks=bounds)
        #plt.colorbar(img,cmap=plot_cmap,norm=plot_norm,boundaries=plot_bounds,ticks=plot_bounds)
        title_font = {'family':'sans-serif', 'size':18}
        subtitle_font = {'family':'sans-serif', 'size':12}
        source_font = {'family':'sans-serif', 'size':9}
        bbox_style = dict(boxstyle="square", fc='#F4F4F4', lw=.4, pad=0.5)
        # plot the legend patches way off-screen
        ax.add_patch(pat.Rectangle((10000, 10000), 1, 1, color="#3F51A3", label='1'))
        ax.add_patch(pat.Rectangle((10000, 10000), 1, 1, color="#6FCCDC", label='2'))
        ax.add_patch(pat.Rectangle((10000, 10000), 1, 1, color="#6FBF44", label='3'))
        ax.add_patch(pat.Rectangle((10000, 10000), 1, 1, color="#FCB117", label='4'))
        ax.add_patch(pat.Rectangle((10000, 10000), 1, 1, color="#F02511", label='5+'))
        #ax.add_patch(pat.Rectangle((600, 600), 50,30, color="#CCCCCC", label='Land'))
        ax.legend(loc=7, title='Age (years)', edgecolor='0.4', facecolor='#F4F4F4',
                  framealpha=1.) #center-right
        
        fig.text(0.02, 0.95, "EASE-Grid Arctic Sea Ice Age", fontdict=title_font)
        fig.text(0.02, 0.88, "Week {} {}".format(week, year), fontdict=subtitle_font)
        #fig.text(0.02, 0.02, "{}".format(version), fontdict=source_font)
        fig.text(0.615, 0.02, "Source: Tschudi, Fowler, Maslanik,\nStewart, and Meier (2016)\ndx.doi.org/10.5067/PFSVFZA9Y85G", 
                 ha='left', va='bottom', fontdict=source_font, bbox=bbox_style)

        fig.savefig(outfile)
        plt.close(fig)
        print os.path.basename(outfile)
        #plt.show()