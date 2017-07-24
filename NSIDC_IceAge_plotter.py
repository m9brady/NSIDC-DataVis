# -*- coding: utf-8 -*-
import os
import tempfile
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as pat
from matplotlib.colors import ListedColormap, BoundaryNorm
import mpl_toolkits.basemap as bm
from glob import glob

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