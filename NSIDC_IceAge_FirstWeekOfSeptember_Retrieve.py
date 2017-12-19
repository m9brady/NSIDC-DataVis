# -*- coding: utf-8 -*-
import os
import tempfile
import numpy as np
from glob import glob


data_dir = os.path.join(tempfile.gettempdir(), 'NSIDC_SeaIceAge', 'nsidc0611_seaice_age_v3')
if not os.path.exists(data_dir): os.mkdir(data_dir)

# set to false for panArctic
beaufort_subset = True

# minLat, maxLat, minLon, maxLon
beaufort_bnds = [60.0, 82.0, -170.0, -105.0]

os.chdir(data_dir)
lat_file = os.path.join('Na12500-CF_latitude.dat')
lon_file = os.path.join('Na12500-CF_longitude.dat')
bin_list = sorted(glob(os.path.join('data', '*', 'iceage.grid.week.????.36.n.*.bin')))
# final array is 722**2 rows by lat+lon+number_of_bins columns
final_arr = np.empty((722**2,2+len(bin_list)),dtype='float32')
# lat column
final_arr[:,0] = np.fromfile(lat_file,dtype='float32')
# lon column
final_arr[:,1] = np.fromfile(lon_file,dtype='float32')

csv_header = "lat,lon"
for b in bin_list:
    csv_header += ","
    csv_header += "{}_age".format(b.split("\\")[-1].split("_")[2])

for idx, bin_file in enumerate(bin_list):
    final_arr[:,idx+2] = np.fromfile(bin_file,dtype='uint8')

if beaufort_subset:
    # subset to Beaufort Sea region
    final_arr = final_arr[final_arr[:,0] >= beaufort_bnds[0]]
    final_arr = final_arr[final_arr[:,0] <= beaufort_bnds[1]]
    final_arr = final_arr[final_arr[:,1] >= beaufort_bnds[2]]
    final_arr = final_arr[final_arr[:,1] <= beaufort_bnds[3]]

np.savetxt("final_arr.csv", final_arr, delimiter=",", comments='', header=csv_header)