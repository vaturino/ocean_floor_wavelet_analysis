import sys
import os
import json
import numpy as np
import matplotlib.pyplot as plt
import rasterio
from juwavelet import transform
from sklearn.mixture import BayesianGaussianMixture
from scipy.signal import find_peaks
import xarray as xr
import cartopy.crs as ccrs
import matplotlib.colors as mcolors
import pandas as pd 
import cmocean.cm as cmo
from scipy.ndimage import maximum_filter
from scipy.spatial import distance_matrix
from functions import *
import warnings

# Suppress all execution warnings (like Sklearn GMM convergence or SciPy peaks warnings)
warnings.filterwarnings("ignore")


# ======================================================
# USER PARAMETERS
# ======================================================
fig_dpi = 120
downsample_factor = 1  

# ======================================================
# PATHS & ARGV SETUP
# ======================================================
system = str(sys.argv[1])
json_folder = '/Users/valeturino/postdoc_local/wavelet_analysis/json_folder'
json_file = os.path.join(json_folder, f'{system}.json')

# New unified output infrastructure
base_output_dir = f"/Users/valeturino/postdoc_local/wavelet_analysis_results/{system}"
figure_folder_1d = os.path.join(base_output_dir, "1D_analysis")
figure_folder_2d = os.path.join(base_output_dir, "2D_analysis")

os.makedirs(figure_folder_1d, exist_ok=True)
os.makedirs(figure_folder_2d, exist_ok=True)

# ======================================================
# LOGGING SYSTEM CONFIGURATION
# ======================================================

log_file_path = os.path.join(base_output_dir, f'{system}.log')

# Redirect stdout and stderr quietly to the isolated log file
log_file = open(log_file_path, 'w', encoding='utf-8')
sys.stdout = log_file
sys.stderr = log_file  

print(f"--- Execution Started for System: {system} ---")

# ======================================================
# LOAD COORDINATES FROM JSON
# ======================================================
with open(json_file) as f:
    coords = json.load(f)

w = float(coords['min_lon'])
e = float(coords['max_lon'])
s = float(coords['min_lat'])
n = float(coords['max_lat'])

# ======================================================
# LOAD COARSER ML MAP
# ======================================================
dat_folder = f'/Users/valeturino/postdoc_local/datasets'
nc_file = f'{dat_folder}/bathymetry_model_Feb_2025.nc'

nc_ds = xr.open_dataset(nc_file, engine="netcdf4", chunks="auto")

data_var = 'height' if 'height' in nc_ds.variables else 'z'

is_north_to_south = nc_ds.lat[0] > nc_ds.lat[-1]
lat_bounds = slice(n, s) if is_north_to_south else slice(s, n)

w_f, e_f = wrap_lon(w), wrap_lon(e)

if w_f > e_f:
    subset = nc_ds[data_var].sel(lon=((nc_ds.lon >= w_f) | (nc_ds.lon <= e_f)), lat=lat_bounds).compute()
else:
    subset = nc_ds[data_var].sel(lon=slice(w_f, e_f), lat=lat_bounds).compute()

subset = subset.sortby('lat', ascending=True).sortby('lon', ascending=True)

if downsample_factor > 1:
    subset = subset.isel(lat=slice(None, None, downsample_factor), lon=slice(None, None, downsample_factor))

# ======================================================
# ESTIMATE Lcomp as the Avg Distance Between Bathymetric Highs
# ======================================================
# 1. Get resolution track steps 
dlon_km, dlat_km = get_grid_resolution_km(subset.lon.values, subset.lat.values, s, n)

# 2. Map coordinates into kilometer sequences
lons_km, _ = degrees_to_km_coordinates(subset.lon.values, subset.lat.values, s, n)
_, lats_km = degrees_to_km_coordinates(subset.lon.values, subset.lat.values, s, n)

raw_peaks = []

# 3. Extract raw peak coordinates along Longitude profiles
lat_sample_step = max(1, len(subset.lat) // 20)
for i in range(0, len(subset.lat), lat_sample_step):
    profile = subset.isel(lat=i).values
    peaks, _ = find_peaks(profile, prominence=100)  
    for p_idx in peaks:
        raw_peaks.append([lons_km[p_idx], lats_km[i]])

# 4. Extract raw peak coordinates along Latitude profiles
lon_sample_step = max(1, len(subset.lon) // 20)
for j in range(0, len(subset.lon), lon_sample_step):
    profile = subset.isel(lon=j).values
    peaks, _ = find_peaks(profile, prominence=100)
    for p_idx in peaks:
        raw_peaks.append([lons_km[j], lats_km[p_idx]])

raw_peaks = np.array(raw_peaks)

# 5. CONSOLIDATE DUPLICATES (Spatial Clustering)
consolidation_radius = 15.0  
unique_seamounts = []

if len(raw_peaks) > 0:
    remaining_peaks = list(raw_peaks)
    while len(remaining_peaks) > 0:
        current = remaining_peaks.pop(0)
        dists = np.linalg.norm(np.array(remaining_peaks) - current, axis=1) if remaining_peaks else np.array([])
        
        cluster = [current]
        to_remove = []
        for idx, d in enumerate(dists):
            if d < consolidation_radius:
                cluster.append(remaining_peaks[idx])
                to_remove.append(idx)
        
        for idx in sorted(to_remove, reverse=True):
            remaining_peaks.pop(idx)
            
        unique_seamounts.append(np.mean(cluster, axis=0))

unique_seamounts = np.array(unique_seamounts)

# ======================================================
# 6. Calculate Lcomp (Robust Regional Spacing Length)
# ======================================================
if len(unique_seamounts) > 1:
    dist_mat = distance_matrix(unique_seamounts, unique_seamounts)
    np.fill_diagonal(dist_mat, np.inf)
    
    sorted_distances = np.sort(dist_mat, axis=1)
    primary_neighbor_dists = sorted_distances[:, 1]
    
    Lcomp_raw = float(np.median(primary_neighbor_dists))
    Lcomp = Lcomp_raw * 1.68
    
    print(f"Region '{system}': Consolidated profile hits into {len(unique_seamounts)} distinct seamounts.")
    print(f"Calculated Lcomp (Macro-structural spacing): {Lcomp:.2f} km")
else:
    Lcomp = 50.0  
    print(f"Region '{system}': Domain too uniform to isolate separate structures. Baseline Lcomp: {Lcomp} km")

# ======================================================
# DATA SANITIZATION & PRE-REQUISITES
# ======================================================
subset = subset.fillna(0)

if subset.min() == subset.max():
    subset = subset + np.random.normal(0, 1e-6, subset.shape)

if subset.dims[0] != 'lat':
    subset = subset.transpose('lat', 'lon')

# ======================================================
# AUTOMATED WAVELET ORCHESTRATION & DECOUPLED RECONSTRUCTION
# ======================================================
print("Executing Wavelet Analysis...")

# Pass both isolated directories explicitly to the analysis pipeline engine
results = run_wavelet_analysis(
    figure_folder_1d=figure_folder_1d, 
    figure_folder_2d=figure_folder_2d, 
    subset=subset, 
    w=w, e=e, s=s, n=n, 
    Lcomp=Lcomp, 
    system_name=system
)

if results is not None:
    # 1. Unroll the 1D Spectrum reconstructions into the 1D directory path 
    print("Computing automated 1D spectral feature maps...")
    for i, idx in enumerate(results['peaks_1d']):
        wl_val = results['wl_norm'][idx]
        plot_reconstruction(i, idx, results['decomp'], results['data_dict'],
                            wl_val, figure_folder_1d, peak_theta=None)

    # 2. Unroll the 2D GMM Spectrum reconstructions into the 2D directory path
    print("Computing automated 2D GMM directional maps...")
    for i, peak in enumerate(results['gmm_peaks']):
        target_wl = peak['wl']
        target_theta = peak['theta']

        # Index snap verification
        idx = (np.abs(results['wl_norm'] - target_wl)).argmin()

        plot_reconstruction(i, idx, results['decomp'], results['data_dict'],
                            target_wl, figure_folder_2d, peak_theta=target_theta)
        

    # Zip the entire base directory containing both separate subfolders
    zip_name = f"/Users/valeturino/postdoc_local/wavelet_analysis_results/{system}_plots_archive"
    import shutil
    shutil.make_archive(zip_name, 'zip', base_output_dir)
    print(f"Executions completed. Process mappings saved successfully.")

# Clean up resources and close logging stream safely
log_file.close()