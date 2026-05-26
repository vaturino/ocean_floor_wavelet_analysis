import os
import json
import numpy as np
import xarray as xr
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import matplotlib.patches as mpatches

# ======================================================
# PARAMETERS AND FOLDERS
# ======================================================
json_folder = '/Users/valeturino/postdoc_local/seafloor_json_mapping_test'
dat_folder = '/Users/valeturino/postdoc_local/datasets'
nc_file = f'{dat_folder}/bathymetry_model_Feb_2025.nc'

os.makedirs(json_folder, exist_ok=True)

# ======================================================
# LOAD DATASET BOUNDS
# ======================================================
nc_ds = xr.open_dataset(nc_file, engine="netcdf4", chunks="auto")

lon_name = 'lon' if 'lon' in nc_ds.coords else 'longitude'
lat_name = 'lat' if 'lat' in nc_ds.coords else 'latitude'

# global_min_lon = float(nc_ds[lon_name].min())
# global_max_lon = float(nc_ds[lon_name].max())
# global_min_lat = float(nc_ds[lat_name].min())
# global_max_lat = float(nc_ds[lat_name].max())

global_min_lon =  -93
global_max_lon = -89
global_min_lat =  0
global_max_lat = 2.8

# Identify the bathymetry variable (e.g., 'z', 'elevation', 'topo')
data_var = [v for v in nc_ds.data_vars if len(nc_ds[v].dims) >= 2][0]

# ======================================================
# GRID CONFIGURATION (Using 20 and 2 for visualization)
# ======================================================
box_size = 3.0      #og 20.0
overlap = 0.2       # og 2.0
step = box_size - overlap  # 18.0 degree shift between origins

lon_starts = np.arange(global_min_lon, global_max_lon, step)
lat_starts = np.arange(global_min_lat, global_max_lat, step)

# ======================================================
# INITIALIZE MAP PLOT (No netCDF Background)
# ======================================================
fig = plt.figure(figsize=(12, 10), dpi=120)
ax = plt.axes(projection=ccrs.PlateCarree())
ax.set_extent([global_min_lon, global_max_lon, global_min_lat, global_max_lat], crs=ccrs.PlateCarree())

# Contrasting bold colors to emphasize overlapping intersections
colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#17becf']
count = 0
skipped_land_count = 0

# ======================================================
# ITERATE COMPUTE AND PLOT
# ======================================================
# Loop Latitudes (Rows -> A, B, C... Z, AA, AB...)
for i, start_lat in enumerate(lat_starts):
    if i < 26:
        row_letter = chr(65 + i)
    else:
        row_letter = chr(65 + (i // 26) - 1) + chr(65 + (i % 26))
        
    # Loop Longitudes (Columns -> 1, 2, 3...)
    for j, start_lon in enumerate(lon_starts):
        col_number = j + 1
        system_id = f"{row_letter}_{col_number}"
        
        max_lon = start_lon + box_size
        max_lat = start_lat + box_size
        
        # Clamp to dataset outer frame boundary
        if max_lon > global_max_lon: max_lon = global_max_lon
        if max_lat > global_max_lat: max_lat = global_max_lat
        
        # --------------------------------------------------
        # LAND / NAN FILTERING (80% Threshold)
        # --------------------------------------------------
        # Slice the dataset down to the current box region
        box_subset = nc_ds[data_var].sel(
            {lon_name: slice(start_lon, max_lon), lat_name: slice(start_lat, max_lat)}
        )
        
        # If the slice happens to be empty due to strict boundaries, skip it
        if box_subset.size == 0:
            continue
            
        # Count the total elements and the number of NaN elements inside this window
        nan_count = int(box_subset.isnull().sum().compute())
        total_count = box_subset.size
        nan_fraction = nan_count / total_count
        
        # If the tile area is 80% or more NaN (land), drop it entirely
        if nan_fraction >= 0.50:
            skipped_land_count += 1
            continue
            
        # --------------------------------------------------
        # VALID OCEAN BOX -> EXPORT AND PLOT
        # --------------------------------------------------
        # 1. Export JSON Data
        tile_data = {
            "system": system_id,
            "min_lon": round(float(start_lon), 4),
            "max_lon": round(float(max_lon), 4),
            "min_lat": round(float(start_lat), 4),
            "max_lat": round(float(max_lat), 4)
        }
        
        json_filename = f"{json_folder}/tile_{system_id}.json"
        with open(json_filename, 'w') as f:
            json.dump(tile_data, f, indent=2)
            
        # 2. Add Semi-Transparent Polygon To View Overlaps
        box_width = max_lon - start_lon
        box_height = max_lat - start_lat
        box_color = colors[(i + j) % len(colors)]
        
        # Alpha=0.4 stacks visibility up to ~0.8 in overlap strips
        rect = mpatches.Rectangle(
            (start_lon, start_lat), box_width, box_height,
            linewidth=1.2, edgecolor='black', facecolor=box_color, 
            alpha=0.4, transform=ccrs.PlateCarree()
        )
        ax.add_patch(rect)
        
        # Label each individual tile midpoint
        text_lon = start_lon + (box_width / 2.0)
        text_lat = start_lat + (box_height / 2.0)
        ax.text(text_lon, text_lat, system_id, color='black', weight='bold',
                fontsize=7, ha='center', va='center', transform=ccrs.PlateCarree(),
                bbox=dict(facecolor='white', alpha=0.7, edgecolor='none', pad=1))
        
        count += 1

print("Tiling finished")

# ======================================================
# GRAPHIC STYLING & EXPORT
# ======================================================
gl = ax.gridlines(draw_labels=True, dms=True, x_inline=False, y_inline=False, color='black', alpha=0.15, linestyle='-')
gl.top_labels = False
gl.right_labels = False

ax.coastlines(resolution='10m', color='black', linewidth=1.0)
plt.title(f"Tiling Schema: {count} Saved Ocean Tiles ({skipped_land_count} Land Tiles Dropped)", fontsize=13, pad=15)

plt.savefig(f"{json_folder}/_tiling_schema_only.png", bbox_inches='tight', dpi=150)
# plt.show()

print(f"Tiling check complete. {count} grid configurations successfully written to: {json_folder}")
print(f"Total tiles discarded due to land/NaN threshold: {skipped_land_count}")