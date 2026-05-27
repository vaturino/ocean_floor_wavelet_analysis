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


# ======== FUNCTIONS ========== #

import numpy as np

# ======================================================
# GLOBAL DECOUPLED COORDINATE TRANSFORMS
# ======================================================

def get_km_scale_factors(s, n):
    """
    Calculates the conversion factors (km per degree) for a given latitude band.
    """
    avg_lat = (s + n) / 2.0
    lat_deg_to_km = 111.1
    lon_deg_to_km = 111.1 * np.cos(np.radians(avg_lat))
    return lon_deg_to_km, lat_deg_to_km


def degrees_to_km_coordinates(lons, lats, s, n):
    """
    Transforms raw coordinate arrays (degrees) into physical distance tracks (km)
    relative to the bottom-left pixel origin of the dataset.
    
    Parameters:
        lons, lats: 1D numpy arrays or scalars representing coordinates in degrees.
        s, n: Scalars representing the minimum and maximum latitudes of the domain bounding box.
    """
    lon_deg_to_km, lat_deg_to_km = get_km_scale_factors(s, n)
    
    # Establish local origins using the first available pixel index
    lon_origin = lons[0] if isinstance(lons, (np.ndarray, list)) else lons
    lat_origin = lats[0] if isinstance(lats, (np.ndarray, list)) else lats
    
    lons_km = (lons - lon_origin) * lon_deg_to_km
    lats_km = (lats - lat_origin) * lat_deg_to_km
    
    return lons_km, lats_km


def get_grid_resolution_km(subset_lon, subset_lat, s, n):
    """
    Extracts the physical grid spacing step sizes (dlon, dlat) in kilometers.
    """
    lon_deg_to_km, lat_deg_to_km = get_km_scale_factors(s, n)
    
    dlon_km = abs(float(subset_lon[1] - subset_lon[0])) * lon_deg_to_km
    dlat_km = abs(float(subset_lat[1] - subset_lat[0])) * lat_deg_to_km
    
    return dlon_km, dlat_km

def get_hinged_topo_cmap(z_min, z_max, hinge_deep=-1500, hinge_land=0):
    """
    Creates a custom bathymetry/topography colormap with specific
    color transitions at z=hinge_deep and z=hinge_land.
    """
    # Handle cases where z_min and z_max are identical (flat data)
    if z_max == z_min:
        return mcolors.LinearSegmentedColormap.from_list('flat_color', [(0, 'gray'), (1, 'gray')])

    # Normalize hinges to the [0, 1] range of the colormap
    def norm(z): return (z - z_min) / (z_max - z_min)

    _hinge_deep = np.clip(hinge_deep, z_min, z_max)
    _hinge_land = np.clip(hinge_land, z_min, z_max)

    # Calculate normalized positions
    h1 = norm(_hinge_deep)
    h2 = norm(_hinge_land)

    if h1 > h2:
        h1, h2 = h2, h1 

    positions = []
    colors = []
    n_points_per_segment = 128 

    # Abyssal Segment (from 0.0 to h1)
    x_deep = np.linspace(0.0, h1, n_points_per_segment)
    colors_deep_arr = cmo.deep(np.linspace(0.2, 0.8, n_points_per_segment))
    positions.extend(x_deep)
    colors.extend(colors_deep_arr)

    # Shelf/Slope Segment (from h1 to h2)
    if h1 < h2: 
        x_shelf = np.linspace(h1, h2, n_points_per_segment)[1:] 
        colors_shelf_arr = cmo.ice(np.linspace(0.3, 0.9, n_points_per_segment))[1:] 
        positions.extend(x_shelf)
        colors.extend(colors_shelf_arr)

    # Land Segment (from h2 to 1.0)
    if h2 < 1.0: 
        x_land = np.linspace(h2, 1.0, n_points_per_segment)[1:] 
        colors_land_arr = cmo.topo(np.linspace(0.5, 1.0, n_points_per_segment))[1:] 
        positions.extend(x_land)
        colors.extend(colors_land_arr)

    # Combine positions and colors into a DataFrame
    combined_df = pd.DataFrame({'pos': positions, 'color': colors})
    combined_df.sort_values(by='pos', inplace=True)
    combined_df.drop_duplicates(subset=['pos'], keep='last', inplace=True)

    # ==============================================================================
    # CRITICAL FIX: Explicitly copy, snap, and force boundaries to eliminate float drift
    # ==============================================================================
    # Adding .copy() makes this array fully writable and independent of the DataFrame
    pos_array = combined_df['pos'].values.copy() 
    color_list = list(combined_df['color'].values)

    # Force strict alignment to 0.0 and 1.0 bounds safely now
    pos_array[0] = 0.0
    pos_array[-1] = 1.0

    # Safety fall-back if dataframe size collapsed unexpectedly
    if len(pos_array) < 2:
        return mcolors.LinearSegmentedColormap.from_list('fallback_cmap', [(0, 'gray'), (1, 'gray')])

    return mcolors.LinearSegmentedColormap.from_list('hinged_topo', list(zip(pos_array, color_list)))

# def get_hinged_topo_cmap(z_min, z_max, hinge_deep=-1500, hinge_land=0):
#     """
#     Creates a custom bathymetry/topography colormap with specific
#     color transitions at z=hinge_deep and z=hinge_land.
#     """
#     # Handle cases where z_min and z_max are identical (flat data)
#     if z_max == z_min:
#         return mcolors.LinearSegmentedColormap.from_list('flat_color', [(0, 'gray'), (1, 'gray')])

#     # Normalize hinges to the [0, 1] range of the colormap
#     def norm(z): return (z - z_min) / (z_max - z_min)

#     # 1. Adjust hinge points to be within the *actual* data range (z_min, z_max)
#     # This prevents normalized hinge values from going outside [0, 1]
#     _hinge_deep = np.clip(hinge_deep, z_min, z_max)
#     _hinge_land = np.clip(hinge_land, z_min, z_max)

#     # Calculate normalized positions
#     h1 = norm(_hinge_deep)
#     h2 = norm(_hinge_land)

#     # Ensure h1 <= h2. If original hinge_deep was > hinge_land AND they are both in range
#     # e.g., hinge_deep = -100, hinge_land = -500, then h1 > h2.
#     # The segments should still be ordered by normalized value.
#     if h1 > h2:
#         h1, h2 = h2, h1 # Swap if necessary to keep h1 first

#     # Prepare lists to collect all (position, color) points
#     positions = []
#     colors = []
#     n_points_per_segment = 128 # Resolution for each segment

#     # Abyssal Segment (from 0.0 to h1)
#     # Generate points for the deep segment (from 0 to h1)
#     x_deep = np.linspace(0.0, h1, n_points_per_segment)
#     colors_deep_arr = cmo.deep(np.linspace(0.2, 0.8, n_points_per_segment))
#     positions.extend(x_deep)
#     colors.extend(colors_deep_arr)

#     # Shelf/Slope Segment (from h1 to h2)
#     # Generate points for the shelf segment (from h1 to h2)
#     # We exclude the first point to avoid exact duplicates at `h1` as it's already added by x_deep's end.
#     # If h1 == h2, this segment will be empty after slicing.
#     if h1 < h2: # Only add if there's a distinct segment
#         x_shelf = np.linspace(h1, h2, n_points_per_segment)[1:] # [1:] to exclude start (h1)
#         colors_shelf_arr = cmo.ice(np.linspace(0.3, 0.9, n_points_per_segment))[1:] # [1:] corresponding colors
#         positions.extend(x_shelf)
#         colors.extend(colors_shelf_arr)

#     # Land Segment (from h2 to 1.0)
#     # Generate points for the land segment (from h2 to 1.0)
#     # We exclude the first point to avoid exact duplicates at `h2` as it's already added by x_shelf's end.
#     if h2 < 1.0: # Only add if there's a distinct segment
#         x_land = np.linspace(h2, 1.0, n_points_per_segment)[1:] # [1:] to exclude start (h2)
#         colors_land_arr = cmo.topo(np.linspace(0.5, 1.0, n_points_per_segment))[1:] # [1:] corresponding colors
#         positions.extend(x_land)
#         colors.extend(colors_land_arr)

#     # Combine positions and colors into a DataFrame for sorting and duplicate removal
#     combined_df = pd.DataFrame({'pos': positions, 'color': colors})

#     # Sort by position to ensure monotonicity
#     combined_df.sort_values(by='pos', inplace=True)

#     # Remove duplicates based on 'pos', keeping the last color encountered at that position
#     # This is crucial for fixing 'x in increasing order' error
#     combined_df.drop_duplicates(subset=['pos'], keep='last', inplace=True)

#     # Ensure colormap starts exactly at 0.0 and ends exactly at 1.0
#     if combined_df['pos'].iloc[0] > 0.0:
#         # Ensure the color at 0.0 is explicitly defined. If not, use the first color defined.
#         first_color = combined_df['color'].iloc[0] if not combined_df.empty else 'gray'
#         combined_df = pd.concat([pd.DataFrame([{'pos': 0.0, 'color': first_color}]), combined_df]).reset_index(drop=True)
#     if combined_df['pos'].iloc[-1] < 1.0:
#         # Ensure the color at 1.0 is explicitly defined. If not, use the last color defined.
#         last_color = combined_df['color'].iloc[-1] if not combined_df.empty else 'gray'
#         combined_df = pd.concat([combined_df, pd.DataFrame([{'pos': 1.0, 'color': last_color}]).reset_index(drop=True)])


#     # If, after processing, we have fewer than 2 unique points, return a fallback colormap
#     if len(combined_df) < 2:
#         return mcolors.LinearSegmentedColormap.from_list('fallback_cmap', [(0, 'gray'), (1, 'gray')])

#     return mcolors.LinearSegmentedColormap.from_list('hinged_topo', list(zip(combined_df['pos'], combined_df['color'])))

def wrap_lon(lon):
    return (lon + 180) % 360 - 180

def prepare_grid_and_data(subset, w, e, s, n, Lcomp):
    if subset is None: return None

    # Standardize matrix orientation strictly to [Lat, Lon]
    Z = subset.values.astype(np.float32)
    if subset.dims[0] != 'lat': Z = Z.T
    ny, nx = Z.shape

    # CALCULATE METRIC DISTANCES (Exactly Once)
    avg_lat = (s + n) / 2.0
    lat_deg_to_km = 111.1
    lon_deg_to_km = 111.1 * np.cos(np.radians(avg_lat))
    
    # Calculate the true delta distance step per pixel matrix element
    dx_km = abs(float(subset.lon[1] - subset.lon[0])) * lon_deg_to_km
    dy_km = abs(float(subset.lat[1] - subset.lat[0])) * lat_deg_to_km

    # Detrend and clear out any NaN padding
    wavefield = np.nan_to_num(Z - np.nanmean(Z))
    wavefield = wavefield.astype(np.float32)
    max_bathy = np.nanmax(np.abs(wavefield)) if np.any(wavefield) else 1.0

    return {
        'wavefield': wavefield,
        'wavefield_raw': Z,
        'max_bathy': max_bathy,
        'nx': nx, 'ny': ny,
        'dx_phys': dx_km,        # Handed directly to wavelet transform module
        'dy_phys': dy_km,        # Handed directly to wavelet transform module
        'x_max_norm': (dx_km * (nx - 1)) / Lcomp,
        'y_max_norm': (dy_km * (ny - 1)) / Lcomp,
        'bounds': (w, e, s, n)
    }

def plot_geographic_preview(data_dict, system_name, figure_folder):
    """
    Creates a publication-quality bathymetry preview with dual axes:
    Primary: Geodetic (Lon/Lat) | Secondary: Normalized (x/Lcomp)
    """
    Z = data_dict['wavefield_raw'] # Using raw values for bathymetry colors
    w, e, s, n = data_dict['bounds']
    xn_max, yn_max = data_dict['x_max_norm'], data_dict['y_max_norm']

    # Coordinate mapping functions for secondary axes
    def lon_to_norm(lon): return (lon - w) / (e - w) * xn_max
    def norm_to_lon(xn):  return w + (xn / xn_max) * (e - w)
    def lat_to_norm(lat): return (lat - s) / (n - s) * yn_max
    def norm_to_lat(yn):  return s + (yn / yn_max) * (n - s)

    fig, ax = plt.subplots(figsize=(10, 8))

    # Use terrain or hinged topo if available
    z_min, z_max = np.nanmin(Z), np.nanmax(Z)
    try:
        cmap = get_hinged_topo_cmap(z_min, z_max)
    except NameError:
        cmap = plt.get_cmap('terrain')

    # Plot in Geodetic Space
    im = ax.imshow(Z, extent=[w, e, s, n], cmap=cmap,
                   vmin=z_min, vmax=z_max, origin='lower')
    ax.set_aspect('equal')

    # Colorbar with Geological Labels
    cb = fig.colorbar(im, ax=ax, orientation='horizontal', fraction=0.05, pad=0.15)
    cb.set_label("Elevation (m)")
    for b, label in [(-1500, "Abyssal"), (-200, "Slope"), (0, "Land")]:
        if z_min < b < z_max:
            cb.ax.axvline(b, color='white', lw=1)
            cb.ax.text(b, 1.2, label, ha='center', va='bottom',
                       transform=cb.ax.get_xaxis_transform(), fontsize=9, fontweight='bold')

    # Dual Axes Setup
    ax.set_title(f"Bathymetry: {system_name}", pad=35, fontweight='bold')
    ax.set_xlabel("Longitude")
    ax.set_ylabel("Latitude")

    # Add the Normalized (x/Lcomp) axes on Top and Right
    secax_x = ax.secondary_xaxis('top', functions=(lon_to_norm, norm_to_lon))
    secax_x.set_xlabel("$x/L_{comp}$", labelpad=10)
    secax_y = ax.secondary_yaxis('right', functions=(lat_to_norm, norm_to_lat))
    secax_y.set_ylabel("$y/L_{comp}$", labelpad=10)

    plt.tight_layout()
    # Extract the parent directory tree path to save in the system_name folder directly
    parent_folder = os.path.dirname(figure_folder)
    fig.savefig(os.path.join(parent_folder, "00_Bathymetry_Preview.pdf"), dpi=300)
    plt.close(fig) # Adding to cleanly manage memory
     

# ======================================================
# 2. ANALYSIS (Isotropic Pixel-Space Transform)
# ======================================================

def compute_cwt2d(data_dict, Lcomp):
    """
    CLEAN MATHEMATICAL TRANSFORM: Consumes pre-computed kilometer metrics 
    directly, correcting the 1D and 2D peak drift.
    """
    import numpy as np
    from scipy.signal import find_peaks
    from juwavelet import transform

    # Physical initialization parameter matches baseline notebook
    s0 = 2.0 * data_dict['dx_phys']  
    lambda_max_x = data_dict['nx'] * data_dict['dx_phys']
    lambda_max_y = data_dict['ny'] * data_dict['dy_phys']
    lambda_max = max(lambda_max_x, lambda_max_y)
    
    dj, jt = 1/16, 18
    js = int(np.log2(lambda_max / s0) / dj) + 1

    # Execute transformation mapping geodetic aspect ratio stretching cleanly
    cwt = transform.decompose2d(
        data_dict['wavefield'], data_dict['dx_phys'], data_dict['dy_phys'], s0, dj, js, jt,
        aspect=data_dict['dy_phys']/data_dict['dx_phys'], nxpad=None, nypad=None,
        opts={'param':  2*np.pi}, mode='scaled', dtype=np.complex128
    )

    # FIX: Output scale units are already physical.
    # We divide by Lcomp directly, preventing the 1.78x compounding shift!
    wavelengths_km = np.asarray(cwt['period'])
    wavelengths_norm = wavelengths_km / Lcomp

    decomp = cwt['decomposition']
    
    # 1D Power Curve tracking
    power = np.sum(np.abs(decomp), axis=(1, 2, 3))
    power_norm = power / np.max(power)
    peaks_idx, _ = find_peaks(power_norm, height=0.05)

    return decomp, wavelengths_norm, peaks_idx, power_norm, cwt



# ======================================================
# 3. RECONSTRUCTION (Anti-Aliased sum)
# ======================================================

def plot_reconstruction(i, idx, decomp, data_dict, wl_norm, figure_folder, peak_theta=None, Lcomp =None):

    Z = data_dict['wavefield_raw'] # Using raw values for bathymetry colors
    w, e, s, n = data_dict['bounds']
    xn_max, yn_max = data_dict['x_max_norm'], data_dict['y_max_norm']

    # Coordinate mapping functions for secondary axes
    def lon_to_norm(lon): return (lon - w) / (e - w) * xn_max
    def norm_to_lon(xn):  return w + (xn / xn_max) * (e - w)
    def lat_to_norm(lat): return (lat - s) / (n - s) * yn_max
    def norm_to_lat(yn):  return s + (yn / yn_max) * (n - s)

    # BROADBAND SUMMATION: Summing neighbors eliminates 'sine wave' artifacts
    indices = [idx-1, idx, idx+1]
    weights = [0.5, 1.0, 0.5] # Gaussian kernel in scale-space

    recon_complex = np.zeros((data_dict['ny'], data_dict['nx']), dtype=np.complex64)
    for s_idx, weight in zip(indices, weights):
        if 0 <= s_idx < decomp.shape[0]:
            recon_complex += weight * np.sum(decomp[s_idx], axis=0)

    recon = np.real(recon_complex)
    amp = np.abs(recon_complex)
    ori = np.angle(recon_complex)

    recon /= np.nanmax(np.abs(recon))

    # Coordinate mesh for plotting
    Xn, Yn = np.meshgrid(np.linspace(0, data_dict['x_max_norm'], data_dict['nx']),
                         np.linspace(0, data_dict['y_max_norm'], data_dict['ny']))

    fig, axes = plt.subplots(2, 2, figsize=(14, 14))
    axes = axes.flatten()
    cbar_opt = {'orientation': 'horizontal', 'fraction': 0.05, 'pad': 0.2, 'aspect': 30}

    # Panel 0: Input
    im0 = axes[0].pcolormesh(Xn, Yn, data_dict['wavefield']/data_dict['max_bathy'], cmap='RdBu_r', vmin=-1, vmax=1, shading='auto', rasterized=True)
    fig.colorbar(im0, ax=axes[0], **cbar_opt, label=r'$Z / Z_{\max}$')


    # Panel 1: Normalized Reconstruction (The 'Clean' Seafloor)
    vmax = np.nanmax(np.abs(recon))
    im1 = axes[1].pcolormesh(Xn, Yn, recon, cmap='RdBu_r', vmin=-1, vmax=1, shading='auto', rasterized=True)
    fig.colorbar(im1, ax=axes[1], **cbar_opt, label='Norm. Amplitude')

    # Panel 2: Energy Density (Amplitude)
    im2 = axes[2].contourf(Xn, Yn, amp, levels=20, cmap='magma', rasterized=True)
    fig.colorbar(im2, ax=axes[2], **cbar_opt, label='Absolute Amplitude')

    # Panel 3: Phase with Alpha-Masking
    amp_alpha = amp / np.nanmax(amp)
    cmap_tw = plt.get_cmap('twilight')
    norm_tw = mcolors.Normalize(vmin=-np.pi, vmax=np.pi)
    rgba_img = cmap_tw(norm_tw(ori))
    rgba_img[..., 3] = amp_alpha
    rgba = plt.get_cmap('twilight')(mcolors.Normalize(vmin=-np.pi, vmax=np.pi)(ori))
    rgba[..., 3] = amp / (np.nanmax(amp) + 1e-10) # Stronger signals are more opaque
    axes[3].imshow(rgba, extent=[0, data_dict['x_max_norm'], 0, data_dict['y_max_norm']], origin='lower')
    sm = plt.cm.ScalarMappable(cmap=cmap_tw, norm=norm_tw)
    fig.colorbar(sm, ax=axes[3], **cbar_opt, label='Phase Angle (rad)')

    # Update Title Logic
    if peak_theta is not None:
        # 2D Mode title
        main_title = fr'Peak {i+1}: $\lambda/L_c={wl_norm:.2f}$ | $\theta={peak_theta:.1f}^\circ$'
    else:
        # 1D Mode title (Orientation is unknown/collapsed)
        main_title = fr'Peak {i+1}: $\lambda/L_c={wl_norm:.2f}$'

    # Apply to the reconstruction panel (Panel 1)
    titles = ['Original Bathymetry', main_title, 'Amplitude', 'Phase']


    for j, ax in enumerate(axes):
        ax.set_title(titles[j], fontweight='bold')
        ax.set_aspect('equal')
        ax.set_xlabel('x/Lcomp')
        ax.set_ylabel('y/Lcomp')

        # Geographic Twin Axes (Secondary)
        secax_x_recon = ax.secondary_xaxis('top', functions=(norm_to_lon, lon_to_norm))
        secax_x_recon.set_xlabel(r'Longitude ($^\circ$E)', labelpad=5)
        secax_y_recon = ax.secondary_yaxis('right', functions=(norm_to_lat, lat_to_norm))
        secax_y_recon.set_ylabel(r'Latitude ($^\circ$N)', labelpad=5)

    if Lcomp is not None:
        axes[0].set_xlabel(f'x/Lcomp -- Lcomp = {Lcomp} km')

    # plt.tight_layout()
    plt.subplots_adjust(wspace=0.3, hspace=0.4)
    fig.savefig(os.path.join(figure_folder, f"peak_{i+1}_final.pdf"))
    
    plt.close(fig)





############################
##### 2D power spectrum ####
############################
import matplotlib.patheffects as path_effects

def find_peaks_2d_gmm(power_2d, angles_deg, wl_norm, n_components=5):
    """
    Directly fits GMM to the 2D energy distribution (Raw Scale-Angle space).
    Returns the (wavelength, angle) coordinates of distinct energy 'blobs'.
    """
    import numpy as np
    from sklearn.mixture import GaussianMixture

    # 1. Create a coordinate grid [Angle, Wavelength]
    A_mesh, W_mesh = np.meshgrid(angles_deg, wl_norm)

    # 2. Extract all pixels and their corresponding power values
    # Flatten everything into a list of [angle, wavelength] points
    coords = np.column_stack((A_mesh.ravel(), W_mesh.ravel()))
    weights = power_2d.ravel()

    # 3. Clean the data (ignore zero-power areas to speed up fitting)
    mask = weights > (0.05 * np.max(weights)) # Keep top 95% of energy
    clean_coords = coords[mask]
    clean_weights = weights[mask]

    # 4. Resample to create a "Point Cloud" where density = power
    # This is the "Raw Data" approach for GMM
    n_samples = 15000
    prob = clean_weights / np.sum(clean_weights)
    resampled_indices = np.random.choice(len(clean_coords), size=n_samples, p=prob)
    data_cloud = clean_coords[resampled_indices]

    # 5. Fit the GMM
    gmm = GaussianMixture(n_components=n_components, covariance_type='full', random_state=42)
    gmm.fit(data_cloud)

    # Returns: [Angle Center, Wavelength Center], Covariance Matrix
    return gmm.means_, gmm.covariances_

def plot_raw_2d_gmm_analysis(figure_folder, power_2d, angles_deg, wl_norm, system_name):
    import matplotlib.patheffects as path_effects
    import matplotlib.patches as patches

    # 1. Run the Raw Detection
    means, covs = find_peaks_2d_gmm(power_2d, angles_deg, wl_norm)

    fig, ax = plt.subplots(figsize=(11, 7))

    # Plot the Raw 2D Spectrum
    norm_power = power_2d / np.max(power_2d)
    im = ax.pcolormesh(angles_deg, wl_norm, norm_power, cmap='magma', shading='auto', rasterized=True)

    # 2. Draw the GMM results
    found_peaks = []
    for i, (mean, cov) in enumerate(zip(means, covs)):
        theta_peak, wl_peak = mean

        # Filter: Only keep peaks within the actual data range
        if (angles_deg.min() <= theta_peak <= angles_deg.max()) and \
           (wl_norm.min() <= wl_peak <= wl_norm.max()):

            # Draw an X at the GMM center
            ax.scatter(theta_peak, wl_peak, color='cyan', marker='x', s=100, zorder=10)

            # Add Label
            label_text = f"Peak {i+1}\n" + fr"$\lambda$: {wl_peak:.2f}" + "\n" + fr"$\theta$: {theta_peak:.1f}°"
            
            txt = ax.text(theta_peak, wl_peak, label_text, 
                          color='white', fontweight='bold', fontsize=9, va='bottom')
            txt.set_path_effects([path_effects.withStroke(linewidth=2, foreground='black')])

            # Store for reconstruction
            found_peaks.append({'wl': wl_peak, 'theta': theta_peak})

            # Optional: Draw the 1-Sigma Ellipse
            v, w = np.linalg.eigh(cov)
            angle = np.degrees(np.arctan2(w[0,1], w[0,0]))
            ell = patches.Ellipse(xy=(theta_peak, wl_peak), width=2*np.sqrt(v[0]), height=2*np.sqrt(v[1]),
                                  angle=angle, color='cyan', fc='none', lw=1, ls='--', alpha=0.5)
            ax.add_patch(ell)

    ax.set_title(f"Raw 2D Wavelet Energy Distribution (GMM Peaks): {system_name}", fontweight='bold')
    ax.set_xlabel(r"Orientation $\theta$ (Degrees)")
    ax.set_ylabel(r"Normalized Wavelength $\lambda / L_{comp}$")
    plt.colorbar(im, label="Normalized Power")

    fig.savefig(os.path.join(figure_folder, "02_Raw_2D_GMM_Peaks.pdf"))
    # 

    return found_peaks

# ======================================================
# 4. MAIN ORCHESTRATOR
# ======================================================


def run_wavelet_analysis(figure_folder_1d, figure_folder_2d, subset, w, e, s, n, Lcomp, system_name, jt=18):
    print(f"Running Analysis: {system_name}")
    
    # Ensure both target directories exist
    os.makedirs(figure_folder_1d, exist_ok=True)
    os.makedirs(figure_folder_2d, exist_ok=True)
    
    data_dict = prepare_grid_and_data(subset, w, e, s, n, Lcomp)
    if data_dict is None: return

    # Save the bathymetry preview at the base level or 1D folder
    plot_geographic_preview(data_dict, system_name, figure_folder_1d)
    decomp, wl_norm, peaks_1d, power_curve, cwt = compute_cwt2d(data_dict, Lcomp)

    # --- Plot 1D Spectrum ---
    fig_pow, ax_p = plt.subplots(figsize=(10, 4))
    ax_p.plot(wl_norm, power_curve, 'k-', lw=2, label='Global Power')
    ax_p.scatter(wl_norm[peaks_1d], power_curve[peaks_1d], color='red', label='Peaks')

    # Annotate peaks
    for wl, yval in zip(wl_norm[peaks_1d], power_curve[peaks_1d]):
        ax_p.annotate(f'{wl:.2f}', (wl, yval), xytext=(0,6),
                    textcoords='offset points', ha='center', color='red', weight='bold')

    ax_p.set_xlabel(r'Wavelength $\lambda / L_{\rm comp}$', labelpad=5)
    ax_p.set_ylabel(r'Power $P / P_{\max}$', labelpad=5)
    ax_p.grid(True, linestyle='--', alpha=0.7)
    ax_p.legend()
    ax_p.set_title(f"Global Power Spectrum: {system_name}")

    # ROUTE DIRECTLY TO 1D_ANALYSIS FOLDER
    fig_pow.savefig(os.path.join(figure_folder_1d, "00_global_power.pdf"), bbox_inches='tight')
    plt.close(fig_pow)

    # --- Plot 2D Spectrum & Run GMM ---
    power_2d = np.sum(np.abs(decomp), axis=(2, 3))
    power_2d_norm = power_2d / np.max(power_2d)
    angles_deg = cwt['angle'] * (180.0 / np.pi) if 'angle' in cwt else np.linspace(0, 180, jt, endpoint=False)

    # ROUTE DIRECTLY TO 2D_ANALYSIS FOLDER
    gmm_peaks = plot_raw_2d_gmm_analysis(figure_folder_2d, power_2d_norm, angles_deg, wl_norm, system_name)

    # Return dictionaries containing references to both destination environments
    return {
        'decomp': decomp, 'wl_norm': wl_norm, 'peaks_1d': peaks_1d,
        'gmm_peaks': gmm_peaks, 'data_dict': data_dict,
        'power_2d': power_2d, 'angles_deg': angles_deg
    }



def execute_reconstruction(choice, analysis_results, sys_name_inp: str):
    res = analysis_results
    print(f"Starting reconstruction using {choice} peaks...")
    Lcomp = res.get('Lcomp', None)

    if choice == '1D Spectrum':
        for i, idx in enumerate(res['peaks_1d']):
            # Pass the exact wavelength from the 1D spectrum array
            wl_val = res['wl_norm'][idx]
            plot_reconstruction(i, idx, res['decomp'], res['data_dict'],
                                wl_val, res['figure_folder'], Lcomp=Lcomp)

    else:
        # 2D GMM Spectrum
        for i, peak in enumerate(res['gmm_peaks']):
            target_wl = peak['wl']
            target_theta = peak['theta']

            # Snap to nearest wavelength index for the math
            idx = (np.abs(res['wl_norm'] - target_wl)).argmin()

            # Pass the snapped index but the ORIGINAL GMM coordinates for the title
            plot_reconstruction(i, idx, res['decomp'], res['data_dict'],
                                target_wl, res['figure_folder'], peak_theta=target_theta, Lcomp=Lcomp)

    # Move zip logic here so it includes the new plots
    zip_name = f"{sys_name_inp.value}_results"
    import shutil
    shutil.make_archive(zip_name, 'zip', res['figure_folder'])
    print(f"Done! All plots saved in {res['figure_folder']}")




