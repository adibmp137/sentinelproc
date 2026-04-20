#!/usr/bin/env python3
"""
Soil Moisture Estimation from Sentinel-1 SAR Data
Following aanwarigeo method with modifications for in-situ sensor calibration
"""

import os
import re
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from glob import glob
import rasterio
from scipy import stats
from sklearn.linear_model import LinearRegression
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')

# ============================================================================
# CONFIGURATION
# ============================================================================
SAR_DIR = "c:/Users/Adib/Documents/Kuliah/Term 2/Thesis/sentinelproc/SAR_timeseries_masked"
INPUT_DIR = "c:/Users/Adib/Documents/Kuliah/Term 2/Thesis/sentinelproc/input"
OUTPUT_DIR = "c:/Users/Adib/Documents/Kuliah/Term 2/Thesis/sentinelproc/output"

GROWING_SEASON_START = datetime(2025, 7, 14)
GROWING_SEASON_END = datetime(2025, 9, 18)
TIME_TOLERANCE_HOURS = 24  # 1 day tolerance
NUM_BINS = 100
PERCENTILE = 0.98

# ============================================================================
# HELPER FUNCTIONS
# ============================================================================
def parse_date(filename):
    """Parse date from filename"""
    basename = os.path.basename(filename)
    match = re.search(r'(\d{4}-\d{2}-\d{2})_(\d{6})', basename)
    if match:
        date_str = match.group(1)
        time_str = match.group(2)
        return datetime.strptime(f"{date_str}_{time_str}", "%Y-%m-%d_%H%M%S")
    raise ValueError(f"Could not parse date from {filename}")

def remove_duplicate_dates(file_list):
    """Remove duplicate dates from different satellites"""
    seen_dates = set()
    unique_files = []
    for f in sorted(file_list):
        dt = parse_date(f)
        date_only = dt.date()
        if date_only not in seen_dates:
            seen_dates.add(date_only)
            unique_files.append(f)
    return unique_files

def load_tiff(filepath):
    """Load TIFF file"""
    with rasterio.open(filepath) as src:
        vv = src.read(1)
        vh = src.read(2) if src.count >= 2 else None
        mask = src.read(3) if src.count >= 3 else src.read(2)
        transform = src.transform
        crs = src.crs
    return vv, vh, mask, transform, crs

def load_category_data(tiff_list, label):
    """Load data for one category (irrigated or non-irrigated)"""
    data_dict = {}
    for tiff_path in sorted(tiff_list):
        dt = parse_date(tiff_path)
        vv, vh, mask, transform, crs = load_tiff(tiff_path)
        
        # Apply mask (keep where dataMask == 1)
        valid_mask = mask == 1
        
        # Convert to dB
        vv_linear = np.where(valid_mask, vv, np.nan)
        vv_db = 10 * np.log10(np.where(vv_linear > 0, vv_linear, np.nan))
        
        # Calculate DpRVIc if VH is available
        if vh is not None:
            vh_linear = np.where(valid_mask, vh, np.nan)
            q = vh_linear / vv_linear
            dprvic = np.where(valid_mask, q * (q + 3) / (q + 1)**2, np.nan)
        else:
            dprvic = None
        
        data_dict[dt] = {
            'vv': vv_db,
            'mask': valid_mask,
            'dprvic': dprvic,
            'transform': transform,
            'crs': crs
        }
        print(f"  Loaded {label}: {dt.strftime('%Y-%m-%d %H:%M')}")
    
    return data_dict

def stack_variables(data_dict, dates, var_name):
    """Stack all dates into a 3D array"""
    stacked = []
    dates_list = []
    for dt in dates:
        if dt in data_dict:
            stacked.append(data_dict[dt][var_name])
            dates_list.append(dt)
    return np.stack(stacked, axis=0), dates_list

# ============================================================================
# STEP 1: LOAD SAR DATA
# ============================================================================
print("="*60)
print("STEP 1: Loading SAR Data")
print("="*60)

# Find all ascending TIFFs
all_tiffs = glob(os.path.join(SAR_DIR, "*.tif"))
ascending_tiffs = [f for f in all_tiffs if "_ascending_" in f]
print(f"Found {len(ascending_tiffs)} ascending TIFF files")

# Separate into irrigated and non-irrigated and remove duplicates
irr_tiffs = [f for f in ascending_tiffs if "irrigated.tif" in f]
non_irr_tiffs = [f for f in ascending_tiffs if "nonirrigated.tif" in f]

irr_tiffs = remove_duplicate_dates(irr_tiffs)
non_irr_tiffs = remove_duplicate_dates(non_irr_tiffs)

print(f"Irrigated: {len(irr_tiffs)} unique dates")
print(f"Non-irrigated: {len(non_irr_tiffs)} unique dates")

print("\nLoading irrigated data...")
irr_data = load_category_data(irr_tiffs, "Irrigated")

print("\nLoading non-irrigated data...")
non_irr_data = load_category_data(non_irr_tiffs, "Non-irrigated")

# Get all unique dates
all_dates = sorted(set(list(irr_data.keys()) + list(non_irr_data.keys())))
print(f"\nTotal unique dates: {len(all_dates)}")

# ============================================================================
# STEP 3: WET & DRY REFERENCE CALIBRATION
# ============================================================================
print("\n" + "="*60)
print("STEP 3: Wet & Dry Reference Calibration")
print("="*60)

# Get stacks for both irrigated and non-irrigated areas
irr_stack, irr_dates = stack_variables(irr_data, all_dates, 'vv')
non_irr_stack, non_irr_dates = stack_variables(non_irr_data, all_dates, 'vv')
dprvic_stack_irr, _ = stack_variables(irr_data, irr_dates, 'dprvic')
dprvic_stack_nonirr, _ = stack_variables(non_irr_data, non_irr_dates, 'dprvic')

# Calculate dry reference for irrigated
print("\nStep 3a: Calculating dry reference (sigma_dry)...")
sigma_dry_irr = np.nanmin(irr_stack, axis=0)
sigma_dry_nonirr = np.nanmin(non_irr_stack, axis=0)
print(f"  Irrigated sigma_dry: min={np.nanmin(sigma_dry_irr):.2f}, max={np.nanmax(sigma_dry_irr):.2f} dB")
print(f"  Non-irrigated sigma_dry: min={np.nanmin(sigma_dry_nonirr):.2f}, max={np.nanmax(sigma_dry_nonirr):.2f} dB")

# Calculate backscatter change (delta = VV_dB - sigma_dry) for both areas
print("\nStep 3b: Calculating delta sigma (backscatter change)...")
delta_sigma_irr = irr_stack - sigma_dry_irr
delta_sigma_irr = np.maximum(delta_sigma_irr, 0)  # Clip negative values to 0
print(f"  Irrigated delta sigma: min={np.nanmin(delta_sigma_irr):.2f}, max={np.nanmax(delta_sigma_irr):.2f} dB")

delta_sigma_nonirr = non_irr_stack - sigma_dry_nonirr
delta_sigma_nonirr = np.maximum(delta_sigma_nonirr, 0)
print(f"  Non-irrigated delta sigma: min={np.nanmin(delta_sigma_nonirr):.2f}, max={np.nanmax(delta_sigma_nonirr):.2f} dB")

# Use irrigated for primary calibration (as per original approach)
delta_sigma = delta_sigma_irr
dprvic_stack = dprvic_stack_irr

# ============================================================================
# STEP 3d: UPPER ENVELOPE REGRESSION WITH DPRVIc
# ============================================================================
print("\nStep 3d: Upper envelope regression with DpRVIc...")

# Combine all valid pixel data
valid_mask = ~np.isnan(delta_sigma) & ~np.isnan(dprvic_stack)
all_delta = delta_sigma[valid_mask]
all_dprvic = dprvic_stack[valid_mask]

print(f"  Total valid pixels: {len(all_delta)}")

# Create bins
dprvic_min, dprvic_max = np.nanmin(all_dprvic), np.nanmax(all_dprvic)
bin_width = (dprvic_max - dprvic_min) / NUM_BINS
bin_indices = ((all_dprvic - dprvic_min) / bin_width).astype(int)
bin_indices = np.clip(bin_indices, 0, NUM_BINS - 1)

# Calculate 98th percentile for each bin
bin_thresholds = []
for b in range(NUM_BINS):
    mask = bin_indices == b
    if np.sum(mask) > 10:
        threshold = np.percentile(all_delta[mask], PERCENTILE * 100)
    else:
        threshold = np.nan
    bin_thresholds.append(threshold)
bin_thresholds = np.array(bin_thresholds)

# Get upper envelope points
upper_delta = []
upper_dprvic = []
for i, (d, dp) in enumerate(zip(all_delta, all_dprvic)):
    bi = bin_indices[i]
    if not np.isnan(bin_thresholds[bi]) and d >= bin_thresholds[bi]:
        upper_delta.append(d)
        upper_dprvic.append(dp)

upper_delta = np.array(upper_delta)
upper_dprvic = np.array(upper_dprvic)
print(f"  Upper envelope points: {len(upper_delta)}")

# Fit linear model
linear_model = np.polyfit(upper_dprvic, upper_delta, 1)
linear_pred = np.polyval(linear_model, upper_dprvic)
linear_r2 = 1 - np.sum((upper_delta - linear_pred)**2) / np.sum((upper_delta - np.mean(upper_delta))**2)
linear_rmse = np.sqrt(np.mean((upper_delta - linear_pred)**2))
n = len(upper_delta)
k_linear = 2  # number of parameters (a, b)
linear_sse = np.sum((upper_delta - linear_pred)**2)
linear_aic = n * np.log(linear_sse / n) + 2 * k_linear
linear_bic = n * np.log(linear_sse / n) + k_linear * np.log(n)

# Fit quadratic model
quadratic_model = np.polyfit(upper_dprvic, upper_delta, 2)
quadratic_pred = np.polyval(quadratic_model, upper_dprvic)
quadratic_r2 = 1 - np.sum((upper_delta - quadratic_pred)**2) / np.sum((upper_delta - np.mean(upper_delta))**2)
quadratic_rmse = np.sqrt(np.mean((upper_delta - quadratic_pred)**2))
k_quad = 3  # number of parameters (a, b, c)
quadratic_sse = np.sum((upper_delta - quadratic_pred)**2)
quadratic_aic = n * np.log(quadratic_sse / n) + 2 * k_quad
quadratic_bic = n * np.log(quadratic_sse / n) + k_quad * np.log(n)

print(f"\n  Linear model: delta_sigma = {linear_model[0]:.4f} x DpRVIc + {linear_model[1]:.4f}")
print(f"    R2: {linear_r2:.4f}, RMSE: {linear_rmse:.4f}, AIC: {linear_aic:.4f}, BIC: {linear_bic:.4f}")

print(f"\n  Quadratic model: delta_sigma = {quadratic_model[0]:.4f} x DpRVIc2 + {quadratic_model[1]:.4f} x DpRVIc + {quadratic_model[2]:.4f}")
print(f"    R2: {quadratic_r2:.4f}, RMSE: {quadratic_rmse:.4f}, AIC: {quadratic_aic:.4f}, BIC: {quadratic_bic:.4f}")

# Select best model using score formula: score = (R2 * 0.4) + (1 - normalized_AIC * 0.3) + (1 - normalized_RMSE * 0.3)
# Normalize AIC and RMSE across both models
aic_min = min(linear_aic, quadratic_aic)
aic_max = max(linear_aic, quadratic_aic)
rmse_min = min(linear_rmse, quadratic_rmse)
rmse_max = max(linear_rmse, quadratic_rmse)

linear_norm_aic = (linear_aic - aic_min) / (aic_max - aic_min + 1e-10)
linear_norm_rmse = (linear_rmse - rmse_min) / (rmse_max - rmse_min + 1e-10)
linear_score = (linear_r2 * 0.4) + ((1 - linear_norm_aic) * 0.3) + ((1 - linear_norm_rmse) * 0.3)

quad_norm_aic = (quadratic_aic - aic_min) / (aic_max - aic_min + 1e-10)
quad_norm_rmse = (quadratic_rmse - rmse_min) / (rmse_max - rmse_min + 1e-10)
quad_score = (quadratic_r2 * 0.4) + ((1 - quad_norm_aic) * 0.3) + ((1 - quad_norm_rmse) * 0.3)

print(f"\n  Linear model score: {linear_score:.4f}")
print(f"  Quadratic model score: {quad_score:.4f}")

# Select best model
if quad_score > linear_score:
    selected_model = quadratic_model
    model_type = "quadratic"
    print(f"\n*** Selected model: QUADRATIC (better score) ***")
else:
    selected_model = linear_model
    model_type = "linear"
    print(f"\n*** Selected model: LINEAR (better score) ***")

model_r2 = quadratic_r2 if model_type == "quadratic" else linear_r2

# Plot: Delta Backscatter vs DpRVIc with regression
try:
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    
    fig, ax = plt.subplots(figsize=(10, 8))
    fig.patch.set_facecolor('black')
    ax.set_facecolor('black')
    
    # Scatter all points ( downsampled for visibility)
    ax.scatter(all_dprvic[::10], all_delta[::10], 
              c='lightblue', s=10, alpha=0.3, label='All pixels', edgecolors='none')
    
    # Plot upper envelope points
    ax.scatter(upper_dprvic, upper_delta, 
               c='blue', s=30, alpha=0.7, label='Upper envelope (98th percentile)', edgecolors='white', linewidths=0.5)
    
    # Plot regression line
    dprvic_range = np.linspace(all_dprvic.min(), all_dprvic.max(), 100)
    if model_type == "quadratic":
        delta_pred = (selected_model[0] * dprvic_range**2 + 
                     selected_model[1] * dprvic_range + 
                     selected_model[2])
    else:
        delta_pred = selected_model[0] * dprvic_range + selected_model[1]
    
    ax.plot(dprvic_range, delta_pred, 'r-', linewidth=2, 
           label=f'Best model ({model_type}): R2={model_r2:.3f}')
    
    ax.set_xlabel('DpRVIc', color='white')
    ax.set_ylabel('Delta Backscatter (dB)', color='white')
    ax.set_title('Step 3d: Delta Backscatter vs DpRVIc with Upper Envelope Regression', color='white')
    ax.legend(facecolor='gray', edgecolor='white', labelcolor='white')
    ax.tick_params(colors='white')
    ax.spines['bottom'].set_color('white')
    ax.spines['left'].set_color('white')
    ax.spines['top'].set_color('black')
    ax.spines['right'].set_color('black')
    ax.grid(True, alpha=0.3, color='gray')
    plt.tight_layout()
    
    chart_path = os.path.join(OUTPUT_DIR, "regression_dprvic.png")
    plt.savefig(chart_path, dpi=150, facecolor='black')
    plt.close()
    print(f"\nSaved: {chart_path}")
except Exception as e:
    print(f"\nWarning: Could not generate regression chart: {e}")

# ============================================================================
# STEP 3f: SAVE INTERMEDIATE STACKS (one file per date in folder)
# ============================================================================
print("\nStep 3f: Saving intermediate stacks...")

# Get transform and CRS from first file
sample_tiff = irr_tiffs[0]
with rasterio.open(sample_tiff) as src:
    sample_transform = src.transform
    sample_crs = src.crs
    height = src.height
    width = src.width

# Create output folders
delta_folder = os.path.join(OUTPUT_DIR, "delta_backscatter")
dprvic_folder = os.path.join(OUTPUT_DIR, "dprvic")
os.makedirs(delta_folder, exist_ok=True)
os.makedirs(dprvic_folder, exist_ok=True)

# Save delta_backscatter per date for IRRIGATED
for i, dt in enumerate(irr_dates):
    date_str = dt.strftime("%Y%m%d")
    out_path = os.path.join(delta_folder, f"irr_{date_str}.tif")
    with rasterio.open(
        out_path, 'w',
        driver='GTiff',
        height=height,
        width=width,
        count=1,
        dtype=delta_sigma_irr.dtype,
        crs=sample_crs,
        transform=sample_transform
    ) as dst:
        dst.write(delta_sigma_irr[i], 1)
print(f"  Saved: {len(irr_dates)} irrigated delta_backscatter files")

# Save delta_backscatter per date for NON-IRRIGATED
for i, dt in enumerate(non_irr_dates):
    date_str = dt.strftime("%Y%m%d")
    out_path = os.path.join(delta_folder, f"nonirr_{date_str}.tif")
    with rasterio.open(
        out_path, 'w',
        driver='GTiff',
        height=height,
        width=width,
        count=1,
        dtype=delta_sigma_nonirr.dtype,
        crs=sample_crs,
        transform=sample_transform
    ) as dst:
        dst.write(delta_sigma_nonirr[i], 1)
print(f"  Saved: {len(non_irr_dates)} non-irrigated delta_backscatter files")

# Save dprvic per date for IRRIGATED (one file per date)
for i, dt in enumerate(irr_dates):
    date_str = dt.strftime("%Y%m%d")
    out_path = os.path.join(dprvic_folder, f"irr_{date_str}.tif")
    with rasterio.open(
        out_path, 'w',
        driver='GTiff',
        height=height,
        width=width,
        count=1,
        dtype=dprvic_stack_irr.dtype,
        crs=sample_crs,
        transform=sample_transform
    ) as dst:
        dst.write(dprvic_stack_irr[i], 1)
print(f"  Saved: {len(irr_dates)} irrigated dprvic files")

# Save dprvic per date for NON-IRRIGATED (one file per date)
for i, dt in enumerate(non_irr_dates):
    date_str = dt.strftime("%Y%m%d")
    out_path = os.path.join(dprvic_folder, f"nonirr_{date_str}.tif")
    with rasterio.open(
        out_path, 'w',
        driver='GTiff',
        height=height,
        width=width,
        count=1,
        dtype=dprvic_stack_nonirr.dtype,
        crs=sample_crs,
        transform=sample_transform
    ) as dst:
        dst.write(dprvic_stack_nonirr[i], 1)
print(f"  Saved: {len(non_irr_dates)} non-irrigated dprvic files")
print("\nStep 3e: Calculating relative SSM (Theta)...")

# Calculate delta_sigma_max for each pixel
if model_type == "quadratic":
    delta_sigma_max = (selected_model[0] * dprvic_stack**2 + 
                     selected_model[1] * dprvic_stack + 
                     selected_model[2])
else:
    delta_sigma_max = selected_model[0] * dprvic_stack + selected_model[1]

# Calculate Theta
Theta = delta_sigma / np.maximum(delta_sigma_max, 0.01)
Theta = np.clip(Theta, 0, 1)
print(f"  Theta range: [{np.nanmin(Theta):.4f}, {np.nanmax(Theta):.4f}]")

# ============================================================================
# STEP 3g: FARM-AVERAGED VALUES
# ============================================================================
print("\nStep 3g: Calculating farm-averaged values...")

# Calculate Theta for non-irrigated too
if model_type == "quadratic":
    delta_sigma_max_nonirr = (selected_model[0] * dprvic_stack_nonirr**2 + 
                            selected_model[1] * dprvic_stack_nonirr + 
                            selected_model[2])
else:
    delta_sigma_max_nonirr = selected_model[0] * dprvic_stack_nonirr + selected_model[1]

Theta_nonirr = delta_sigma_nonirr / np.maximum(delta_sigma_max_nonirr, 0.01)
Theta_nonirr = np.clip(Theta_nonirr, 0, 1)

results = []
for i, dt in enumerate(irr_dates):
    avg_theta = np.nanmean(Theta[i])
    avg_delta = np.nanmean(delta_sigma[i])
    avg_dprvic = np.nanmean(dprvic_stack[i])
    results.append({
        'date': dt,
        'Theta': avg_theta,
        'delta_sigma': avg_delta,
        'DpRVIc': avg_dprvic,
        'irrigation': 'irrigated'
    })

for i, dt in enumerate(non_irr_dates):
    avg_theta = np.nanmean(Theta_nonirr[i])
    avg_delta = np.nanmean(delta_sigma_nonirr[i])
    avg_dprvic = np.nanmean(dprvic_stack_nonirr[i])
    results.append({
        'date': dt,
        'Theta': avg_theta,
        'delta_sigma': avg_delta,
        'DpRVIc': avg_dprvic,
        'irrigation': 'non-irrigated'
    })

results_df = pd.DataFrame(results)
results_df['date_str'] = results_df['date'].apply(lambda x: x.strftime('%Y-%m-%d'))

print(results_df[['date_str', 'Theta', 'DpRVIc']])

# ============================================================================
# STEP 4: LOAD IN-SITU SENSOR DATA
# ============================================================================
print("\n" + "="*60)
print("STEP 4: Loading In-Situ Sensor Data")
print("="*60)

sensor_files_nonirr = ['Neeroeteren2_A1_1FDF6C4.csv', 'Neeroeteren2_A2_1FE02E4.csv', 'Neeroeteren2_B_C560E7.csv']
sensor_files_irr = ['Neeroeteren2_C1_1FDFA06.csv', 'Neeroeteren2_C2_1FDFC4E.csv', 'Neeroeteren2_D_1FDE64E.csv']

def load_sensor_data(filenames):
    all_data = []
    for fn in filenames:
        fp = os.path.join(INPUT_DIR, fn)
        if os.path.exists(fp):
            df = pd.read_csv(fp)
            df['datetime'] = pd.to_datetime(df['time_parsed'])
            df = df[(df['datetime'] >= GROWING_SEASON_START) & (df['datetime'] <= GROWING_SEASON_END)]
            
            vmc_cols = [c for c in df.columns if c.startswith('vmc')]
            if vmc_cols:
                df['vmc_avg'] = df[vmc_cols].mean(axis=1)
                df = df[df['vmc_avg'] > 0]
                all_data.append(df[['datetime', 'vmc_avg']])
                print(f"  Loaded {fn}: {len(df)} records")
    
    if all_data:
        combined = pd.concat(all_data)
        combined['hour'] = combined['datetime'].dt.floor('H')
        hourly = combined.groupby('hour')['vmc_avg'].mean().reset_index()
        return hourly
    return pd.DataFrame()

print("\nNon-irrigated sensors (A1+A2)...")
sensor_nonirr = load_sensor_data(sensor_files_nonirr)

print("\nIrrigated sensors (C1+C2)...")
sensor_irr = load_sensor_data(sensor_files_irr)

# ============================================================================
# STEP 5: MATCH SAR WITH IN-SITU
# ============================================================================
print("\n" + "="*60)
print("STEP 5: Matching SAR with In-Situ Data")
print("="*60)

matched_data = []

irr_results = results_df[results_df['irrigation'] == 'irrigated']
nonirr_results = results_df[results_df['irrigation'] == 'non-irrigated']

# Match irrigated area with irrigated sensors
for _, row in irr_results.iterrows():
    sar_date = row['date']
    if len(sensor_irr) > 0:
        sensor_copy = sensor_irr.copy()
        sensor_copy['time_diff'] = abs((sensor_copy['hour'] - sar_date).dt.total_seconds() / 3600)
        closest_idx = sensor_copy['time_diff'].idxmin()
        if sensor_copy.loc[closest_idx, 'time_diff'] <= TIME_TOLERANCE_HOURS:
            matched_data.append({
                'date': sar_date,
                'Theta_SAR': row['Theta'],
                'VMC_in_situ': sensor_copy.loc[closest_idx, 'vmc_avg'],
                'irrigation': 'irrigated'
            })

# Match non-irrigated area with non-irrigated sensors
for _, row in nonirr_results.iterrows():
    sar_date = row['date']
    if len(sensor_nonirr) > 0:
        sensor_copy = sensor_nonirr.copy()
        sensor_copy['time_diff'] = abs((sensor_copy['hour'] - sar_date).dt.total_seconds() / 3600)
        closest_idx = sensor_copy['time_diff'].idxmin()
        if sensor_copy.loc[closest_idx, 'time_diff'] <= TIME_TOLERANCE_HOURS:
            matched_data.append({
                'date': sar_date,
                'Theta_SAR': row['Theta'],
                'VMC_in_situ': sensor_copy.loc[closest_idx, 'vmc_avg'],
                'irrigation': 'non-irrigated'
            })

matched_df = pd.DataFrame(matched_data)
print(f"\nMatched {len(matched_df)} data points (irr={len(matched_df[matched_df['irrigation']=='irrigated'])} + nonirr={len(matched_df[matched_df['irrigation']=='non-irrigated'])})")

if len(matched_df) > 0:
    print(matched_df)
else:
    print("WARNING: No matches found - check time tolerance or sensor data")

# ============================================================================
# STEP 6: LINEAR REGRESSION & CALIBRATION
# ============================================================================
print("\n" + "="*60)
print("STEP 6: Linear Regression & Calibration")
print("="*60)

if len(matched_df) >= 2:
    from scipy.optimize import curve_fit
    
    irr_matched = matched_df[matched_df['irrigation'] == 'irrigated']
    nonirr_matched = matched_df[matched_df['irrigation'] == 'non-irrigated']
    
    results_summary = {}
    
    def linear_model(theta, a, b):
        return a * theta + b
    
    # Irrigated regression
    if len(irr_matched) >= 2:
        X_irr = irr_matched['Theta_SAR'].values
        y_irr = irr_matched['VMC_in_situ'].values
        
        popt, _ = curve_fit(linear_model, X_irr, y_irr, p0=[0.3, 0.02], bounds=([0, 0], [np.inf, np.inf]))
        a_irr, b_irr = popt
        y_pred_irr = linear_model(X_irr, a_irr, b_irr)
        r2_irr = 1 - np.sum((y_irr - y_pred_irr)**2) / np.sum((y_irr - np.mean(y_irr))**2)
        rmse_irr = np.sqrt(np.mean((y_irr - y_pred_irr)**2))
        
        irr_matched = irr_matched.copy()
        irr_matched['VSM_volumetric'] = b_irr + a_irr * irr_matched['Theta_SAR']
        
        results_summary['irr'] = {'a': a_irr, 'b': b_irr, 'r2': r2_irr, 'rmse': rmse_irr}
        
        print(f"\nIrrigated: VMC = {a_irr:.4f} x Theta + {b_irr:.4f}, R2={r2_irr:.4f}, RMSE={rmse_irr:.4f}")
    
    # Non-irrigated regression
    if len(nonirr_matched) >= 2:
        X_nonirr = nonirr_matched['Theta_SAR'].values
        y_nonirr = nonirr_matched['VMC_in_situ'].values
        
        popt, _ = curve_fit(linear_model, X_nonirr, y_nonirr, p0=[0.3, 0.02], bounds=([0, 0], [np.inf, np.inf]))
        a_nonirr, b_nonirr = popt
        y_pred_nonirr = linear_model(X_nonirr, a_nonirr, b_nonirr)
        r2_nonirr = 1 - np.sum((y_nonirr - y_pred_nonirr)**2) / np.sum((y_nonirr - np.mean(y_nonirr))**2)
        rmse_nonirr = np.sqrt(np.mean((y_nonirr - y_pred_nonirr)**2))
        
        nonirr_matched = nonirr_matched.copy()
        nonirr_matched['VSM_volumetric'] = b_nonirr + a_nonirr * nonirr_matched['Theta_SAR']
        
        results_summary['nonirr'] = {'a': a_nonirr, 'b': b_nonirr, 'r2': r2_nonirr, 'rmse': rmse_nonirr}
        
        print(f"\nNon-irrigated: VMC = {a_nonirr:.4f} x Theta + {b_nonirr:.4f}, R2={r2_nonirr:.4f}, RMSE={rmse_nonirr:.4f}")
    
    # Combine results
    matched_df = pd.concat([irr_matched, nonirr_matched], ignore_index=True)
    
    # Plot: separate regression charts for each area
    try:
        fig, ax = plt.subplots(figsize=(10, 8))
        fig.patch.set_facecolor('black')
        ax.set_facecolor('black')
        
        # Plot irrigated
        if 'irr' in results_summary:
            ax.scatter(irr_matched['Theta_SAR'], irr_matched['VMC_in_situ'], 
                      c='blue', s=80, alpha=0.8, label='Irrigated', edgecolors='white', linewidths=1)
            theta_range = np.linspace(0, 1, 100)
            vmc_pred = results_summary['irr']['a'] * theta_range + results_summary['irr']['b']
            ax.plot(theta_range, vmc_pred, 'b--', linewidth=2, 
                   label=f"Irrigated: VMC = {results_summary['irr']['a']:.4f}x + {results_summary['irr']['b']:.4f} (R2={results_summary['irr']['r2']:.3f})")
        
        # Plot non-irrigated
        if 'nonirr' in results_summary:
            ax.scatter(nonirr_matched['Theta_SAR'], nonirr_matched['VMC_in_situ'], 
                      c='green', s=80, alpha=0.8, label='Non-irrigated', edgecolors='white', linewidths=1)
            theta_range = np.linspace(0, 1, 100)
            vmc_pred = results_summary['nonirr']['a'] * theta_range + results_summary['nonirr']['b']
            ax.plot(theta_range, vmc_pred, 'g--', linewidth=2, 
                   label=f"Non-irrigated: VMC = {results_summary['nonirr']['a']:.4f}x + {results_summary['nonirr']['b']:.4f} (R2={results_summary['nonirr']['r2']:.3f})")
        
        ax.set_xlabel('Theta (SAR-derived)', color='white')
        ax.set_ylabel('VMC (in-situ)', color='white')
        ax.set_title('Step 6: VMC In-Situ vs Theta SAR with Separate Regressions', color='white')
        ax.legend(facecolor='gray', edgecolor='white', labelcolor='white')
        ax.tick_params(colors='white')
        ax.spines['bottom'].set_color('white')
        ax.spines['left'].set_color('white')
        ax.spines['top'].set_color('black')
        ax.spines['right'].set_color('black')
        ax.grid(True, alpha=0.3, color='gray')
        plt.tight_layout()
        
        chart_path = os.path.join(OUTPUT_DIR, "regression_calibration.png")
        plt.savefig(chart_path, dpi=150, facecolor='black')
        plt.close()
        print(f"\nSaved: {chart_path}")
    except Exception as e:
        print(f"\nWarning: Could not generate calibration chart: {e}")
    
    # Store for summary
    a = results_summary.get('irr', {}).get('a', 0)
    b = results_summary.get('irr', {}).get('b', 0)
    r2 = results_summary.get('irr', {}).get('r2', 0)
    
    if 'irr' in results_summary:
        min_VMC_irr = results_summary['irr']['b']
        max_VMC_irr = results_summary['irr']['a'] + results_summary['irr']['b']
        print(f"\nIrrigated Calibrated min: {min_VMC_irr:.4f} m3/m3")
        print(f"Irrigated Calibrated max: {max_VMC_irr:.4f} m3/m3")
    
    if 'nonirr' in results_summary:
        min_VMC_nonirr = results_summary['nonirr']['b']
        max_VMC_nonirr = results_summary['nonirr']['a'] + results_summary['nonirr']['b']
        print(f"\nNon-irrigated Calibrated min: {min_VMC_nonirr:.4f} m3/m3")
        print(f"Non-irrigated Calibrated max: {max_VMC_nonirr:.4f} m3/m3")
    
    print("\n" + "="*60)
    print("RESULTS SUMMARY")
    print("="*60)
    print(f"DpRVIc Model Selected: {model_type}")
    print(f"DpRVIc Model R2: {model_r2:.4f}")
    
    if 'irr' in results_summary:
        print(f"\nIrrigated Regression: VMC = {results_summary['irr']['a']:.4f} x Theta + {results_summary['irr']['b']:.4f}, R2={results_summary['irr']['r2']:.4f}")
    
    if 'nonirr' in results_summary:
        print(f"Non-irrigated Regression: VMC = {results_summary['nonirr']['a']:.4f} x Theta + {results_summary['nonirr']['b']:.4f}, R2={results_summary['nonirr']['r2']:.4f}")
else:
    print("Not enough matched data points for regression")
    a, b, r2 = 1, 0, 0

# ============================================================================
# STEP 8: OUTPUTS
# ============================================================================
print("\n" + "="*60)
print("STEP 8: Generating Outputs")
print("="*60)

# Save results
if len(matched_df) > 0:
    output_file = os.path.join(OUTPUT_DIR, "soil_moisture_timeseries.xlsx")
    try:
        matched_df.to_excel(output_file, index=False)
        print(f"Saved: {output_file}")
    except Exception as e:
        print(f"Warning: Could not save Excel file: {e}")
else:
    output_file = os.path.join(OUTPUT_DIR, "soil_moisture_timeseries.xlsx")
    try:
        results_df.to_excel(output_file, index=False)
        print(f"Saved: {output_file}")
    except Exception as e:
        print(f"Warning: Could not save Excel file: {e}")

# Generate separate validation charts for each area
try:
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    
    # Irrigated validation chart
    if len(sensor_irr) > 0:
        fig, ax = plt.subplots(figsize=(12, 6))
        ax.plot(sensor_irr['hour'], sensor_irr['vmc_avg'], 
                'b-', alpha=0.5, label='In-situ Irrigated', linewidth=1)
        
        irr_matched = matched_df[matched_df['irrigation'] == 'irrigated']
        if len(irr_matched) > 0:
            ax.scatter(irr_matched['date'], irr_matched['VSM_volumetric'], 
                      color='red', s=100, zorder=5, label='Satellite Irrigated', edgecolors='black')
        
        ax.set_xlabel('Date')
        ax.set_ylabel('Volumetric Soil Moisture (m3/m3)')
        ax.set_title('Validation: In-Situ vs Satellite-Derived Soil Moisture (Irrigated)')
        ax.legend()
        ax.grid(True, alpha=0.3)
        plt.xticks(rotation=45)
        plt.tight_layout()
        
        chart_path = os.path.join(OUTPUT_DIR, "validation_chart_irr.png")
        plt.savefig(chart_path, dpi=150)
        plt.close()
        print(f"Saved: {chart_path}")
    
    # Non-irrigated validation chart
    if len(sensor_nonirr) > 0:
        fig, ax = plt.subplots(figsize=(12, 6))
        ax.plot(sensor_nonirr['hour'], sensor_nonirr['vmc_avg'], 
                'g-', alpha=0.5, label='In-situ Non-irrigated', linewidth=1)
        
        nonirr_matched = matched_df[matched_df['irrigation'] == 'non-irrigated']
        if len(nonirr_matched) > 0:
            ax.scatter(nonirr_matched['date'], nonirr_matched['VSM_volumetric'], 
                      color='red', s=100, zorder=5, label='Satellite Non-irrigated', edgecolors='black')
        
        ax.set_xlabel('Date')
        ax.set_ylabel('Volumetric Soil Moisture (m3/m3)')
        ax.set_title('Validation: In-Situ vs Satellite-Derived Soil Moisture (Non-irrigated)')
        ax.legend()
        ax.grid(True, alpha=0.3)
        plt.xticks(rotation=45)
        plt.tight_layout()
        
        chart_path = os.path.join(OUTPUT_DIR, "validation_chart_nonirr.png")
        plt.savefig(chart_path, dpi=150)
        plt.close()
        print(f"Saved: {chart_path}")
except Exception as e:
    print(f"Warning: Could not generate validation chart: {e}")

# Save volumetric SSM map (mean across all dates)
try:
    vsm_volumetric = b + a * Theta
    vsm_mean = np.nanmean(vsm_volumetric, axis=0)
    
    vsm_path = os.path.join(OUTPUT_DIR, "volumetric_SSM.tif")
    with rasterio.open(
        vsm_path, 'w',
        driver='GTiff',
        height=vsm_mean.shape[0],
        width=vsm_mean.shape[1],
        count=1,
        dtype=vsm_mean.dtype,
        crs=sample_crs,
        transform=sample_transform
    ) as dst:
        dst.write(vsm_mean, 1)
    print(f"Saved: {vsm_path}")
except Exception as e:
    print(f"Warning: Could not generate volumetric SSM: {e}")

print("\n" + "="*60)
print("PROCESSING COMPLETE")
print("="*60)