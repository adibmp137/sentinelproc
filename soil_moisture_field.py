import os
import re
import json
import numpy as np
import pandas as pd
from datetime import datetime
from glob import glob
import rasterio
import geopandas as gpd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
matplotlib.rcParams.update({
    'font.family': 'serif',
    'mathtext.fontset': 'dejavuserif',
    'font.size': 18,
})
import warnings
warnings.filterwarnings('ignore')

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
SAR_DIR = os.path.join(BASE_DIR, "SAR_timeseries_normalized")
OUTPUT_DIR = os.path.join(BASE_DIR, "output")

GROWING_SEASON_START = datetime(2025, 4, 24)
GROWING_SEASON_END = datetime(2025, 6, 5)
ORBIT_FILTER = None
NUM_BINS = 100
PERCENTILE = 0.98

# Configure with actual sensor UTM coordinates for your study area
ZONE_CENTERS_UTM = {
    'MZ1': (0.0, 0.0),
    'MZ2': (0.0, 0.0),
}
BOX_HALF = 35  # meters -> 70x70m box (7x7 pixels at 10m)


def parse_date(filename):
    basename = os.path.basename(filename)
    match = re.search(r'(\d{4}-\d{2}-\d{2})_(\d{6})', basename)
    if match:
        date_str = match.group(1)
        time_str = match.group(2)
        return datetime.strptime(f"{date_str}_{time_str}", "%Y-%m-%d_%H%M%S")
    raise ValueError(f"Could not parse date from {filename}")


def remove_duplicate_dates(file_list):
    seen_dates = set()
    unique_files = []
    for f in sorted(file_list):
        dt = parse_date(f)
        date_only = dt.date()
        if date_only not in seen_dates:
            seen_dates.add(date_only)
            unique_files.append(f)
    return unique_files


def make_zone_mask(transform, height, width, center_x, center_y, half_size):
    col_min = int(np.floor((center_x - half_size - transform.c) / transform.a))
    col_max = int(np.ceil((center_x + half_size - transform.c) / transform.a))
    row_min = int(np.floor((center_y + half_size - transform.f) / transform.e))
    row_max = int(np.ceil((center_y - half_size - transform.f) / transform.e))

    col_min = max(col_min, 0)
    col_max = min(col_max, width)
    row_min = max(row_min, 0)
    row_max = min(row_max, height)

    mask_arr = np.zeros((height, width), dtype=bool)
    mask_arr[row_min:row_max, col_min:col_max] = True
    return mask_arr


def fit_regression(delta_sigma, dprvic_stack, zone_name):
    valid_mask = ~np.isnan(delta_sigma) & ~np.isnan(dprvic_stack)
    all_delta = delta_sigma[valid_mask]
    all_dprvic = dprvic_stack[valid_mask]
    print(f"  Valid pixels: {len(all_delta)}")

    if len(all_delta) < 10:
        print(f"  WARNING: Too few valid pixels for regression ({len(all_delta)})")
        return np.array([0, 1]), "linear", 0.0, all_dprvic, all_delta, all_dprvic, all_delta

    dprvic_min, dprvic_max = np.nanmin(all_dprvic), np.nanmax(all_dprvic)
    bin_width = (dprvic_max - dprvic_min) / NUM_BINS
    bin_indices = ((all_dprvic - dprvic_min) / bin_width).astype(int)
    bin_indices = np.clip(bin_indices, 0, NUM_BINS - 1)

    bin_thresholds = []
    for b in range(NUM_BINS):
        mask = bin_indices == b
        if np.sum(mask) > 10:
            threshold = np.percentile(all_delta[mask], PERCENTILE * 100)
        else:
            threshold = np.nan
        bin_thresholds.append(threshold)
    bin_thresholds = np.array(bin_thresholds)

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

    if len(upper_delta) < 3:
        print(f"  WARNING: Too few upper envelope points ({len(upper_delta)}), using linear fallback")
        linear_model = np.polyfit(all_dprvic, all_delta, 1)
        return linear_model, "linear", 0.0, all_dprvic, all_delta, all_dprvic, all_delta

    linear_model = np.polyfit(upper_dprvic, upper_delta, 1)
    linear_pred = np.polyval(linear_model, upper_dprvic)
    linear_r2 = 1 - np.sum((upper_delta - linear_pred)**2) / np.sum((upper_delta - np.mean(upper_delta))**2)
    linear_rmse = np.sqrt(np.mean((upper_delta - linear_pred)**2))
    n = len(upper_delta)
    linear_sse = np.sum((upper_delta - linear_pred)**2)
    linear_aic = n * np.log(linear_sse / n) + 2 * 2

    dprvic_sq = upper_dprvic**2
    A_constrained = np.vstack([dprvic_sq, np.ones(len(upper_dprvic))]).T
    constrained_coeffs, _, _, _ = np.linalg.lstsq(A_constrained, upper_delta, rcond=None)
    quad_a, quad_c = constrained_coeffs
    quadratic_pred = quad_a * dprvic_sq + quad_c
    quadratic_model = np.array([quad_a, 0.0, quad_c])
    quadratic_r2 = 1 - np.sum((upper_delta - quadratic_pred)**2) / np.sum((upper_delta - np.mean(upper_delta))**2)
    quadratic_rmse = np.sqrt(np.mean((upper_delta - quadratic_pred)**2))
    quadratic_sse = np.sum((upper_delta - quadratic_pred)**2)
    quadratic_aic = n * np.log(quadratic_sse / n) + 2 * 2

    print(f"\n  Linear: delta = {linear_model[0]:.4f}*DpRVIc + {linear_model[1]:.4f}")
    print(f"    R2={linear_r2:.4f}, RMSE={linear_rmse:.4f}, AIC={linear_aic:.4f}")
    print(f"\n  Quadratic: delta = {quad_a:.4f}*DpRVIc^2 + {quad_c:.4f}")
    print(f"    R2={quadratic_r2:.4f}, RMSE={quadratic_rmse:.4f}, AIC={quadratic_aic:.4f}")

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

    if quad_score > linear_score:
        print(f"  *** Selected: QUADRATIC (score={quad_score:.4f}) ***")
        return quadratic_model, "quadratic", quadratic_r2, all_dprvic, all_delta, upper_dprvic, upper_delta
    else:
        print(f"  *** Selected: LINEAR (score={linear_score:.4f}) ***")
        return linear_model, "linear", linear_r2, all_dprvic, all_delta, upper_dprvic, upper_delta


# ============================================================================
# STEP 1: LOAD SAR DATA
# ============================================================================
print("=" * 60)
print("STEP 1: Loading SAR Data")
print("=" * 60)

all_tiffs = glob(os.path.join(SAR_DIR, "*.tif"))
field_tiffs = [f for f in all_tiffs if "_field" in f]
field_tiffs = remove_duplicate_dates(field_tiffs)
print(f"Found {len(field_tiffs)} field TIFF files (after dedup)")

data_dict = {}
for tiff_path in sorted(field_tiffs):
    dt = parse_date(tiff_path)
    if dt < GROWING_SEASON_START or dt > GROWING_SEASON_END:
        continue

    basename = os.path.basename(tiff_path)
    if ORBIT_FILTER and ORBIT_FILTER not in basename.lower():
        continue

    with rasterio.open(tiff_path) as src:
        vv_norm = src.read(1)
        vh_norm = src.read(2)
        mask_band = src.read(3)
        transform = src.transform
        crs = src.crs
        height, width = src.height, src.width

    valid_mask = mask_band == 1
    vv_linear = np.where(valid_mask, vv_norm, np.nan)
    vv_db = 10 * np.log10(np.where(vv_linear > 0, vv_linear, np.nan))

    q = np.where(valid_mask, vh_norm / np.where(vv_norm > 0, vv_norm, np.nan), np.nan)

    dprvic = np.where(valid_mask, q * (q + 3) / (q + 1)**2, np.nan)

    data_dict[dt] = {
        'vv': vv_db,
        'dprvic': dprvic,
        'transform': transform,
        'crs': crs,
        'height': height,
        'width': width,
    }
    print(f"  Loaded: {dt.strftime('%Y-%m-%d %H:%M')}")

dates = sorted(data_dict.keys())
print(f"\nTotal dates: {len(dates)}")

if len(dates) == 0:
    print("ERROR: No dates found. Check SAR data and date range.")
    exit()

sample_transform = data_dict[dates[0]]['transform']
sample_crs = data_dict[dates[0]]['crs']
height = data_dict[dates[0]]['height']
width = data_dict[dates[0]]['width']

# ============================================================================
# STEP 2: BUILD ZONE MASKS FROM SENSOR COORDINATES (70x70m boxes)
# ============================================================================
print("\n" + "=" * 60)
print("STEP 2: Building zone masks from sensor coordinates (70x70m boxes)")
print("=" * 60)

zone_masks = {}
for zone_name, (cx, cy) in ZONE_CENTERS_UTM.items():
    mask_arr = make_zone_mask(sample_transform, height, width, cx, cy, BOX_HALF)
    zone_masks[zone_name] = mask_arr
    print(f"  {zone_name}: center=({cx:.2f}, {cy:.2f}), {np.sum(mask_arr)} pixels")

# ============================================================================
# STEP 3: PROCESS EACH ZONE
# ============================================================================
all_results = {}

for zone_name, zone_mask in zone_masks.items():
    print("\n" + "=" * 60)
    print(f"STEP 3: Processing {zone_name.upper()}")
    print("=" * 60)

    vv_stack = np.stack([data_dict[dt]['vv'] for dt in dates], axis=0)
    dprvic_stack = np.stack([data_dict[dt]['dprvic'] for dt in dates], axis=0)

    combined_mask = np.stack([zone_mask] * len(dates), axis=0)
    vv_masked = np.where(combined_mask, vv_stack, np.nan)
    dprvic_masked = np.where(combined_mask, dprvic_stack, np.nan)

    print("Step 3a: Dry reference & delta backscatter...")
    sigma_dry = np.nanpercentile(vv_masked, 5, axis=0)
    delta_sigma = vv_masked - sigma_dry
    delta_sigma = np.maximum(delta_sigma, 0)
    print(f"  sigma_dry range: [{np.nanmin(sigma_dry):.2f}, {np.nanmax(sigma_dry):.2f}] dB")
    print(f"  delta_sigma range: [{np.nanmin(delta_sigma):.2f}, {np.nanmax(delta_sigma):.2f}] dB")

    print("\nStep 3b: Upper envelope regression...")
    model, model_type, model_r2, all_dprvic, all_delta, upper_dprvic, upper_delta = \
        fit_regression(delta_sigma, dprvic_masked, zone_name)

    fig, ax = plt.subplots(figsize=(6.2, 5.0))
    fig.patch.set_facecolor('white')
    ax.set_facecolor('white')
    ax.scatter(all_dprvic[::5], all_delta[::5],
               c='dodgerblue', s=22, alpha=0.55, label='All pixels', edgecolors='none')
    ax.scatter(upper_dprvic, upper_delta,
               c='red', s=55, alpha=0.9, label='Upper envelope (98th pct)', edgecolors='black', linewidths=0.6)
    dprvic_range = np.linspace(all_dprvic.min(), all_dprvic.max(), 100)
    if model_type == "quadratic":
        delta_pred = model[0] * dprvic_range**2 + model[1] * dprvic_range + model[2]
    else:
        delta_pred = model[0] * dprvic_range + model[1]
    ax.plot(dprvic_range, delta_pred, 'darkred', linewidth=3.0,
            label=f'Best fit ({model_type}): R2={model_r2:.3f}')
    ax.set_xlabel('DpRVIc', fontsize=21)
    ax.set_ylabel('Delta Backscatter (dB)', fontsize=21)
    ax.set_title("Management Zone 1" if "MZ1" in zone_name else "Management Zone 2", fontsize=21)
    ax.tick_params(labelsize=16)
    _y0, _y1 = ax.get_ylim()
    ax.set_ylim(_y0, _y1 + 0.30 * (_y1 - _y0))
    ax.legend(loc='upper right', facecolor='white', edgecolor='black', fontsize=13, framealpha=0.9)
    ax.grid(True, alpha=0.3, color='gray')
    plt.tight_layout()

    chart_path = os.path.join(OUTPUT_DIR, f"regression_dprvic_{zone_name}.png")
    plt.savefig(chart_path, dpi=150, facecolor='white')
    plt.savefig(chart_path.replace('.png', '.pdf'))
    plt.close()
    print(f"  Saved: {chart_path}")

    print("\nStep 3c: Computing Theta...")
    if model_type == "quadratic":
        delta_sigma_max = model[0] * dprvic_masked**2 + model[2]
    else:
        delta_sigma_max = model[0] * dprvic_masked + model[1]

    Theta = delta_sigma / np.maximum(delta_sigma_max, 0.01)
    Theta = np.clip(Theta, 0, 1)
    print(f"  Theta range: [{np.nanmin(Theta):.4f}, {np.nanmax(Theta):.4f}]")

    results = []
    for i, dt in enumerate(dates):
        results.append({
            'date': dt,
            'area': zone_name,
            'Theta': np.nanmean(Theta[i][zone_mask]),
            'delta_sigma': np.nanmean(delta_sigma[i][zone_mask]),
            'DpRVIc': np.nanmean(dprvic_masked[i]),
        })

    all_results[zone_name] = {
        'results': results,
        'Theta': Theta,
        'delta_sigma': delta_sigma,
        'dprvic': dprvic_masked,
        'model': model,
        'model_type': model_type,
    }

    print(f"\n  {zone_name.upper()} zone-averaged Theta:")
    for r in results:
        print(f"    {r['date'].strftime('%Y-%m-%d')}: Theta={r['Theta']:.4f}")

# ============================================================================
# STEP 4: SAVE OUTPUTS
# ============================================================================
print("\n" + "=" * 60)
print("STEP 4: Saving Outputs")
print("=" * 60)

for zone_name, zone_data in all_results.items():
    Theta = zone_data['Theta']
    delta_sigma = zone_data['delta_sigma']
    dprvic = zone_data['dprvic']

    delta_folder = os.path.join(OUTPUT_DIR, f"delta_backscatter_{zone_name}")
    dprvic_folder = os.path.join(OUTPUT_DIR, f"dprvic_{zone_name}")
    theta_folder = os.path.join(OUTPUT_DIR, f"theta_{zone_name}")
    os.makedirs(delta_folder, exist_ok=True)
    os.makedirs(dprvic_folder, exist_ok=True)
    os.makedirs(theta_folder, exist_ok=True)

    for i, dt in enumerate(dates):
        date_str = dt.strftime("%Y%m%d")

        with rasterio.open(os.path.join(delta_folder, f"field_{date_str}.tif"), 'w',
                           driver='GTiff', height=height, width=width,
                           count=1, dtype=delta_sigma.dtype, crs=sample_crs,
                           transform=sample_transform) as dst:
            dst.write(delta_sigma[i], 1)

        with rasterio.open(os.path.join(dprvic_folder, f"field_{date_str}.tif"), 'w',
                           driver='GTiff', height=height, width=width,
                           count=1, dtype=dprvic.dtype, crs=sample_crs,
                           transform=sample_transform) as dst:
            dst.write(dprvic[i], 1)

        with rasterio.open(os.path.join(theta_folder, f"field_{date_str}.tif"), 'w',
                           driver='GTiff', height=height, width=width,
                           count=1, dtype=Theta.dtype, crs=sample_crs,
                           transform=sample_transform) as dst:
            dst.write(Theta[i], 1)

    theta_mean = np.nanmean(Theta, axis=0)
    out_mean = os.path.join(OUTPUT_DIR, f"mean_theta_{zone_name}_desc_jun15.tif")
    with rasterio.open(out_mean, 'w',
                       driver='GTiff', height=height, width=width,
                       count=1, dtype=theta_mean.dtype, crs=sample_crs,
                       transform=sample_transform) as dst:
        dst.write(theta_mean, 1)

    print(f"Saved: {zone_name} outputs ({len(dates)} dates)")

all_rows = []
for zone_name, zone_data in all_results.items():
    df = pd.DataFrame(zone_data['results'])
    df['date_str'] = df['date'].apply(lambda x: x.strftime('%Y-%m-%d'))
    all_rows.append(df[['date_str', 'area', 'Theta', 'delta_sigma', 'DpRVIc']])

combined_df = pd.concat(all_rows, ignore_index=True)
csv_path = os.path.join(OUTPUT_DIR, "theta_timeseries.csv")
try:
    combined_df.to_csv(csv_path, index=False)
    print(f"Saved: {csv_path}")
except PermissionError:
    csv_path = os.path.join(OUTPUT_DIR, "theta_timeseries_new.csv")
    combined_df.to_csv(csv_path, index=False)
    print(f"Saved (original was locked): {csv_path}")

# ============================================================================
# STEP 5: PER-PIXEL EXPONENTIAL FILTER (SWI)
# ============================================================================
# Applies Wagner et al. (1999) / Albergel et al. (2008) exponential filter
# pixel-by-pixel to convert surface Theta -> profile SWI at depth.
# T parameter: loam soil (Albergel 2008 Table 2)
#   5 cm:  T = 5 days
#   10 cm: T = 8 days
#   15 cm: T = 12 days
# Best validated result: SWI(DpRVIc) T=12d, R²=0.80 (MZ2), 0.35 (MZ1)

VWC_CAL_A = 0.236
VWC_CAL_B = 0.139
SWI_T_VALUES = {'15cm': 12.0}

# Per-zone calibration: SWI → VWC at 15cm depth (from Deming regression)
# These are computed by exp_filter_retrieval.py and stored in exp_filter_params.json
SWI_CALIB_FILE = os.path.join(OUTPUT_DIR, "exp_filter_params.json")
SWI_CALIB = {}
if os.path.exists(SWI_CALIB_FILE):
    with open(SWI_CALIB_FILE) as f:
        _params = json.load(f)
    for zone_name in ZONE_CENTERS_UTM:
        _zone_cal = _params.get('validation', {}).get(zone_name, {})
        for T_label in SWI_T_VALUES:
            slope_key = f'swi_dp_{T_label}_slope'
            int_key = f'swi_dp_{T_label}_intercept'
            if slope_key in _zone_cal and int_key in _zone_cal:
                SWI_CALIB[(zone_name, T_label)] = {
                    'slope': _zone_cal[slope_key],
                    'intercept': _zone_cal[int_key],
                }
    if SWI_CALIB:
        print(f"  Loaded SWI calibration for zones: {list(set(k[0] for k in SWI_CALIB.keys()))}")
    else:
        print("  WARNING: No SWI calibration found in params, using surface VWC coeffs")
else:
    print("  WARNING: exp_filter_params.json not found, using surface VWC coeffs")

print("\n" + "=" * 60)
print("STEP 5: Per-pixel exponential filter (SWI)")
print("=" * 60)

for zone_name, zone_data in all_results.items():
    Theta_stack = zone_data['Theta']  # shape: (n_dates, height, width)
    h, w = Theta_stack.shape[1], Theta_stack.shape[2]
    valid_pixel = ~np.all(np.isnan(Theta_stack), axis=0)  # pixels with any valid data
    n_valid = np.sum(valid_pixel)
    print(f"\n  {zone_name}: applying exp filter to {n_valid} valid pixels "
          f"({Theta_stack.shape[0]} dates)")

    # Convert dates to day offsets for gap calculation
    date_offsets = np.array([(dt - dates[0]).days for dt in dates])

    for T_label, T_val in SWI_T_VALUES.items():
        SWI_stack = np.full_like(Theta_stack, np.nan)

        for row in range(h):
            for col in range(w):
                if not valid_pixel[row, col]:
                    continue
                pixel_theta = Theta_stack[:, row, col]
                valid_idx = np.where(~np.isnan(pixel_theta))[0]
                if len(valid_idx) == 0:
                    continue
                # Apply exponential filter on valid observations only
                swi = np.full(len(valid_idx), np.nan)
                swi[0] = pixel_theta[valid_idx[0]]
                for k in range(1, len(valid_idx)):
                    dt = date_offsets[valid_idx[k]] - date_offsets[valid_idx[k - 1]]
                    K = 1.0 - np.exp(-dt / T_val)
                    swi[k] = swi[k - 1] + K * (pixel_theta[valid_idx[k]] - swi[k - 1])
                # Write back (no rescaling — SWI retains dampened physics)
                for k, idx in enumerate(valid_idx):
                    SWI_stack[idx, row, col] = swi[k]

        # Save SWI maps
        swi_folder = os.path.join(OUTPUT_DIR, f"swi_{zone_name}_T{T_val:.0f}d")
        os.makedirs(swi_folder, exist_ok=True)
        for i, dt in enumerate(dates):
            date_str = dt.strftime("%Y%m%d")
            with rasterio.open(os.path.join(swi_folder, f"field_{date_str}.tif"), 'w',
                               driver='GTiff', height=h, width=w,
                               count=1, dtype=SWI_stack.dtype, crs=sample_crs,
                               transform=sample_transform) as dst:
                dst.write(SWI_stack[i], 1)

        # Save mean SWI and VWC maps
        swi_mean = np.nanmean(SWI_stack, axis=0)
        cal = SWI_CALIB.get((zone_name, T_label), {'slope': VWC_CAL_B, 'intercept': VWC_CAL_A})
        vwc_mean = cal['intercept'] + cal['slope'] * swi_mean

        with rasterio.open(os.path.join(OUTPUT_DIR, f"mean_swi_{zone_name}_T{T_val:.0f}d.tif"), 'w',
                           driver='GTiff', height=h, width=w,
                           count=1, dtype=swi_mean.dtype, crs=sample_crs,
                           transform=sample_transform) as dst:
            dst.write(swi_mean, 1)

        with rasterio.open(os.path.join(OUTPUT_DIR, f"mean_vwc_{zone_name}_T{T_val:.0f}d.tif"), 'w',
                           driver='GTiff', height=h, width=w,
                           count=1, dtype=vwc_mean.dtype, crs=sample_crs,
                           transform=sample_transform) as dst:
            dst.write(vwc_mean, 1)

        print(f"    T={T_val:.0f}d ({T_label}): saved {len(dates)} SWI maps + mean SWI/VWC")

        zone_data[f'SWI_{T_label}'] = SWI_stack
        zone_data[f'swi_mean_{T_label}'] = swi_mean
        zone_data[f'vwc_mean_{T_label}'] = vwc_mean

# ============================================================================
# STEP 6: SAVE SUMMARY CSVs
# ============================================================================
print("\n" + "=" * 60)
print("STEP 6: Saving SWI/VWC summary and zone-averaged timeseries")
print("=" * 60)

for zone_name, zone_data in all_results.items():
    zone_mask = zone_masks[zone_name]
    sw_rows = []
    for i, dt in enumerate(dates):
        row = {
            'date': dt.strftime('%Y-%m-%d'),
            'area': zone_name,
            'Theta_mean': np.nanmean(zone_data['Theta'][i][zone_mask]),
            'DpRVIc_mean': np.nanmean(zone_data['dprvic'][i][zone_mask]),
        }
        for T_label, T_val in SWI_T_VALUES.items():
            swi_key = f'SWI_{T_label}'
            if swi_key in zone_data:
                row[f'SWI_{T_label}_mean'] = np.nanmean(zone_data[swi_key][i][zone_mask])
                cal = SWI_CALIB.get((zone_name, T_label), {'slope': VWC_CAL_B, 'intercept': VWC_CAL_A})
                row[f'VWC_{T_label}_mean'] = cal['intercept'] + cal['slope'] * row[f'SWI_{T_label}_mean']
        sw_rows.append(row)

    sw_df = pd.DataFrame(sw_rows)
    sw_csv = os.path.join(OUTPUT_DIR, f"swi_timeseries_{zone_name}.csv")
    sw_df.to_csv(sw_csv, index=False)
    print(f"  Saved: {sw_csv}")

print("\n" + "=" * 60)
print("PROCESSING COMPLETE")
print("=" * 60)
