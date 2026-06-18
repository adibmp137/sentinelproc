import os
import re
import numpy as np
from glob import glob
from datetime import datetime
import rasterio

INPUT_DIR = os.path.join(os.path.dirname(__file__), "SAR_timeseries_masked")
OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "SAR_timeseries_normalized")

GROWING_SEASON_START = datetime(2025, 4, 24)
GROWING_SEASON_END = datetime(2025, 7, 15)

N_VV = None
N_VH = None

DYNAMIC_N_VV_SLOPE = 0.40
DYNAMIC_N_VV_INTERCEPT = -0.38
DYNAMIC_N_VH_SLOPE = 0.26
DYNAMIC_N_VH_INTERCEPT = -0.11


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


def process_field():
    all_tiffs = glob(os.path.join(INPUT_DIR, "*.tif"))
    field_tiffs = [f for f in all_tiffs if "_field" in f]
    field_tiffs = remove_duplicate_dates(field_tiffs)
    field_tiffs = [f for f in field_tiffs
                   if GROWING_SEASON_START <= parse_date(f) <= GROWING_SEASON_END]

    print(f"Found {len(field_tiffs)} field TIFF files in growing season")

    print("=" * 60)
    print("STEP 1: Collecting scene statistics")
    print("=" * 60)

    scene_stats = []
    for tiff_path in sorted(field_tiffs):
        with rasterio.open(tiff_path) as src:
            num_bands = src.count
            vv = src.read(1)
            vh = src.read(2)
            mask_band = src.read(3)
            lia = src.read(4) if num_bands >= 4 else None

        valid = mask_band == 1

        if lia is None:
            print(f"  {os.path.basename(tiff_path)}: no LIA band, skipping")
            continue

        vv_valid = vv[valid]
        vh_valid = vh[valid]
        lia_valid = lia[valid]

        vv_pos = vv_valid[vv_valid > 0]
        vh_pos = vh_valid[vh_valid > 0]
        vv_db_mean = np.nanmean(10 * np.log10(vv_pos))
        vh_db_mean = np.nanmean(10 * np.log10(vh_pos))
        sar_ratio_db = vv_db_mean - vh_db_mean

        scene_stats.append({
            'file': tiff_path,
            'mean_vv_db': vv_db_mean,
            'mean_vh_db': vh_db_mean,
            'mean_lia': np.nanmean(lia_valid),
            'sar_ratio_db': sar_ratio_db,
        })

    print(f"Scenes with LIA: {len(scene_stats)}")

    if len(scene_stats) == 0:
        print("ERROR: No valid scenes found. Exiting.")
        return

    theta_ref = np.mean([s['mean_lia'] for s in scene_stats])
    print(f"  Reference angle (theta_ref): {theta_ref:.2f} deg")

    print("\n" + "=" * 60)
    if N_VV is not None and N_VH is not None:
        print("STEP 2: Using fixed N values")
        print("=" * 60)
        print(f"  N_VV = {N_VV:.2f} (fixed)")
        print(f"  N_VH = {N_VH:.2f} (fixed)")
        for s in scene_stats:
            s['n_vv'] = N_VV
            s['n_vh'] = N_VH
    else:
        print("STEP 2: Computing dynamic N from SAR Ratio (Najem et al. 2024, Table 2)")
        print("=" * 60)
        print(f"  N_VV = {DYNAMIC_N_VV_SLOPE} * SAR_Ratio_dB + ({DYNAMIC_N_VV_INTERCEPT})")
        print(f"  N_VH = {DYNAMIC_N_VH_SLOPE} * SAR_Ratio_dB + ({DYNAMIC_N_VH_INTERCEPT})")
        for s in scene_stats:
            s['n_vv'] = DYNAMIC_N_VV_SLOPE * s['sar_ratio_db'] + DYNAMIC_N_VV_INTERCEPT
            s['n_vh'] = DYNAMIC_N_VH_SLOPE * s['sar_ratio_db'] + DYNAMIC_N_VH_INTERCEPT

    print(f"\n  {'Date':<12} {'Mean LIA':>10} {'SAR Ratio':>10} {'N_VV':>8} {'N_VH':>8}")
    for s in scene_stats:
        dt = parse_date(s['file'])
        print(f"  {dt.strftime('%Y-%m-%d'):<12} {s['mean_lia']:>10.2f} "
              f"{s['sar_ratio_db']:>10.2f} {s['n_vv']:>8.2f} {s['n_vh']:>8.2f}")

    print("\n" + "=" * 60)
    print("STEP 3: Applying incidence angle normalization")
    print("=" * 60)
    print(f"  Formula: sigma_norm = sigma * (cos({theta_ref:.2f}) / cos(LIA))^N")

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    stats_by_file = {s['file']: s for s in scene_stats}

    for tiff_path in sorted(field_tiffs):
        basename = os.path.basename(tiff_path)

        with rasterio.open(tiff_path) as src:
            vv = src.read(1)
            vh = src.read(2)
            mask_band = src.read(3)
            has_lia = src.count >= 4
            lia = src.read(4) if has_lia else np.full_like(vv, theta_ref)
            meta = src.meta.copy()

        valid = mask_band == 1

        s = stats_by_file.get(tiff_path)
        n_vv = s['n_vv'] if s else (N_VV or 2.5)
        n_vh = s['n_vh'] if s else (N_VH or 2.0)

        cos_ratio = np.where(valid,
                             np.cos(np.radians(theta_ref)) / np.cos(np.radians(lia)),
                             1.0)
        cos_ratio = np.clip(cos_ratio, 0.1, 10.0)

        vv_norm = vv * np.power(cos_ratio, n_vv)
        vh_norm = vh * np.power(cos_ratio, n_vh)

        vv_norm = np.where(valid, vv_norm, vv)
        vh_norm = np.where(valid, vh_norm, vh)

        dt = parse_date(tiff_path)
        mean_lia = np.nanmean(lia[valid]) if valid.any() else theta_ref
        print(f"  {dt.strftime('%Y-%m-%d')}: LIA={mean_lia:.1f}, "
              f"N_VV={n_vv:.2f}, N_VH={n_vh:.2f}")

        out_bands = np.stack([vv_norm, vh_norm, mask_band, lia], axis=0)

        meta.update({
            "count": 4,
            "height": out_bands.shape[1],
            "width": out_bands.shape[2],
        })

        out_path = os.path.join(OUTPUT_DIR, basename)
        with rasterio.open(out_path, "w", **meta) as dst:
            dst.write(out_bands)

    print(f"\nSaved {len(field_tiffs)} normalized files to {OUTPUT_DIR}")


process_field()

print("\nDone!")
