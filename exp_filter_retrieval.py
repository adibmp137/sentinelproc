"""
Exponential filter retrieval: DpRVIc Theta -> SWI (Soil Water Index).

Applies the exponential filter of Wagner et al. (1999) and Albergel et al. (2008)
to DpRVIc-derived surface soil moisture (Theta, 0-1) to produce profile soil moisture
at sensor depth (~5-15 cm). Uses all orbits (~32 obs) for better temporal sampling.

Key result: SWI(DpRVIc) T=12d achieves R²=0.80 (MZ2) and R²=0.35 (MZ1),
compared to raw DpRVIc R²=-0.02 or raw VV-descending R²=0.36.

References:
  [1] Wagner et al. (1999) - Remote Sensing of Environment, 70(2), 191-207
  [2] Albergel et al. (2008) - Hydrol. Earth Syst. Sci., 12(6), 1323-1337
  [3] Albergel et al. (2011) - IEEE TGRS, 49(9), 3358-3367

T parameters (loam, Albergel 2008 Table 2):
  5 cm:  T = 5 days
  10 cm: T = 8 days
  15 cm: T = 12 days
"""

import os
import re
import json
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from glob import glob
from scipy import stats
from scipy.odr import ODR, Model, RealData
import rasterio
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.offsetbox import TextArea, VPacker, AnchoredOffsetbox
matplotlib.rcParams.update({
    'font.family': 'serif',
    'mathtext.fontset': 'dejavuserif',
    'font.size': 16,
    'axes.titlesize': 17,
    'axes.labelsize': 16,
    'xtick.labelsize': 13,
    'ytick.labelsize': 13,
    'legend.fontsize': 12,
    'lines.linewidth': 2.0,
})
import warnings
warnings.filterwarnings('ignore')

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
INPUT_DIR = os.path.join(BASE_DIR, "input")
OUTPUT_DIR = os.path.join(BASE_DIR, "output")
os.makedirs(OUTPUT_DIR, exist_ok=True)

GROWING_START = datetime(2025, 4, 24)
GROWING_END = datetime(2025, 6, 5)

SENSING_DEPTH_CM = 5

SENSOR_CAL_A = -0.006
SENSOR_CAL_B = 1.26

T_VALUES = {'15cm': 12.0}

VWC_CAL_A = 0.236
VWC_CAL_B = 0.139

# Configure with actual sensor UTM coordinates for your study area
ZONE_CENTERS_UTM = {
    'MZ1': (0.0, 0.0),
    'MZ2': (0.0, 0.0),
}


# =============================================================================
# Functions
# =============================================================================

def load_sensor_daily(filename):
    path = os.path.join(INPUT_DIR, filename)
    df = pd.read_csv(path)
    df['date'] = pd.to_datetime(df['time_parsed']).dt.date
    gs = GROWING_START.date()
    ge = GROWING_END.date()
    df = df[(df['date'] >= gs) & (df['date'] <= ge)]
    vmc_cols = [c for c in df.columns if c.startswith('vmc')]
    df['vmc_mean'] = df[vmc_cols].mean(axis=1)
    df = df[df['vmc_mean'] > 0]
    daily = df.groupby('date')['vmc_mean'].mean().reset_index()
    daily.columns = ['date', 'vmc_raw']
    daily['sensor_cal'] = SENSOR_CAL_A + SENSOR_CAL_B * daily['vmc_raw']
    return daily


def load_rain_daily(filename):
    path = os.path.join(INPUT_DIR, filename)
    df = pd.read_csv(path)
    df['date'] = pd.to_datetime(df['time_parsed']).dt.date
    gs = GROWING_START.date()
    ge = GROWING_END.date()
    df = df[(df['date'] >= gs) & (df['date'] <= ge)]
    df['relPrecipitation'] = pd.to_numeric(df['relPrecipitation'], errors='coerce').fillna(0)
    daily = df.groupby('date')['relPrecipitation'].sum().reset_index()
    daily.columns = ['date', 'rain_mm']
    return daily


def deming_regression(x, y, lambda_ratio=1.0):
    def f(B, x):
        return B[0] * x + B[1]
    linear = Model(f)
    data = RealData(x, y, sx=np.std(x) / np.sqrt(lambda_ratio), sy=np.std(y))
    odr = ODR(data, linear, beta0=[1.0, 0.0])
    output = odr.run()
    return output.beta[0], output.beta[1], output.sd_beta[0], output.sd_beta[1]


def exponential_filter(dates_obs, surface_sm, T):
    """SWI(t_n) = SWI(t_{n-1}) + K(t_n) * [mvs(t_n) - SWI(t_{n-1})]
    K(t_n) = 1 - exp(-(t_n - t_{n-1}) / T)"""
    n = len(dates_obs)
    swi = np.zeros(n)
    swi[0] = surface_sm[0]
    for i in range(1, n):
        dt = (dates_obs[i] - dates_obs[i - 1]).days
        K = 1.0 - np.exp(-dt / T)
        swi[i] = swi[i - 1] + K * (surface_sm[i] - swi[i - 1])
    return swi


def exponential_filter_continuous(date_range, obs_dates, obs_values, T):
    """Daily continuous SWI from sparse observations. SWI only updates on obs dates.
    Before first observation, backward-fill with the first observed value."""
    swi_daily = np.full(len(date_range), np.nan)
    obs_dict = {d: v for d, v in zip(obs_dates, obs_values)}

    first_obs_idx = None
    for i, d in enumerate(date_range):
        if d in obs_dict:
            if first_obs_idx is None:
                first_obs_idx = i
            swi_daily[i] = obs_dict[d]

    if first_obs_idx is None:
        return pd.Series(swi_daily, index=date_range, name='SWI')

    # Backward-fill: set all days before first obs to first observed value
    for j in range(first_obs_idx):
        swi_daily[j] = swi_daily[first_obs_idx]

    obs_indices = [i for i, d in enumerate(date_range) if d in obs_dict]
    for idx in range(1, len(obs_indices)):
        i_curr = obs_indices[idx]
        i_prev = obs_indices[idx - 1]
        dt = (date_range[i_curr] - date_range[i_prev]).days
        K = 1.0 - np.exp(-dt / T)
        swi_daily[i_curr] = swi_daily[i_prev] + K * (obs_dict[date_range[i_curr]] - swi_daily[i_prev])
        for j in range(i_prev + 1, i_curr):
            swi_daily[j] = swi_daily[j - 1]

    # Fill any remaining gaps forward
    for i in range(1, len(date_range)):
        if np.isnan(swi_daily[i]) and not np.isnan(swi_daily[i - 1]):
            swi_daily[i] = swi_daily[i - 1]

    return pd.Series(swi_daily, index=date_range, name='SWI')


def date_to_dt(d):
    if isinstance(d, pd.Timestamp):
        return d.to_pydatetime()
    return datetime.combine(d, datetime.min.time()) if not isinstance(d, datetime) else d


# =============================================================================
# Main pipeline
# =============================================================================

print("=" * 60)
print("DpRVIc + Exponential Filter Retrieval (SWI)")
print("References: Wagner et al. (1999), Albergel et al. (2008)")
print("=" * 60)

# --- 1. Load DpRVIc Theta from soil_moisture_field.py output ---
print("\nStep 1: Loading DpRVIc Theta timeseries...")
theta_csv = os.path.join(OUTPUT_DIR, "theta_timeseries.csv")
if not os.path.exists(theta_csv):
    print("ERROR: theta_timeseries.csv not found. Run soil_moisture_field.py first.")
    exit()

theta_df = pd.read_csv(theta_csv)
theta_df['date'] = pd.to_datetime(theta_df['date_str']).dt.date

# --- 2. Load sensor and rain data ---
print("Step 2: Loading sensor and rain data...")
sensor_mz1 = load_sensor_daily("MZ1.csv")
sensor_mz2 = load_sensor_daily("MZ2.csv")
rain_df = load_rain_daily("MZ1.csv")

gs = GROWING_START.date()
ge = GROWING_END.date()
date_range = pd.date_range(gs, ge, freq='D').date

# --- 3. Apply exponential filter to DpRVIc Theta ---
print("\nStep 3: Applying exponential filter to DpRVIc Theta (all orbits)...")

results = {}

for zone_name in ZONE_CENTERS_UTM:
    sensor_daily = sensor_mz1 if 'MZ1' in zone_name else sensor_mz2
    theta_zone = theta_df[theta_df['area'] == zone_name].copy()
    theta_zone = theta_zone.sort_values('date').reset_index(drop=True)

    dates_dp = theta_zone['date'].values
    theta_dp = theta_zone['Theta'].values

    obs_dt = np.array([date_to_dt(d) for d in dates_dp])

    obs_dict_dp = {}
    for d, th in zip(dates_dp, theta_dp):
        dt_key = d.date() if hasattr(d, 'date') else d
        obs_dict_dp[dt_key] = th
    sorted_keys = sorted(obs_dict_dp.keys())
    sorted_vals = [obs_dict_dp[k] for k in sorted_keys]

    swi_obs = {}
    swi_daily_dict = {}
    for T_label, T_val in T_VALUES.items():
        swi_obs[T_label] = exponential_filter(obs_dt, theta_dp, T_val)
        swi_daily_dict[T_label] = exponential_filter_continuous(date_range, sorted_keys, sorted_vals, T_val)

    # Validation: DpRVIc direct (no filter)
    matched_dp = pd.DataFrame({'date': dates_dp, 'theta_dp': theta_dp})
    matched_dp['date'] = pd.to_datetime(matched_dp['date'])
    sensor_copy = sensor_daily.copy()
    sensor_copy['date'] = pd.to_datetime(sensor_copy['date'])
    matched_dp = pd.merge(matched_dp, sensor_copy, on='date', how='inner')

    vwc_dp = VWC_CAL_A + VWC_CAL_B * matched_dp['theta_dp'].values
    r_dp, _ = stats.pearsonr(vwc_dp, matched_dp['sensor_cal'].values)
    coeffs_dp = np.polyfit(vwc_dp, matched_dp['sensor_cal'].values, 1)
    pred_dp = np.polyval(coeffs_dp, vwc_dp)
    ss_res_dp = np.sum((matched_dp['sensor_cal'].values - pred_dp) ** 2)
    ss_tot_dp = np.sum((matched_dp['sensor_cal'].values - matched_dp['sensor_cal'].values.mean()) ** 2)
    r2_dp = 1 - ss_res_dp / ss_tot_dp
    rmse_dp = np.sqrt(np.mean((matched_dp['sensor_cal'].values - vwc_dp) ** 2))

    print(f"\n  {zone_name} (DpRVIc, all orbits, n={len(matched_dp)}):")
    print(f"    DpRVIc (no filter):      r={r_dp:.4f}, R²={r2_dp:.4f}, RMSE={rmse_dp:.4f}")

    zone_results = {
        'n': int(len(matched_dp)),
        'dp_r': float(r_dp), 'dp_r2': float(r2_dp), 'dp_rmse': float(rmse_dp),
    }

    # Validation: SWI at each T — calibrate directly to 15cm sensor via Deming regression
    for T_label, T_val in T_VALUES.items():
        swi_dp = swi_obs[T_label]
        if len(swi_dp) != len(matched_dp):
            print(f"    SWI(DpRVIc) T={T_val:.0f}d: length mismatch, skipping")
            continue

        # Deming regression: sensor_cal = slope * SWI + intercept
        slope_swi, int_swi, slope_se, int_se = deming_regression(swi_dp, matched_dp['sensor_cal'].values)
        vwc_swi_dp = int_swi + slope_swi * swi_dp

        r_swi, _ = stats.pearsonr(vwc_swi_dp, matched_dp['sensor_cal'].values)
        coeffs_swi = np.polyfit(vwc_swi_dp, matched_dp['sensor_cal'].values, 1)
        pred_swi = np.polyval(coeffs_swi, vwc_swi_dp)
        ss_res_swi = np.sum((matched_dp['sensor_cal'].values - pred_swi) ** 2)
        ss_tot_swi = np.sum((matched_dp['sensor_cal'].values - matched_dp['sensor_cal'].values.mean()) ** 2)
        r2_swi = 1 - ss_res_swi / ss_tot_swi
        rmse_swi = np.sqrt(np.mean((matched_dp['sensor_cal'].values - vwc_swi_dp) ** 2))
        bias_swi = np.mean(vwc_swi_dp - matched_dp['sensor_cal'].values)

        print(f"    SWI(DpRVIc) T={T_val:.0f}d ({T_label}): r={r_swi:.4f}, R²={r2_swi:.4f}, "
              f"RMSE={rmse_swi:.4f}, bias={bias_swi:.4f}")
        print(f"      Calibration: VWC = {int_swi:.4f} + {slope_swi:.4f} * SWI")

        zone_results[f'swi_dp_{T_label}_r'] = float(r_swi)
        zone_results[f'swi_dp_{T_label}_r2'] = float(r2_swi)
        zone_results[f'swi_dp_{T_label}_rmse'] = float(rmse_swi)
        zone_results[f'swi_dp_{T_label}_bias'] = float(bias_swi)
        zone_results[f'swi_dp_{T_label}_slope'] = float(slope_swi)
        zone_results[f'swi_dp_{T_label}_intercept'] = float(int_swi)

    results[zone_name] = {
        'dates': dates_dp,
        'theta': theta_dp,
        'swi': swi_obs,
        'swi_daily': swi_daily_dict,
        'matched': matched_dp,
        'validation': zone_results,
    }

# --- 4. Summary ---
print("\n" + "=" * 60)
print("Step 4: Validation Summary — DpRVIc + Exponential Filter")
print("=" * 60)

for zone_name, res in results.items():
    vr = res['validation']
    print(f"\n  {zone_name} (n={vr['n']}):")
    print(f"    DpRVIc (no filter):      r={vr['dp_r']:.4f}, R²={vr['dp_r2']:.4f}")
    for T_label, T_val in T_VALUES.items():
        key_r = f'swi_dp_{T_label}_r'
        key_r2 = f'swi_dp_{T_label}_r2'
        if key_r2 in vr:
            print(f"    SWI(DpRVIc) T={T_val:.0f}d ({T_label}): r={vr[key_r]:.4f}, R²={vr[key_r2]:.4f}")

# --- 5. Time series figure ---
print("\n" + "=" * 60)
print("Step 5: Generating time series figure")
print("=" * 60)

rain_dict = dict(zip(rain_df['date'], rain_df['rain_mm']))
rain_vals = [rain_dict.get(d, 0) for d in date_range]

# irrigation forcing (kept separate from precipitation, matching the SWB figures)
irrig_df = pd.read_csv(os.path.join(INPUT_DIR, "irrigation.csv"), sep=';')
irrig_df['date'] = pd.to_datetime(irrig_df['date'], format='%d-%b-%y').dt.date
irrig_df['irrig_mm'] = pd.to_numeric(irrig_df['irrig_mm'], errors='coerce').fillna(0)
irrig_dict = dict(zip(irrig_df['date'], irrig_df['irrig_mm']))
irrig_vals = [irrig_dict.get(d, 0) for d in date_range]

fig, axes = plt.subplots(len(ZONE_CENTERS_UTM), 1, figsize=(10, 3.8 * len(ZONE_CENTERS_UTM)),
                           sharex=True)
if len(ZONE_CENTERS_UTM) == 1:
    axes = [axes]
fig.patch.set_facecolor('white')

for ax_idx, (zone_name, res) in enumerate(results.items()):
    ax = axes[ax_idx]
    ax2 = ax.twinx()
    sensor_daily = sensor_mz1 if 'MZ1' in zone_name else sensor_mz2

    ax2.bar(date_range, rain_vals, width=0.8, color='#4292c6', alpha=0.55, label='Precipitation')
    ax2.bar(date_range, irrig_vals, width=0.8, color='#e6550d', alpha=0.70, label='Irrigation')
    _fmx = max(max(rain_vals) if rain_vals else 1, max(irrig_vals) if irrig_vals else 1, 1)
    ax2.set_ylim(0, _fmx * 4)
    _frags = [('Precip.', '#2171b5'), (' / ', '#444444'), ('irrig.', '#d94801'), (' (mm)', '#444444')]
    _fboxes = [TextArea(t, textprops=dict(color=c, rotation=90, ha='left', va='bottom', fontsize=12))
               for t, c in _frags[::-1]]
    _fybox = VPacker(children=_fboxes, align='center', pad=0, sep=2)
    ax2.add_artist(AnchoredOffsetbox(loc='center left', child=_fybox, pad=0, frameon=False,
                                     bbox_to_anchor=(1.06, 0.5), bbox_transform=ax2.transAxes, borderpad=0))
    ax2.tick_params(axis='y', labelcolor='#444444')

    ax.plot(sensor_daily['date'], sensor_daily['sensor_cal'], '-',
            color='steelblue', alpha=0.95, linewidth=2.2, label='Sensor (calibrated)')

    theta_dp = res['theta']
    vwc_dp = VWC_CAL_A + VWC_CAL_B * theta_dp
    ax.scatter(list(res['dates']), vwc_dp, c='darkorange', s=48, marker='s',
               alpha=0.95, edgecolors='black', linewidths=0.4, label=f'Before exp. filter', zorder=4)

    for T_label, T_val in T_VALUES.items():
        swi_daily = res['swi_daily'].get(T_label)
        if swi_daily is not None:
            slope_t = res['validation'].get(f'swi_dp_{T_label}_slope', VWC_CAL_B)
            intercept_t = res['validation'].get(f'swi_dp_{T_label}_intercept', VWC_CAL_A)
            vwc_swi_dp = intercept_t + slope_t * swi_daily.values
            style = '-' if T_label == '5cm' else ('--' if T_label == '10cm' else ':')
            ax.plot(date_range, vwc_swi_dp, style, color='purple', alpha=0.95,
                    linewidth=2.4, label=f'After exp. filter')

    ax.set_ylabel('VWC (m³/m³)', fontsize=15)
    zone_label = 'Management Zone 1' if 'MZ1' in zone_name else 'Management Zone 2'
    ax.set_title(f'{zone_label}', fontsize=15)
    _y0, _y1 = ax.get_ylim()
    ax.set_ylim(_y0, _y1 + 0.45 * (_y1 - _y0))
    _h1, _l1 = ax.get_legend_handles_labels()
    _h2, _l2 = ax2.get_legend_handles_labels()
    ax.legend(_h1 + _h2, _l1 + _l2, loc='upper left', fontsize=11, framealpha=0.9, ncol=2)
    ax.grid(True, alpha=0.3)
    ax.set_zorder(ax2.get_zorder() + 1)
    ax.patch.set_visible(False)

axes[-1].set_xlabel('Date')
fig.autofmt_xdate()
plt.tight_layout()
fig_path = os.path.join(OUTPUT_DIR, "exp_filter_timeseries.png")
fig.savefig(fig_path, dpi=150, facecolor='white', bbox_inches='tight')
fig.savefig(fig_path.replace('.png', '.pdf'))
print(f"Saved: {fig_path}")
plt.close(fig)

# --- 6. Scatter plots ---
print("\nStep 6: Generating scatter plots...")

zone_names = list(results.keys())
zone_labels = ['Management Zone 1' if 'MZ1' in z else 'Management Zone 2' for z in zone_names]

fig, axes = plt.subplots(1, 2, figsize=(8.5, 4.2))
fig.patch.set_facecolor('white')

for ax_idx, (zone_name, res) in enumerate(results.items()):
    ax = axes[ax_idx]
    sensor_daily = sensor_mz1 if 'MZ1' in zone_name else sensor_mz2
    matched_dp = res['matched']
    vr = res['validation']

    x_sensor = matched_dp['sensor_cal'].values

    # Use only the after-filter plot (first T value)
    T_label = list(T_VALUES.keys())[0]
    if T_label in res['swi'] and matched_dp.shape[0] == len(res['swi'][T_label]):
        swi_dp_vals = res['swi'][T_label]
        slope_t = vr.get(f'swi_dp_{T_label}_slope', VWC_CAL_B)
        intercept_t = vr.get(f'swi_dp_{T_label}_intercept', VWC_CAL_A)
        vwc_swi_dp = intercept_t + slope_t * swi_dp_vals

        ax.scatter(x_sensor, vwc_swi_dp, c='purple', s=55, alpha=0.9,
                   edgecolors='black', linewidths=0.5)
        lims = [min(ax.get_xlim()[0], ax.get_ylim()[0]),
                max(ax.get_xlim()[1], ax.get_ylim()[1])]
        ax.plot(lims, lims, 'k--', alpha=0.7, lw=1.5)
        slope_sw, int_sw, _, _ = deming_regression(x_sensor, vwc_swi_dp)
        ax.plot(lims, [slope_sw * x + int_sw for x in lims], 'purple', alpha=0.95, lw=2.4)
        r2_key = f'swi_dp_{T_label}_r2'
        r_key = f'swi_dp_{T_label}_r'
        rmse_key = f'swi_dp_{T_label}_rmse'
        ax.set_xlabel('Sensor VWC (m³/m³)', fontsize=12)
        ax.set_ylabel('SWI VWC (m³/m³)', fontsize=12)
        ax.set_title(f'{zone_labels[ax_idx]}\nr={vr.get(r_key, 0):.3f}, R²={vr.get(r2_key, 0):.3f}, '
                     f'RMSE={vr.get(rmse_key, 0):.3f}', fontsize=11)
        ax.set_aspect('equal', adjustable='box')
        ax.grid(True, alpha=0.3)

plt.tight_layout()
fig_path = os.path.join(OUTPUT_DIR, "exp_filter_scatter.png")
fig.savefig(fig_path, dpi=150, facecolor='white', bbox_inches='tight')
fig.savefig(fig_path.replace('.png', '.pdf'))
print(f"Saved: {fig_path}")
plt.close(fig)

# --- 7. Save results ---
print("\n" + "=" * 60)
print("Step 7: Saving results")
print("=" * 60)

output_data = []
for zone_name, res in results.items():
    dates_dp_all = res['dates']
    theta_dp_all = res['theta']
    for i, d in enumerate(dates_dp_all):
        row = {
            'date': d.strftime('%Y-%m-%d') if hasattr(d, 'strftime') else str(d),
            'area': zone_name,
            'orbit': 'all',
            'theta_dp': theta_dp_all[i],
            'vwc_dp': VWC_CAL_A + VWC_CAL_B * theta_dp_all[i],
        }
        for T_label in T_VALUES:
            swi_key = T_label
            if swi_key in res['swi'] and i < len(res['swi'][swi_key]):
                slope_t = res['validation'].get(f'swi_dp_{T_label}_slope', VWC_CAL_B)
                intercept_t = res['validation'].get(f'swi_dp_{T_label}_intercept', VWC_CAL_A)
                row[f'swi_dp_{T_label}'] = res['swi'][swi_key][i]
                row[f'vwc_swi_dp_{T_label}'] = intercept_t + slope_t * res['swi'][swi_key][i]
        output_data.append(row)

output_df = pd.DataFrame(output_data)
csv_path = os.path.join(OUTPUT_DIR, "exp_filter_timeseries.csv")
output_df.to_csv(csv_path, index=False)
print(f"Saved: {csv_path}")

params = {
    "method": "exponential_filter_DpRVIc",
    "references": [
        "Wagner et al. (1999) - Remote Sensing of Environment, 70(2), 191-207",
        "Albergel et al. (2008) - Hydrology and Earth System Sciences, 12(6), 1323-1337",
        "Albergel et al. (2011) - IEEE Trans. Geoscience and Remote Sensing, 49(9), 3358-3367",
    ],
    "T_values_days": T_VALUES,
    "orbit": "all",
    "sensing_depth_cm": SENSING_DEPTH_CM,
    "VWC_calibration": {"a": VWC_CAL_A, "b": VWC_CAL_B},
    "sensor_calibration": {"a": SENSOR_CAL_A, "b": SENSOR_CAL_B},
    "validation": {zone: {k: v for k, v in res['validation'].items()} for zone, res in results.items()},
}
params_path = os.path.join(OUTPUT_DIR, "exp_filter_params.json")
with open(params_path, 'w') as f:
    json.dump(params, f, indent=2)
print(f"Saved: {params_path}")

print("\n" + "=" * 60)
print("DpRVIc EXPONENTIAL FILTER RETRIEVAL COMPLETE")
print("=" * 60)