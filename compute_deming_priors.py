# %% [markdown]
# # Compute Deming Regression Priors for Sentinel Calibration (N-dependent)
# #
# For each calibration window length N, compute the Deming regression of
# MZ1 Sentinel SWI vs MZ1 sensor VWC. Uses zone-averaged time series
# from exp_filter_timeseries.csv (SWI) and in-situ sensor VWC.
# The prior mean comes from the full-season Deming calibration in
# exp_filter_params.json. The SE decreases with N as more data points
# are available, following standard regression uncertainty scaling.

# %%
import numpy as np
import pandas as pd
import json
import os
from datetime import datetime

BASE_DIR = os.path.dirname(os.path.realpath(__file__))
OUTPUT_DIR = os.path.join(BASE_DIR, 'output')

# %% Load zone-averaged Deming calibration (for reference)
with open(os.path.join(OUTPUT_DIR, 'exp_filter_params.json'), 'r') as f:
    exp_params = json.load(f)

# %% Load MZ1 Sentinel SWI (exponential filter output)
swi_df = pd.read_csv(os.path.join(OUTPUT_DIR, 'exp_filter_timeseries.csv'))
swi_mz1 = swi_df[swi_df['area'] == 'MZ1'].copy()
swi_mz1['date'] = pd.to_datetime(swi_mz1['date'])
swi_mz1 = swi_mz1[(swi_mz1['date'] >= '2025-04-24') &
                     (swi_mz1['date'] <= '2025-06-05')].copy()

# Load MZ1 sensor data
import sys
sys.path.insert(0, os.path.join(BASE_DIR, 'swim2'))
from swim2_data import load_sensordata
from Sensordata import ConvertToSerialDate

SENSOR_CAL_A = -0.006
SENSOR_CAL_B = 1.26
OBS_START_DATE = datetime(2025, 4, 24)
OBS_START_SERIAL = ConvertToSerialDate(OBS_START_DATE)

df_sensor = load_sensordata('MZ1')
df_sensor.sort_values(by='Datetime', inplace=True)
df_sensor.reset_index(drop=True, inplace=True)

vwc0 = SENSOR_CAL_A + SENSOR_CAL_B * df_sensor['vwc0 (m3/m3)'].values
vwc1 = SENSOR_CAL_A + SENSOR_CAL_B * df_sensor['vwc1 (m3/m3)'].values
vwc2 = SENSOR_CAL_A + SENSOR_CAL_B * df_sensor['vwc2 (m3/m3)'].values

dates_str = [df_sensor['Datetime'].iloc[i].strftime('%Y-%m-%d') for i in range(len(df_sensor))]
serial_dates = [ConvertToSerialDate(df_sensor['Datetime'].iloc[i]) for i in range(len(df_sensor))]

df_daily = pd.DataFrame({
    'Date': dates_str,
    'Serial_date': serial_dates,
    'VWC0': vwc0,
    'VWC1': vwc1,
    'VWC2': vwc2,
})
df_daily = df_daily[df_daily['Serial_date'] >= OBS_START_SERIAL].copy()
df_daily = df_daily[(df_daily[['VWC0', 'VWC1', 'VWC2']] > 0.01).all(axis=1)]
df_daily = df_daily[(df_daily[['VWC0', 'VWC1', 'VWC2']] < 1.0).all(axis=1)]
df_daily_mean = df_daily.groupby('Date', as_index=False).agg({
    'Serial_date': 'first',
    'VWC0': 'mean',
    'VWC1': 'mean',
    'VWC2': 'mean',
})
df_daily_mean['VWC_daily'] = df_daily_mean[['VWC0', 'VWC1', 'VWC2']].mean(axis=1, skipna=True)
df_daily_mean['date'] = pd.to_datetime(df_daily_mean['Date'])


# %% Deming regression function
def deming_regression(x, y, lambda_ratio=1.0):
    """Deming regression (errors in both x and y).
    lambda_ratio = ratio of variance in x to variance in y.
    Returns: slope, intercept, SE_slope, SE_intercept
    """
    n = len(x)
    if n < 2:
        return None, None, None, None
    x_mean = np.mean(x)
    y_mean = np.mean(y)
    sxx = np.sum((x - x_mean) ** 2)
    syy = np.sum((y - y_mean) ** 2)
    sxy = np.sum((x - x_mean) * (y - y_mean))

    slope = (syy - lambda_ratio * sxx +
             np.sqrt((syy - lambda_ratio * sxx) ** 2 + 4 * lambda_ratio * sxy ** 2)) / (2 * sxy)
    intercept = y_mean - slope * x_mean

    residuals = y - (intercept + slope * x)
    s2 = np.sum(residuals ** 2) / (n - 2)
    se_slope = np.sqrt(s2 / sxx) if sxx > 0 else np.nan
    se_intercept = se_slope * np.sqrt(np.sum(x ** 2) / n)

    return slope, intercept, se_slope, se_intercept


# %% Compute for each N using matched SWI-VWC pairs within window
CAL_DAYS_LIST = [10, 20, 30, 40]

priors = {}
for N in CAL_DAYS_LIST:
    cutoff_date = OBS_START_DATE + pd.Timedelta(days=N - 1)
    swi_in_window = swi_mz1[swi_mz1['date'] <= cutoff_date].copy()

    swi_dates = swi_in_window['date'].values
    swi_vals = swi_in_window['swi_dp_15cm'].values.astype(float)

    matched_swi = []
    matched_vwc = []
    matched_dates = []
    for i, swi_date in enumerate(swi_dates):
        pd_date = pd.Timestamp(swi_date)
        sensor_match = df_daily_mean[np.abs(df_daily_mean['date'] - pd_date) <= pd.Timedelta(days=1)]
        if len(sensor_match) > 0:
            vwc_val = sensor_match['VWC_daily'].iloc[0]
            if not np.isnan(vwc_val):
                matched_swi.append(swi_vals[i])
                matched_vwc.append(vwc_val)
                matched_dates.append(pd_date)

    matched_swi = np.array(matched_swi)
    matched_vwc = np.array(matched_vwc)

    if len(matched_swi) < 2:
        priors[str(N)] = {'a': None, 'b': None, 'SE_a': None, 'SE_b': None,
                           'alpha2': None, 'epsilon2': None, 'n_obs': 0}
        continue

    b, a, SE_b, SE_a = deming_regression(matched_swi, matched_vwc)

    vwc_sentinel = a + b * matched_swi
    residuals = vwc_sentinel - matched_vwc
    date_residual_pairs = sorted(zip(matched_dates, residuals))
    sorted_residuals = np.array([r for _, r in date_residual_pairs])

    if len(sorted_residuals) >= 3:
        diffs = np.diff(sorted_residuals)
        epsilon2 = float(np.var(diffs, ddof=1) / 2)
        alpha2 = float(np.var(residuals, ddof=1) - epsilon2)
        if alpha2 < 0:
            alpha2 = 0.0
    else:
        epsilon2 = float(np.var(residuals, ddof=1))
        alpha2 = 0.0

    print(f"N={N}: a={a:.4f}, b={b:.4f}, SE_a={SE_a:.4f}, SE_b={SE_b:.4f}, "
          f"alpha2={alpha2:.6f}, epsilon2={epsilon2:.6f}, n={len(matched_swi)}")
    priors[str(N)] = {
        'a': float(a),
        'b': float(b),
        'SE_a': float(SE_a) if SE_a is not None else None,
        'SE_b': float(SE_b) if SE_b is not None else None,
        'alpha2': alpha2,
        'epsilon2': epsilon2,
        'n_obs': int(len(matched_swi))
    }

# %% Save
output_path = os.path.join(OUTPUT_DIR, 'deming_priors_by_N.json')
with open(output_path, 'w') as f:
    json.dump(priors, f, indent=2)
print(f"Deming priors saved to {output_path}")