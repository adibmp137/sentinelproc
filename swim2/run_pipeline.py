# %% [markdown]
# # SWIM2 Temporal Sweep Pipeline
# # Runs 5 configurations x N calibration windows for the study area 2025

# %%
import os, sys, time, pickle, json, argparse, csv
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from datetime import datetime
from math import isnan
from collections import OrderedDict

parser = argparse.ArgumentParser()
parser.add_argument('--test', action='store_true', help='Quick test: T=50, seq=3')
parser.add_argument('--configs', type=str, default=None, help='Comma-separated config names (e.g. Ref,2b)')
parser.add_argument('--N-values', type=str, default=None, help='Comma-separated N values (e.g. 30,40,50)')
parser.add_argument('--figures-only', action='store_true', help='Skip DREAM, regenerate figures from saved results')
parser.add_argument('--compute-ci', action='store_true', help='Compute CI.npy from saved ParSet files (no DREAM, no plotting)')
parser.add_argument('--recompute-metrics', action='store_true', help='Recompute validation metrics from saved ParSet files')
parser.add_argument('--parallel', type=int, default=1, help='Number of concurrent DREAM runs (experimental, Windows-only: use subprocesses)')
parser.add_argument('--run-id', type=str, default=None, help='Run ID subfolder (e.g. prod). Outputs go to output/dream_results/<run-id>/')
args = parser.parse_args()

BASE_DIR = os.path.dirname(os.path.realpath(__file__))
os.chdir(BASE_DIR)
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

from swim2_data import (load_soildata, load_eto, load_precipitation,
                        load_irrigation, load_soilobs, load_sensordata,
                        load_sensor_overview, BASE_DIR as DATA_DIR, CASES, YEAR,
                        GROWING_START, PLANTING_DATE, OBS_START_DATE,
                        SENSOR_CAL_A, SENSOR_CAL_B)
from Sensordata import sensordata, sentineldata, ConvertToSerialDate, sensorEQ
from covariance_analysis import (build_sensor_covariance, build_sentinel_covariance_n,
                                  build_sample_covariance, build_combined_covariance,
                                  S_POOLED)
from SWB_model import SWB, initial_func, settings_func, ConvertToSerialDate as SWB_ConvertToSerialDate
from mcmc import Sampler

OUTPUT_DIR = os.path.join(BASE_DIR, '..', 'output', 'dream_results')
if args.run_id:
    OUTPUT_DIR = os.path.join(OUTPUT_DIR, args.run_id)
os.makedirs(OUTPUT_DIR, exist_ok=True)

# %% [markdown]
# # Configuration definitions

# %%
FULL_CONFIGS = OrderedDict({
    'Ref': {
        'case': 'MZ1', 'obsdata': 'Sensor+stalen',
        'lik_sigma_est': False, 'corr_est': False, 'zero_cov': False,
        'Prior': 'LHS', 'prior_file': None,
        'sentinel_cal_on': False, 'n_params': 12,
    },
    '2a': {
        'case': 'MZ2', 'obsdata': 'Samples only',
        'lik_sigma_est': False, 'corr_est': False, 'zero_cov': True,
        'Prior': 'LHS', 'prior_file': None,
        'sentinel_cal_on': False, 'n_params': 12,
    },
    '2b': {
        'case': 'MZ2', 'obsdata': 'Sentinel+stalen',
        'lik_sigma_est': False, 'corr_est': False, 'zero_cov': False,
        'Prior': 'LHS', 'prior_file': None,
        'sentinel_cal_on': True, 'n_params': 14,
    },
    '2c': {
        'case': 'MZ2', 'obsdata': 'Sentinel+stalen',
        'lik_sigma_est': False, 'corr_est': False, 'zero_cov': False,
        'Prior': 'Transfer', 'prior_file': None, 'prior_inflation': 2.0,
        'sentinel_cal_on': True, 'n_params': 14,
    },
    '2d': {
        'case': 'MZ2', 'obsdata': 'Samples only',
        'lik_sigma_est': False, 'corr_est': False, 'zero_cov': True,
        'Prior': 'Transfer', 'prior_file': None, 'prior_inflation': 2.0,
        'sentinel_cal_on': False, 'n_params': 12,
    },
})

CONFIGS = OrderedDict(FULL_CONFIGS)

# Max calibration days per config based on data availability + 7-day validation
# Sentinel data ends Jun 3, so Sentinel configs need cal_end + 7 <= Jun 10 => N <= 70
# Sensor data extends to Jun 29+, so Ref can go to N=100
MAX_N_PER_CONFIG = {
    'Ref': 60,
    '2a': 60,
    '2b': 60,
    '2c': 60,
    '2d': 60,
}

CAL_DAYS_LIST = [10, 20, 30, 40]
if args.N_values:
    CAL_DAYS_LIST = [int(x) for x in args.N_values.split(',')]
if args.configs:
    _allowed = [x.strip() for x in args.configs.split(',')]
    CONFIGS = OrderedDict((k, v) for k, v in CONFIGS.items() if k in _allowed)
TEST_MODE = args.test
TEST_T = 50
TEST_SEQ = 3
if TEST_MODE:
    print(f"*** TEST MODE: T={TEST_T}, seq={TEST_SEQ}, N={CAL_DAYS_LIST}, configs={list(CONFIGS.keys())} ***")

# %% [markdown]
# # Data loading

# %%
df_soil_all = load_soildata()
df_soil_all['year'] = df_soil_all['year'].astype(str)
df_crop = pd.read_csv(os.path.join(DATA_DIR, 'crop_FAO.csv'), encoding='unicode_escape')
df_eto = load_eto()
df_precip = load_precipitation()
df_irrig = load_irrigation()
df_obs_all = load_soilobs()
df_sensor_overview = load_sensor_overview()


def build_df_list(case, year='2025'):
    """Build df_list filtered for a specific case and year."""
    df_soil = df_soil_all[df_soil_all['year'] == year].copy()
    df_soil.reset_index(drop=True, inplace=True)

    df_ = df_eto[df_eto['year'].astype(str) == year].copy()
    df_.reset_index(drop=True, inplace=True)

    df_R = df_precip[df_precip['year'].astype(str) == year].copy()
    df_R.reset_index(drop=True, inplace=True)

    df_I = df_irrig[df_irrig['year'].astype(str) == year].copy()
    df_I.reset_index(drop=True, inplace=True)

    df_obs = df_obs_all[df_obs_all['year'].astype(str) == year].copy()
    df_obs = df_obs[df_obs['Sensornr'] == case].copy()
    df_obs.reset_index(drop=True, inplace=True)

    return [df_R, df_I, df_, df_soil, df_obs, df_crop]


# %% [markdown]
# # Helper functions

# %%
def get_initial_params(case, df_soil, df_crop, df_sensor_teler):
    """Get initial parameter values from initial_func()."""
    # initial_func uses global df_sensor_teler in SWB_model
    import SWB_model
    SWB_model.df_sensor_teler = df_sensor_teler
    ini, teler, opkomst, verantw, p1, p2, part, soil_type, crop_name, \
        irr_method, forecast, forecast_date, g0 = initial_func(
            case, df_soil, df_crop, forecast_on=False, show=False)
    return ini, opkomst, crop_name, soil_type, g0, forecast, irr_method, part


def get_param_bounds(soil_type, ini, opkomst, crop_name, sentinel_cal_on=False):
    """Define parameter bounds. Append a_S/b_S bounds for Sentinel configs."""
    if ini[7] >= np.log(200) and soil_type in ('Z', 'S', 'P'):
        Ksat_LB, Ksat_UB = np.log(200), np.log(4000)
    elif ini[7] >= np.log(100) and ini[7] <= np.log(750) and soil_type in ('L', 'A'):
        Ksat_LB, Ksat_UB = np.log(100), np.log(750)
    elif ini[7] >= np.log(5) and ini[7] <= np.log(150) and soil_type in ('E', 'U'):
        Ksat_LB, Ksat_UB = np.log(5), np.log(150)
    else:
        if soil_type in ('Z', 'S', 'P'):
            Ksat_LB, Ksat_UB = np.log(200), np.log(4000)
        elif soil_type in ('L', 'A'):
            Ksat_LB, Ksat_UB = np.log(100), np.log(750)
        elif soil_type in ('E', 'U'):
            Ksat_LB, Ksat_UB = np.log(5), np.log(150)
        else:
            Ksat_LB, Ksat_UB = np.log(5), np.log(4000)

    if opkomst:
        LB = [0, 0.85, 0.65, ini[3]-0.5, -0.5, -0.5, min(ini[6],0.15), Ksat_LB, 58, 100, 0.3, 0.05]
        UB = [0.05, 1.15, 1.15, ini[3]+0.5, 0.5, 0.5, min(0.4,ini[6]+0.25), Ksat_UB, 91, 200, 0.9, min(0.4,ini[6]+0.25)]
    else:
        if crop_name == 'witloof':
            LB = [0.05, 0.85, 0.65, ini[3]-min(ini[3],21), ini[4]-min(ini[4],7), ini[5]-min(ini[5],7), min(ini[6],0.15), Ksat_LB, 58, 100, 0.3, 0.1]
            UB = [0.4, 1.15, 1.15, ini[3]+10, ini[4]+7, ini[5]+7, min(0.4,ini[6]+0.25), Ksat_UB, 91, 200, 0.9, min(0.4,ini[6]+0.25)]
        else:
            LB = [0.05, 0.85, 0.65, ini[3]-min(ini[3],10), ini[4]-min(ini[4],7), ini[5]-min(ini[5],7), min(ini[6],0.15), Ksat_LB, 58, 100, 0.3, 0.1]
            UB = [0.4, 1.15, 1.15, ini[3]+10, ini[4]+7, ini[5]+7, min(0.4,ini[6]+0.25), Ksat_UB, 91, 200, 0.9, min(0.4,ini[6]+0.25)]

    if sentinel_cal_on:
        LB.extend([-0.2, 0.01])
        UB.extend([0.5, 1.5])

    return np.array(LB), np.array(UB)


def get_prior_file(config_name, N_cal, year='2025'):
    """For Transfer configs, return path to MZ1 posterior for same N_cal."""
    if config_name in ('2c', '2d'):
        folder = os.path.join(OUTPUT_DIR, f'MZ1_ui_{year}_Ref_N{N_cal}days')
        return os.path.join(folder, f'ParSet_MZ1{year}ui.npy')
    return None


def compute_bcRMSD(sim, obs, bias):
    return np.sqrt(np.mean((sim - obs - bias) ** 2))


def compute_bcNSE(sim, obs, bias):
    obs_mean = np.mean(obs)
    return 1 - np.sum((sim - obs - bias) ** 2) / np.sum((obs - obs_mean + bias) ** 2)


# %% [markdown]
# # Validation function

# %%
def compute_validation_metrics(case, config_name, N_cal, ParSetMax, df_list,
                               g0_serial, cal_start_serial, year='2025'):
    """Compute validation metrics against held-out sensor data."""
    # Run SWB with ParSetMax over the full cycle
    n_params = len(ParSetMax) - 2 if len(ParSetMax) > 12 else len(ParSetMax)
    SWC, sw_list, g_list, sensor_data_full, covar_full, df_obs_full, _, _ = SWB(
        ParSetMax[0], ParSetMax[1], ParSetMax[2], ParSetMax[3], ParSetMax[4],
        ParSetMax[5], ParSetMax[6], ParSetMax[7], ParSetMax[8], ParSetMax[9],
        ParSetMax[10], ParSetMax[11],
        sensor=True, cal='gen', sensor_cal=np.empty(0), CI=np.empty(0),
        show=[False, ''], case=case, year=year, forecast=np.empty(0), df_list=df_list)

    # Load validation sensor data for this case
    val_df = load_sensordata(case)
    val_df.sort_values(by='Datetime', inplace=True)
    val_df.reset_index(drop=True, inplace=True)

    dates_str = [val_df['Datetime'].iloc[i].strftime('%Y-%m-%d') for i in range(len(val_df))]
    serial_dates = [ConvertToSerialDate(val_df['Datetime'].iloc[i]) for i in range(len(val_df))]
    vwc0 = SENSOR_CAL_A + SENSOR_CAL_B * val_df['vwc0 (m3/m3)'].values
    vwc1 = SENSOR_CAL_A + SENSOR_CAL_B * val_df['vwc1 (m3/m3)'].values
    vwc2 = SENSOR_CAL_A + SENSOR_CAL_B * val_df['vwc2 (m3/m3)'].values

    df_val = pd.DataFrame({
        'Date': dates_str,
        'Serial_date': serial_dates,
        'VWC0': vwc0,
        'VWC1': vwc1,
        'VWC2': vwc2,
    })
    df_val = df_val[df_val['Serial_date'] >= g0_serial].copy()
    df_val = df_val[~(df_val[['VWC0', 'VWC1', 'VWC2']] <= 0.01).all(axis=1)]
    df_val = df_val[~(df_val[['VWC0', 'VWC1', 'VWC2']] >= 1.0).any(axis=1)]
    df_val.reset_index(drop=True, inplace=True)

    # Daily aggregation — floor serial dates to integer day for matching with g_list
    daily_means = df_val.groupby('Date', as_index=False).agg({
        'Serial_date': 'first',
        'VWC0': 'mean',
        'VWC1': 'mean',
        'VWC2': 'mean',
    })
    daily_means['VWC_daily'] = daily_means[['VWC0', 'VWC1', 'VWC2']].mean(axis=1, skipna=True)
    daily_means['Serial_day'] = np.floor(daily_means['Serial_date']).astype(int)

    sensor_days = daily_means['Serial_day'].values
    sensor_vals = daily_means['VWC_daily'].values

    # Extract simulated SWC at sensor dates using integer-day matching
    g_list_arr = np.array(g_list, dtype=float)
    sim_vals = np.empty(len(sensor_days))
    for i in range(len(sensor_days)):
        idx = np.where(g_list_arr == sensor_days[i])[0]
        if len(idx) > 0:
            sim_vals[i] = SWC[idx[0]]
        else:
            sim_vals[i] = np.nan

    valid = ~(np.isnan(sim_vals) | np.isnan(sensor_vals))
    sim_vals = sim_vals[valid]
    sensor_vals = sensor_vals[valid]
    sensor_days_valid = sensor_days[valid]

    # Bias correction over full cycle
    bias = np.mean(sim_vals - sensor_vals)

    # Calibration and validation masks (using integer day numbers)
    cal_end_day = int(cal_start_serial) + N_cal - 1
    val_end_day = cal_end_day + 7
    cal_mask = sensor_days_valid <= cal_end_day
    val_mask = ((sensor_days_valid > cal_end_day) &
                (sensor_days_valid <= val_end_day))

    metrics = {}
    if np.sum(cal_mask) > 0:
        metrics['bcRMSD_cal_sensor'] = compute_bcRMSD(sim_vals[cal_mask], sensor_vals[cal_mask], bias)
        metrics['bcNSE_cal_sensor'] = compute_bcNSE(sim_vals[cal_mask], sensor_vals[cal_mask], bias)
    else:
        metrics['bcRMSD_cal_sensor'] = np.nan
        metrics['bcNSE_cal_sensor'] = np.nan

    if np.sum(val_mask) > 0:
        metrics['bcRMSD_val_sensor'] = compute_bcRMSD(sim_vals[val_mask], sensor_vals[val_mask], bias)
        metrics['bcNSE_val_sensor'] = compute_bcNSE(sim_vals[val_mask], sensor_vals[val_mask], bias)
        metrics['ME_val_sensor'] = np.mean(sim_vals[val_mask] - sensor_vals[val_mask])
        metrics['MAE_val_sensor'] = np.mean(np.abs(sim_vals[val_mask] - sensor_vals[val_mask]))
        metrics['RMSE_val_sensor'] = np.sqrt(np.mean((sim_vals[val_mask] - sensor_vals[val_mask]) ** 2))
    else:
        metrics['bcRMSD_val_sensor'] = np.nan
        metrics['bcNSE_val_sensor'] = np.nan
        metrics['ME_val_sensor'] = np.nan
        metrics['MAE_val_sensor'] = np.nan
        metrics['RMSE_val_sensor'] = np.nan

    if not np.isnan(metrics.get('bcRMSD_cal_sensor', np.nan)) and metrics.get('bcRMSD_val_sensor', 0) > 0:
        metrics['OR_sensor'] = metrics['bcRMSD_cal_sensor'] / metrics['bcRMSD_val_sensor']
    else:
        metrics['OR_sensor'] = np.nan

    return metrics


# %% [markdown]
# # run_dream() function

# %%
def compute_and_save_ci(case, config_name, N_cal, ParSet, folder_path,
                        crop_name, case_cfg=None, df_list_cfg=None, n_ensemble=100):
    parset_file = os.path.join(folder_path, f'ParSet_{case}{YEAR}{crop_name}.npy')
    ci_file = os.path.join(folder_path, 'CI.npy')
    if not os.path.exists(parset_file):
        np.save(ci_file, np.empty(0))
        return np.empty(0)
    if os.path.exists(ci_file):
        return np.load(ci_file, allow_pickle=True)
    ParSet_full = np.load(parset_file) if ParSet is None else ParSet
    istart = int(0.5 * ParSet_full.shape[0])
    ParSet50 = ParSet_full[istart:, :-2]
    n_avail = min(n_ensemble, len(ParSet50))
    rng = np.random.default_rng(42)
    indices = rng.choice(len(ParSet50), size=n_avail, replace=False)
    indices = np.sort(indices)
    if case_cfg is None:
        case_cfg = case
    if df_list_cfg is None:
        df_list_cfg = build_df_list(case_cfg)

    def _run_swb(i, idx_i):
        params = ParSet50[idx_i]
        try:
            SWC_i, sw_i, _, _, _, _, _, _ = SWB(
                params[0], params[1], params[2], params[3],
                params[4], params[5], params[6], params[7],
                params[8], params[9], params[10], params[11],
                sensor=False, cal='', sensor_cal=np.empty(0), CI=np.empty(0),
                show=[False, ''], case=case_cfg, year=YEAR,
                forecast=np.empty(0), df_list=df_list_cfg)
            return SWC_i, sw_i
        except Exception:
            return None, None

    from joblib import Parallel, delayed
    n_jobs = max(1, min(os.cpu_count() - 1, 4))
    print(f"    Running {n_avail} SWB ensemble members with {n_jobs} parallel jobs...")
    results = Parallel(n_jobs=n_jobs, backend='loky')(delayed(_run_swb)(i, idx) for i, idx in enumerate(indices))
    SWC_list = [r[0] for r in results if r[0] is not None]
    sw_list_list = [r[1] for r in results if r[1] is not None]
    print(f"    CI ensemble: {len(SWC_list)}/{n_avail} successful")

    CI = np.empty(0)
    if len(SWC_list) > 2:
        SWC_df = pd.DataFrame(SWC_list)
        SWC_sort = SWC_df.sort_values(by=SWC_df.columns.to_list()).reset_index(drop=True)
        excl = max(1, np.round(SWC_df.shape[0]*0.025).astype(int))
        SWC_95 = SWC_sort[excl:-excl] if excl < SWC_sort.shape[0]//2 else SWC_sort
        SWC_95_min = np.min(SWC_95, axis=0).values
        SWC_95_max = np.max(SWC_95, axis=0).values
        sw_df = pd.DataFrame(sw_list_list)
        sw_sort = sw_df.sort_values(by=sw_df.columns.to_list()).reset_index(drop=True)
        sw_95 = sw_sort[excl:-excl] if excl < sw_sort.shape[0]//2 else sw_sort
        sw_95_min = np.min(sw_95, axis=0).values
        sw_95_max = np.max(sw_95, axis=0).values
        CI_SWC = np.array([SWC_95_min, SWC_95_max])
        CI_sw = np.array([sw_95_min, sw_95_max])
        CI = np.array([CI_SWC, CI_sw]).reshape(4, -1)
    np.save(ci_file, CI)
    return CI


def generate_diagnostics(folder_path, case, config_name, N_cal, n_params, LB, UB):
    par_names_soil = ['fc','log(Ksat)','CN','GWT_max','Zr_max','v_ini']
    par_names_sentinel = ['a_S','b_S']
    par_names_crop = ['Kcb_ini','Kcb_mid','Kcb_end','L_ini','L_dev','L_mid']

    crop_name = 'ui'
    ps_file = os.path.join(folder_path, f'ParSet_{case}{YEAR}{crop_name}.npy')
    if not os.path.exists(ps_file):
        return
    ParSet = np.load(ps_file)
    istart = int(0.5 * ParSet.shape[0])
    ParSetMax = ParSet[np.argmax(ParSet[:, -1])]
    ParSet50 = ParSet[istart:, :-2]
    ini = ParSet[0, :n_params]

    if n_params > 12:
        par_names = par_names_crop + par_names_soil + par_names_sentinel
    else:
        par_names = par_names_crop + par_names_soil

    prefix = f'{case}{YEAR}{crop_name}'

    total_params = n_params
    nrows = (total_params + 3) // 4
    ncols = 4

    plt.figure(figsize=(12, 2 * nrows))
    for i in range(n_params):
        plt.subplot(nrows, ncols, i + 1)
        plt.title(par_names[i], fontsize=8)
        lo, hi = np.percentile(ParSet50[:, i], [1, 99])
        margin = (hi - lo) * 0.15
        plt.xlim(lo - margin, hi + margin)
        plt.hist(ParSet50[:, i], color="orange", bins=30)
        plt.axvline(ini[i], color='r', mew=3, ms=10)
        plt.axvline(ParSetMax[i], color='c', mew=3, ms=10)
        plt.tick_params(axis='both', labelsize=7, pad=1)

    plt.tight_layout()
    plt.subplots_adjust(top=0.92)
    plt.suptitle(f'Posterior distribution — {config_name} N={N_cal}',
                 fontsize='x-large', weight='bold')
    plt.savefig(os.path.join(folder_path, f'posteriors_{prefix}.png'), dpi=300)
    plt.close()

    # Correlation: all free params
    par_plot = n_params
    x = np.transpose(ParSet50[:, :par_plot])
    corr = np.corrcoef(x).round(decimals=2)
    fig_size = max(8, par_plot * 0.7)
    fig, ax = plt.subplots(figsize=(fig_size, fig_size * 0.8))
    im = plt.imshow(corr, cmap="bwr")
    ax.set_xticks(np.arange(par_plot))
    ax.set_yticks(np.arange(par_plot))
    ax.set_xticklabels(par_names[:par_plot], rotation=45, ha="right", fontsize=7)
    ax.set_yticklabels(par_names[:par_plot], fontsize=7)
    for ii in range(par_plot):
        for jj in range(par_plot):
            ax.text(jj, ii, corr[ii, jj], ha="center", va="center", color="k", fontsize=6)
    plt.colorbar(im, ax=ax, format='% .2f')
    im.set_clim(-1, 1)
    ax.set_title(f'Correlation heatmap — {config_name} N={N_cal}', fontweight="bold")
    fig.tight_layout()
    plt.savefig(os.path.join(folder_path, f'correlation_{prefix}.png'), dpi=300)
    plt.close()

    # ParSetMax-95
    p95_lo = np.percentile(ParSet50, 2.5, axis=0)
    p95_hi = np.percentile(ParSet50, 97.5, axis=0)
    with open(os.path.join(folder_path, f'ParSetMax-95_{prefix}.csv'), 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['Parameter', 'Max LLH', 'Min 95%', 'Max 95%'])
        for i in range(n_params):
            writer.writerow([par_names[i], ParSetMax[i], p95_lo[i], p95_hi[i]])


def run_dream(case, config_name, N_cal, config_dict, T_val=4000, seq_val=6):
    """Run DREAM-ZS for one (case, config, N_cal) combination."""
    n_params = config_dict['n_params']
    prior_type = config_dict.get('Prior', 'LHS')
    print(f"\n{'='*60}")
    print(f"  {config_name} | case={case} | N={N_cal} | {n_params} params | prior={prior_type}")
    print(f"  obs={config_dict['obsdata']} | T={T_val} | seq={seq_val} | ndraw={seq_val*T_val}")
    print(f"{'='*60}")
    run_start = time.time()

    df_list = build_df_list(case)

    df_soil_case = df_soil_all[df_soil_all['year'].astype(str) == YEAR].copy()
    ini, opkomst, crop_name, soil_type, g0, forecast, irr_method, part = \
        get_initial_params(case, df_soil_case, df_crop, df_sensor_overview)

    LB, UB = get_param_bounds(soil_type, ini, opkomst, crop_name,
                              sentinel_cal_on=config_dict['sentinel_cal_on'])

    if config_dict['sentinel_cal_on']:
        ini = list(ini) + [0.054, 0.487]
    else:
        ini = list(ini)

    prior_file = None
    prior_inflation = config_dict.get('prior_inflation', 1.0)
    if config_dict['Prior'] == 'Transfer':
        prior_file = get_prior_file(config_name, N_cal)
        if prior_file is not None and not os.path.exists(prior_file):
            raise FileNotFoundError(f"Prior file {prior_file} not found. Run Reference first.")

    seq = seq_val
    T = T_val
    ndraw = seq * T
    steps = 100
    jr_scale = 0.8
    rng_seed = 11

    g0_serial = ConvertToSerialDate(PLANTING_DATE)
    cal_start_serial = ConvertToSerialDate(OBS_START_DATE)

    q = Sampler(
        df_list=df_list, data_dir=BASE_DIR, case=case, year=YEAR,
        ini=ini, LB=LB, UB=UB, opkomst=opkomst, crop_name=crop_name,
        forecast=forecast,
        validation_days=int(N_cal),
        cal_start_serial=cal_start_serial,
        CaseStudy=1, seq=seq, ndraw=ndraw, Prior=config_dict['Prior'],
        prior_file=prior_file,
        prior_inflation=prior_inflation,
        steps=steps, lik_sigma_est=config_dict['lik_sigma_est'],
        corr_est=config_dict['corr_est'],
        DREAM_obsdata=config_dict['obsdata'],
        jr_scale=jr_scale, rng_seed=rng_seed, DoParallel=True,
        cal='gen', cal_par_on=False
    )

    Sequences, Z, OutDiag, fx, MCMCPar, MCMCVar = q.sample()

    with open(os.path.join(BASE_DIR, 'dreamzs_out.pkl'), 'rb') as f:
        tmp_obj = pickle.load(f)
    Sequences = tmp_obj['Sequences']
    Z = tmp_obj['Z']
    OutDiag = tmp_obj['OutDiag']
    fx = tmp_obj['fx']
    MCMCPar = tmp_obj['MCMCPar']
    Measurement = tmp_obj['Measurement']
    Extra = tmp_obj['Extra']
    del tmp_obj

    from mcmc_func import Genparset
    idx = np.argwhere(Sequences[:, 0, 0] != 0)
    Sequences = Sequences[idx[:, 0], :, :]
    ParSet = Genparset(Sequences)

    istart = int(0.5 * ParSet.shape[0])
    ParSet50 = ParSet[istart:, :-2]
    ParSetMax = ParSet[np.argmax(ParSet[:, -1])]

    folder_name = f'{case}_ui_{YEAR}_{config_name}_N{N_cal}days'
    folder_path = os.path.join(OUTPUT_DIR, folder_name)
    os.makedirs(folder_path, exist_ok=True)

    np.save(os.path.join(folder_path, f'ParSet_{case}{YEAR}{crop_name}.npy'), ParSet)
    np.save(os.path.join(folder_path, f'ParSet_MaxLL_{case}{YEAR}{crop_name}.npy'), ParSetMax)

    ParSetMax_for_metrics = ParSetMax

    istart_diag = int(0.8 * ndraw / (MCMCPar.steps * MCMCPar.seq))
    iend_diag = int(ndraw / (MCMCPar.steps * MCMCPar.seq))
    AR = np.mean(OutDiag.AR[istart_diag:iend_diag, 1])
    print(f"  Acceptance rate: {AR:.2f}%")

    df_meas = pd.DataFrame({
        'Validation days': [N_cal],
        'N_total': [Measurement.N],
        'N_sensor': [Measurement.n],
        'N_samples': [Measurement.N - Measurement.n]
    })
    df_meas.to_csv(os.path.join(folder_path, 'Measurements_used_DREAM.csv'), index=False)
    pd.DataFrame(Measurement.Sigma).to_csv(os.path.join(folder_path, 'Covar_DREAM.csv'), index=False)

    g0_serial = ConvertToSerialDate(PLANTING_DATE)
    cal_start_serial = ConvertToSerialDate(OBS_START_DATE)
    metrics = compute_validation_metrics(case, config_name, N_cal, ParSetMax_for_metrics, df_list,
                                         g0_serial, cal_start_serial, year=YEAR)
    metrics_df = pd.DataFrame(metrics, index=[0])
    metrics_df.to_csv(os.path.join(folder_path, f'metrics_{case}_{YEAR}_{config_name}_N{N_cal}days.csv'), index=False)

    print(f"  bcRMSD_val={metrics.get('bcRMSD_val_sensor', float('nan')):.4f} | "
          f"OR={metrics.get('OR_sensor', float('nan')):.3f} | "
          f"AR={AR:.1f}%")

    if config_name == 'Ref':
        np.save(os.path.join(folder_path, f'ParSet_{case}{YEAR}{crop_name}.npy'), ParSet)

    print(f"  Generating diagnostic figures...")
    generate_diagnostics(folder_path, case, config_name, N_cal, n_params, LB, UB)

    print(f"  Computing 95% CI ensemble (200 SWB runs)...")
    ci_start = time.time()
    compute_and_save_ci(case, config_name, N_cal, ParSet, folder_path,
                        crop_name, case_cfg=case, df_list_cfg=None)
    print(f"  CI computed in {time.time()-ci_start:.1f}s")

    run_elapsed = time.time() - run_start
    print(f"  Total run time: {run_elapsed/60:.1f} min")

    return {'folder': folder_name, 'N_cal': N_cal, 'config': config_name,
            'metrics': metrics, 'AR': AR}


# %% [markdown]
# # Main DREAM loop

# %%
def preflight_check(case, config_name, N_cal, config_dict):
    """Check that a config has enough data points. Returns (n_obs, n_sensor, n_samples, n_val)."""
    df_list = build_df_list(case)
    df_obs = df_list[4]
    g0_serial = ConvertToSerialDate(PLANTING_DATE)
    cal_start_serial = ConvertToSerialDate(OBS_START_DATE)
    cal_end = cal_start_serial + N_cal - 1
    val_end = cal_end + 7
    n_samples = len(df_obs[df_obs['Date'] <= cal_end]) if 'Date' in df_obs.columns else 0

    # Count validation observations (days N+1 to N+7 after calibration end)
    if 'Date' in df_obs.columns:
        val_mask = (df_obs['Date'] > cal_end) & (df_obs['Date'] <= val_end)
        n_val_samples = len(df_obs[val_mask])
    else:
        n_val_samples = 0

    if config_dict['obsdata'] == 'Samples only':
        n_sensor = 0
        n_obs = n_samples
        n_val = n_val_samples
    elif config_dict['obsdata'] == 'Sentinel+stalen':
        # Load Sentinel data to count validation days
        try:
            from datetime import datetime as _dt
            sentinel_df = pd.read_csv(os.path.join(BASE_DIR, '..', 'output', 'exp_filter_timeseries.csv'))
            area_map = {'MZ1': 'MZ1', 'MZ2': 'MZ2'}
            sentinel_df = sentinel_df[sentinel_df['area'] == area_map.get(case, case)]
            sentinel_df['date'] = pd.to_datetime(sentinel_df['date'])
            sentinel_serial = sentinel_df['date'].apply(ConvertToSerialDate).values.astype(float)
            n_sensor = 1
            n_obs = n_sensor + n_samples
            n_val_sensor = int(np.sum((sentinel_serial > cal_end) & (sentinel_serial <= val_end)))
            n_val = n_val_sensor + n_val_samples
        except Exception:
            n_sensor = 1
            n_obs = n_sensor + n_samples
            n_val = n_val_samples
    else:
        # Sensor+stalen: load sensor data to count validation days
        try:
            val_df = load_sensordata(case)
            val_df['Serial_date'] = val_df['Datetime'].apply(ConvertToSerialDate)
            val_serial = val_df['Serial_date'].values.astype(float)
            n_sensor = 1
            n_obs = n_sensor + n_samples
            n_val_sensor = int(np.sum((val_serial > cal_end) & (val_serial <= val_end)))
            n_val = n_val_sensor + n_val_samples
        except Exception:
            n_sensor = 1
            n_obs = n_sensor + n_samples
            n_val = n_val_samples

    return n_obs, n_sensor, n_samples, n_val

results = {}

if args.recompute_metrics:
    pass  # skip DREAM, go to recomputation block at end
elif not args.figures_only:
    DEMING_PRIORS = {}
    deming_file = os.path.join(BASE_DIR, '..', 'output', 'deming_priors_by_N.json')
    if os.path.exists(deming_file):
        with open(deming_file, 'r') as f:
            DEMING_PRIORS = json.load(f)
    else:
        print("WARNING: deming_priors_by_N.json not found. Run compute_deming_priors.py first.")
        print("Config 2c will fall back to uniform priors for a_S/b_S.")

    if args.parallel > 1:
        print(f"NOTE: --parallel {args.parallel} requested but parallel mode is not yet supported on Windows. Running sequentially.")

    pipeline_start = time.time()
    for N_cal in CAL_DAYS_LIST:
        for config_name, config_dict in CONFIGS.items():
            if config_name in ('2c', '2d'):
                prior_file = get_prior_file(config_name, N_cal)
                ref_folder = os.path.join(OUTPUT_DIR, f"MZ1_ui_{YEAR}_Ref_N{N_cal}days")
                ref_ps = os.path.join(ref_folder, f'ParSet_MZ1{YEAR}ui.npy')
                if not os.path.exists(ref_ps):
                    base_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'output', 'dream_results')
                    ref_folder = os.path.join(base_dir, f"MZ1_ui_{YEAR}_Ref_N{N_cal}days")
                    ref_ps = os.path.join(ref_folder, f'ParSet_MZ1{YEAR}ui.npy')
                if not os.path.exists(ref_ps):
                    print(f"  Skipping {config_name} N={N_cal}: Ref ParSet not found (run Ref first)")
                    continue
                if prior_file is not None and not os.path.exists(prior_file):
                    print(f"  Skipping {config_name} N={N_cal}: Ref posterior not found")
                    continue

            if config_name == '2c' and str(N_cal) not in DEMING_PRIORS:
                print(f"  WARNING: No Deming prior for N={N_cal}. Using uniform priors for a_S/b_S.")

            max_n = MAX_N_PER_CONFIG.get(config_name, 100)
            if N_cal > max_n:
                print(f"  Skipping {config_name} N={N_cal}: exceeds max N={max_n} (insufficient validation data)")
                continue

            n_obs, n_sensor, n_samples, n_val = preflight_check(
                config_dict['case'], config_name, N_cal, config_dict)
            if n_obs == 0:
                print(f"  Skipping {config_name} N={N_cal}: 0 observations (sensor={n_sensor}, samples={n_samples})")
                continue
            if n_val == 0:
                print(f"  WARNING: {config_name} N={N_cal}: 0 validation observations (fewer than 7 days past cal window)")
            print(f"  Data: {n_sensor} sensor series, {n_samples} samples, {n_obs} cal obs, {n_val} val obs")

            start_time = time.time()

            try:
                _T = TEST_T if TEST_MODE else 4000
                _seq = TEST_SEQ if TEST_MODE else 6
                result = run_dream(config_dict['case'], config_name, N_cal, config_dict,
                                   T_val=_T, seq_val=_seq)
                results[f'{config_name}_{N_cal}'] = result
                elapsed = time.time() - start_time
                print(f"  Wall time: {elapsed/60:.1f} min (DREAM+CI+metrics)")
            except Exception as e:
                print(f"  ERROR: {e}")
                import traceback
                traceback.print_exc()
                results[f'{config_name}_{N_cal}'] = None
                continue

    print(f"\nTotal DREAM time: {(time.time() - pipeline_start)/60:.1f} minutes")
else:
    print("Skipping DREAM runs (--figures-only mode).")


# %% [markdown]
# # Collect all metrics into master DataFrame

# %%
all_metrics = []
for key, result in results.items():
    if result is not None:
        row = {
            'config': result['config'],
            'N_cal': result['N_cal'],
            **result['metrics']
        }
        all_metrics.append(row)

if len(all_metrics) > 0:
    master_df = pd.DataFrame(all_metrics)
    master_df.to_csv(os.path.join(OUTPUT_DIR, 'master_metrics.csv'), index=False)
    print("\nMaster metrics saved.")
    print(master_df.to_string())
else:
    print("\nNo results to save.")

print("\nPipeline complete.")

# %% [markdown]
# # Part 3: Figures and Tables (only if there are results)

# %%
if len(all_metrics) == 0 and not os.path.exists(os.path.join(OUTPUT_DIR, 'master_metrics.csv')):
    print("\nNo DREAM results to generate figures from. Exiting.")
    import sys
    sys.exit(0)

# If metrics exist but master_metrics.csv is just a header, also skip figures
if len(all_metrics) == 0:
    _mcsv = os.path.join(OUTPUT_DIR, 'master_metrics.csv')
    if os.path.exists(_mcsv):
        _mdf = pd.read_csv(_mcsv)
        if len(_mdf) == 0 or 'config' not in _mdf.columns:
            print("\nNo DREAM results in CSV. Exiting.")
            import sys
            sys.exit(0)

# %% [markdown]
# ## 3.1 Collect metrics into master DataFrame (also loads from disk if re-running)

# %%
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
plt.rcParams.update({'font.size': 11})

PAR_NAMES_12 = ['Kcb_ini', 'Kcb_mid', 'Kcb_end', 'L_ini', 'L_dev', 'L_mid',
                'fc', 'log(Ksat)', 'CN', 'GWT_max', 'Zr_max', 'v_ini']
PAR_NAMES_14 = PAR_NAMES_12 + ['a_S', 'b_S']

# Try loading from saved CSV files if direct results are empty
if len(all_metrics) == 0:
    for config_name in CONFIGS:
        for N_cal in CAL_DAYS_LIST:
            folder = f"{CONFIGS[config_name]['case']}_ui_{YEAR}_{config_name}_N{N_cal}days"
            mfile = os.path.join(OUTPUT_DIR, folder,
                                 f"metrics_{CONFIGS[config_name]['case']}_{YEAR}_{config_name}_N{N_cal}days.csv")
            if os.path.exists(mfile):
                m = pd.read_csv(mfile)
                m['config'] = config_name
                m['N_cal'] = N_cal
                all_metrics.append(m.iloc[0].to_dict())
            meas_file = os.path.join(folder, 'Measurements_used_DREAM.csv')
            if os.path.exists(meas_file):
                meas = pd.read_csv(meas_file)
                if len(all_metrics) > 0:
                    all_metrics[-1]['N_obs'] = int(meas['N_total'].iloc[0])
                    all_metrics[-1]['N_sensor_obs'] = int(meas['N_sensor'].iloc[0])
                    all_metrics[-1]['N_samples_obs'] = int(meas['N_samples'].iloc[0])

master_df = pd.DataFrame(all_metrics)
if 'N_obs' not in master_df.columns:
    master_df['N_obs'] = np.nan
    master_df['N_sensor_obs'] = np.nan
    master_df['N_samples_obs'] = np.nan

master_df.to_csv(os.path.join(OUTPUT_DIR, 'master_metrics.csv'), index=False)

CONFIG_COLORS = {'Ref': '#1f77b4', '2a': '#ff7f0e', '2b': '#2ca02c',
                 '2c': '#d62728', '2d': '#9467bd'}
CONFIG_MARKERS = {'Ref': 'o', '2a': 's', '2b': '^', '2c': 'D', '2d': 'v'}

# %% [markdown]
# ## 3.2 Figure 7.1: bcRMSD_val vs. calendar days

# %%
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5), sharey=True)

for cfg in ['Ref', '2a', '2b', '2c', '2d']:
    sub = master_df[master_df['config'] == cfg].sort_values('N_cal')
    if len(sub) == 0:
        continue
    ax1.plot(sub['N_cal'], sub['bcRMSD_val_sensor'],
             marker=CONFIG_MARKERS[cfg], color=CONFIG_COLORS[cfg], label=cfg, linewidth=1.5)

ax1.set_xlabel('Calibration days (N)')
ax1.set_ylabel('bcRMSD validation (m3/m3)')
ax1.set_title('Sensor validation')
ax1.legend()
ax1.grid(True, alpha=0.3)
ax1.set_ylim(bottom=0)

plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, 'fig7_1_bcRMSD_val_vs_N.png'), dpi=300)
plt.close()
print("Figure 7.1 saved.")

# %% [markdown]
# ## 3.3 Figure 7.2: bcRMSD_val vs. observation count

# %%
fig, ax = plt.subplots(figsize=(8, 5))

for cfg in ['Ref', '2a', '2b', '2c', '2d']:
    sub = master_df[master_df['config'] == cfg].sort_values('N_cal')
    if len(sub) == 0 or 'N_obs' not in sub.columns:
        continue
    ax.plot(sub['N_obs'], sub['bcRMSD_val_sensor'],
            marker=CONFIG_MARKERS[cfg], color=CONFIG_COLORS[cfg], label=cfg, linewidth=1.5)

ax.set_xlabel('Total calibration observations')
ax.set_ylabel('bcRMSD validation (m3/m3)')
ax.set_title('Figure 7.2: bcRMSD_val vs observation count')
ax.legend()
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, 'fig7_2_bcRMSD_val_vs_Nobs.png'), dpi=300)
plt.close()
print("Figure 7.2 saved.")

# %% [markdown]
# ## 3.4 Figure 7.3: bcRMSD_val vs. lead day
# Requires running SWB with ParSetMax for each validation day individually.

# %%
def compute_validation_per_lead_day(case, config_name, N_cal, ParSetMax, df_list, g0_serial, cal_start_serial, year='2025'):
    """Compute bcRMSD for each validation day 1..7 after calibration window."""
    from Sensordata import ConvertToSerialDate as CSD
    from swim2_data import load_sensordata, SENSOR_CAL_A, SENSOR_CAL_B

    SWC, sw_list, g_list, sensor_data_full, covar_full, df_obs_full, _, _ = SWB(
        ParSetMax[0], ParSetMax[1], ParSetMax[2], ParSetMax[3], ParSetMax[4],
        ParSetMax[5], ParSetMax[6], ParSetMax[7], ParSetMax[8], ParSetMax[9],
        ParSetMax[10], ParSetMax[11],
        sensor=True, cal='gen', sensor_cal=np.empty(0), CI=np.empty(0),
        show=[False, ''], case=case, year=year, forecast=np.empty(0), df_list=df_list)

    val_df = load_sensordata(case)
    val_df.sort_values(by='Datetime', inplace=True)
    val_df.reset_index(drop=True, inplace=True)

    vwc0 = SENSOR_CAL_A + SENSOR_CAL_B * val_df['vwc0 (m3/m3)'].values
    vwc1 = SENSOR_CAL_A + SENSOR_CAL_B * val_df['vwc1 (m3/m3)'].values
    vwc2 = SENSOR_CAL_A + SENSOR_CAL_B * val_df['vwc2 (m3/m3)'].values
    dates_str = [val_df['Datetime'].iloc[i].strftime('%Y-%m-%d') for i in range(len(val_df))]
    serial_dates = [CSD(val_df['Datetime'].iloc[i]) for i in range(len(val_df))]

    df_val = pd.DataFrame({'Date': dates_str, 'Serial_date': serial_dates,
                            'VWC0': vwc0, 'VWC1': vwc1, 'VWC2': vwc2})
    df_val = df_val[df_val['Serial_date'] >= g0_serial].copy()
    df_val = df_val[(df_val[['VWC0', 'VWC1', 'VWC2']] > 0.01).all(axis=1)]
    df_val = df_val[(df_val[['VWC0', 'VWC1', 'VWC2']] < 1.0).all(axis=1)]
    df_val.reset_index(drop=True, inplace=True)

    daily_means = df_val.groupby('Date', as_index=False).agg({
        'Serial_date': 'first', 'VWC0': 'mean', 'VWC1': 'mean', 'VWC2': 'mean'})
    daily_means['VWC_daily'] = daily_means[['VWC0', 'VWC1', 'VWC2']].mean(axis=1, skipna=True)
    daily_means['Serial_day'] = np.floor(daily_means['Serial_date']).astype(int)

    sensor_days = daily_means['Serial_day'].values
    sensor_vals = daily_means['VWC_daily'].values
    g_list_arr = np.array(g_list, dtype=float)

    sim_vals = np.empty(len(sensor_days))
    for i in range(len(sensor_days)):
        idx = np.where(g_list_arr == sensor_days[i])[0]
        if len(idx) > 0:
            sim_vals[i] = SWC[idx[0]]
        else:
            sim_vals[i] = np.nan

    valid = ~(np.isnan(sim_vals) | np.isnan(sensor_vals))
    sim_vals = sim_vals[valid]
    sensor_vals = sensor_vals[valid]
    sensor_days_valid = sensor_days[valid]

    bias = np.mean(sim_vals - sensor_vals)

    cal_end_day = int(cal_start_serial) + N_cal - 1
    lead_day_rmsd = {}
    for lead in range(1, 8):
        mask = ((sensor_days_valid > cal_end_day) &
                (sensor_days_valid <= (cal_end_day + lead)))
        if np.sum(mask) > 0:
            lead_day_rmsd[lead] = np.sqrt(np.mean((sim_vals[mask] - sensor_vals[mask] - bias) ** 2))
        else:
            lead_day_rmsd[lead] = np.nan
    return lead_day_rmsd

# Only compute if we have results
if len(results) > 0:
    fig, ax = plt.subplots(figsize=(8, 5))
    N_fixed = 40
    for cfg in ['Ref', '2a', '2b', '2c', '2d']:
        key = f'{cfg}_{N_fixed}'
        if key not in results or results[key] is None:
            continue
        r = results[key]
        folder = os.path.join(OUTPUT_DIR, r['folder'])
        cfg_key = results[key]['config']
        ps_file = os.path.join(folder, f'ParSet_MaxLL_{cfg_key}_{YEAR}ui.npy')
        if not os.path.exists(ps_file):
            continue
        ParSetMax = np.load(ps_file)
        case_cfg = FULL_CONFIGS[cfg]['case']
        df_list_cfg = build_df_list(case_cfg)
        df_soil_case = df_soil_all[df_soil_all['year'].astype(str) == YEAR].copy()
        ini_cfg, opkomst_cfg, crop_name_cfg, soil_type_cfg, g0_cfg, forecast_cfg, irr_method_cfg, part_cfg = \
            get_initial_params(case_cfg, df_soil_case, df_crop, df_sensor_overview)
        g0_serial_cfg = ConvertToSerialDate(PLANTING_DATE)
        cal_start_serial_cfg = ConvertToSerialDate(OBS_START_DATE)
        lead_rmsd = compute_validation_per_lead_day(case_cfg, cfg, N_fixed, ParSetMax,
                                                     df_list_cfg, g0_serial_cfg, cal_start_serial_cfg)
        leads = list(lead_rmsd.keys())
        rmsds = list(lead_rmsd.values())
        ax.plot(leads, rmsds, marker=CONFIG_MARKERS[cfg], color=CONFIG_COLORS[cfg],
                label=cfg, linewidth=1.5)

    ax.set_xlabel('Lead day')
    ax.set_ylabel('bcRMSD (m3/m3)')
    ax.set_title(f'Figure 7.3: bcRMSD vs lead day (N={N_fixed})')
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, 'fig7_3_bcRMSD_val_vs_lead_day.png'), dpi=300)
    plt.close()
    print("Figure 7.3 saved.")

# %% [markdown]
# ## 3.5 Figure 7.4: Overfitting ratio vs. calendar days

# %%
fig, ax = plt.subplots(figsize=(8, 5))

for cfg in ['Ref', '2a', '2b', '2c', '2d']:
    sub = master_df[master_df['config'] == cfg].sort_values('N_cal')
    if len(sub) == 0:
        continue
    ax.plot(sub['N_cal'], sub['OR_sensor'],
            marker=CONFIG_MARKERS[cfg], color=CONFIG_COLORS[cfg], label=cfg, linewidth=1.5)

ax.axhline(y=1.0, color='k', linestyle='--', alpha=0.5)
ax.set_xlabel('Calibration days (N)')
ax.set_ylabel('Overfitting ratio (OR)')
ax.set_title('Figure 7.4: Overfitting ratio')
ax.legend()
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, 'fig7_4_OR_vs_N.png'), dpi=300)
plt.close()
print("Figure 7.4 saved.")

# %% [markdown]
# ## 3.6 Figure 7.5: Parameter KLD comparison (2b vs 2c)

# %%
def compute_KLD(posterior_samples, lb, ub):
    """KLD between posterior and uniform prior on [lb, ub]."""
    hist, bin_edges = np.histogram(posterior_samples, bins=50, density=True)
    bin_width = bin_edges[1] - bin_edges[0]
    prior_density = 1.0 / (ub - lb)
    kld = np.sum(hist * bin_width * np.log((hist + 1e-10) / prior_density))
    return kld

# Collect KLDs for 2b and 2c across N
kld_data = []
for N_cal in CAL_DAYS_LIST:
    for cfg in ['2b', '2c']:
        folder = os.path.join(OUTPUT_DIR, f"MZ2_ui_{YEAR}_{cfg}_N{N_cal}days")
        ps_file = os.path.join(folder, f'ParSet_MZ2{YEAR}ui.npy')
        if not os.path.exists(ps_file):
            continue
        ParSet = np.load(ps_file)
        istart = int(0.5 * ParSet.shape[0])
        ParSet50 = ParSet[istart:, :-2]
        n_params = ParSet50.shape[1]
        names = PAR_NAMES_12 if n_params == 12 else PAR_NAMES_14

        # Get bounds for this config
        df_soil_case = df_soil_all[df_soil_all['year'].astype(str) == YEAR].copy()
        ini_cfg, opkomst_cfg, crop_name_cfg, soil_type_cfg, g0_cfg, _, _, _ = \
            get_initial_params('MZ2', df_soil_case, df_crop, df_sensor_overview)
        sent_on = FULL_CONFIGS[cfg]['sentinel_cal_on']
        LB, UB = get_param_bounds(soil_type_cfg, ini_cfg, opkomst_cfg, crop_name_cfg,
                                   sentinel_cal_on=sent_on)

        for p in range(min(12, n_params)):
            kld_val = compute_KLD(ParSet50[:, p], LB[p], UB[p])
            kld_data.append({'config': cfg, 'N_cal': N_cal, 'parameter': names[p],
                             'KLD': kld_val, 'param_idx': p})

if len(kld_data) > 0:
    kld_df = pd.DataFrame(kld_data)

    fig, ax = plt.subplots(figsize=(12, 6))
    for cfg in ['2b', '2c']:
        for p_idx in range(12):
            sub = kld_df[(kld_df['config'] == cfg) & (kld_df['param_idx'] == p_idx)]
            if len(sub) == 0:
                continue
            linestyle = '-' if cfg == '2b' else '--'
            ax.plot(sub['N_cal'], sub['KLD'], linestyle=linestyle,
                    color=plt.cm.tab20(p_idx / 12), label=f'{cfg}_{PAR_NAMES_12[p_idx]}' if p_idx == 0 else None)

    ax.set_xlabel('Calibration days (N)')
    ax.set_ylabel('KLD from uniform prior')
    ax.set_title('Figure 7.5: Parameter KLD comparison (2b solid, 2c dashed)')
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, 'fig7_5_KLD_comparison.png'), dpi=300)
    plt.close()
    print("Figure 7.5 saved.")

# %% [markdown]
# ## 3.7 Figure 7.6: Posteriors at N=50 and N=100 for 2b vs 2c

# %%
for N_show in [50, 100]:
    fig, axes = plt.subplots(3, 4, figsize=(16, 10))
    axes_flat = axes.flatten()

    for cfg in ['2b', '2c']:
        folder = os.path.join(OUTPUT_DIR, f"MZ2_ui_{YEAR}_{cfg}_N{N_show}days")
        ps_file = os.path.join(folder, f'ParSet_MZ2{YEAR}ui.npy')
        if not os.path.exists(ps_file):
            continue
        ParSet = np.load(ps_file)
        istart = int(0.5 * ParSet.shape[0])
        ParSet50 = ParSet[istart:, :-2]
        n_params = ParSet50.shape[1]
        names = PAR_NAMES_12 if n_params == 12 else PAR_NAMES_14

        df_soil_case = df_soil_all[df_soil_all['year'].astype(str) == YEAR].copy()
        ini_cfg, opkomst_cfg, crop_name_cfg, soil_type_cfg, g0_cfg, _, _, _ = \
            get_initial_params('MZ2', df_soil_case, df_crop, df_sensor_overview)
        sent_on = FULL_CONFIGS[cfg]['sentinel_cal_on']
        LB, UB = get_param_bounds(soil_type_cfg, ini_cfg, opkomst_cfg, crop_name_cfg,
                                   sentinel_cal_on=sent_on)

        alpha = 0.5 if cfg == '2b' else 0.5
        color = CONFIG_COLORS[cfg]
        for p_idx in range(min(12, n_params)):
            axes_flat[p_idx].hist(ParSet50[:, p_idx], bins=40, alpha=alpha,
                                  color=color, label=cfg, density=True)
            axes_flat[p_idx].set_xlim(LB[p_idx], UB[p_idx])
            if p_idx >= 8:
                axes_flat[p_idx].set_xlabel(names[p_idx])
            if p_idx % 4 == 0:
                axes_flat[p_idx].set_ylabel('Density')
            axes_flat[p_idx].set_title(names[p_idx], fontsize=10)

    fig.suptitle(f'Figure 7.6: Posteriors at N={N_show} days (2b=green, 2c=red)', fontsize=13)
    fig.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, f'fig7_6_posteriors_N{N_show}.png'), dpi=300)
    plt.close()
    print(f"Figure 7.6 (N={N_show}) saved.")

# %% [markdown]
# ## 3.8 Figure 7.7: Sample-based RMSD vs. calendar days

# %%
def compute_sample_metrics(case, config_name, N_cal, ParSetMax, df_list, g0_serial, cal_start_serial, year='2025'):
    """Compute RMSD against ground samples for calibration and validation periods."""
    from Sensordata import ConvertToSerialDate as CSD

    # Expand ParSetMax for Sentinel configs (remove a_S, b_S loglik cols)
    if len(ParSetMax) > 12:
        ParSetMax_swim = ParSetMax[:12]
    else:
        ParSetMax_swim = ParSetMax

    SWC, sw_list, g_list, sensor_data_full, covar_full, df_obs, _, _ = SWB(
        ParSetMax_swim[0], ParSetMax_swim[1], ParSetMax_swim[2], ParSetMax_swim[3], ParSetMax_swim[4],
        ParSetMax_swim[5], ParSetMax_swim[6], ParSetMax_swim[7], ParSetMax_swim[8], ParSetMax_swim[9],
        ParSetMax_swim[10], ParSetMax_swim[11],
        sensor=True, cal='gen', sensor_cal=np.empty(0), CI=np.empty(0),
        show=[False, ''], case=case, year=year, forecast=np.empty(0), df_list=df_list)

    g_list_arr = np.array(g_list)
    cal_end = cal_start_serial + N_cal - 1
    val_end = cal_start_serial + N_cal - 1 + 7

    cal_rmsd = np.nan
    val_rmsd = np.nan
    n_cal_samples = 0
    n_val_samples = 0

    obs_data = load_soilobs()
    obs_data = obs_data[obs_data['Sensornr'] == case].copy()
    obs_data = obs_data[obs_data['year'].astype(str) == year].copy()

    if len(obs_data) > 0 and 'Mean30' in obs_data.columns:
        cal_samples = obs_data[(obs_data['Date'] >= g0_serial) & (obs_data['Date'] <= cal_end)]
        val_samples = obs_data[(obs_data['Date'] > cal_end) & (obs_data['Date'] <= val_end)]

        for samples, label in [(cal_samples, 'cal'), (val_samples, 'val')]:
            if len(samples) > 0:
                sim_at_obs = []
                obs_at_obs = []
                for _, row in samples.iterrows():
                    idx = np.where(g_list_arr == row['Date'])[0]
                    if len(idx) > 0:
                        sim_at_obs.append(SWC[idx[0]])
                        obs_at_obs.append(row['Mean30'])
                if len(sim_at_obs) > 0:
                    rmsd = np.sqrt(np.mean((np.array(sim_at_obs) - np.array(obs_at_obs)) ** 2))
                    if label == 'cal':
                        cal_rmsd = rmsd
                        n_cal_samples = len(sim_at_obs)
                    else:
                        val_rmsd = rmsd
                        n_val_samples = len(sim_at_obs)

    return {'RMSD_samples_cal': cal_rmsd, 'RMSD_samples_val': val_rmsd,
            'n_cal_samples': n_cal_samples, 'n_val_samples': n_val_samples}

# Compute sample metrics for all configs
for key, result in results.items():
    if result is None:
        continue
    cfg_name = result['config']
    N_cal = result['N_cal']
    case_cfg = FULL_CONFIGS[cfg_name]['case']
    folder = os.path.join(OUTPUT_DIR, result['folder'])
    ps_file = os.path.join(folder, f'ParSet_MaxLL_{case_cfg}{YEAR}ui.npy')
    if not os.path.exists(ps_file):
        continue
    ParSetMax = np.load(ps_file)
    df_list_cfg = build_df_list(case_cfg)
    g0_serial_cfg = ConvertToSerialDate(PLANTING_DATE)
    cal_start_serial_cfg = ConvertToSerialDate(OBS_START_DATE)
    sample_metrics = compute_sample_metrics(case_cfg, cfg_name, N_cal, ParSetMax,
                                             df_list_cfg, g0_serial_cfg, cal_start_serial_cfg)
    result['metrics'].update(sample_metrics)

# Update master_df with sample metrics
master_df_updated = []
for key, result in results.items():
    if result is not None:
        row = {'config': result['config'], 'N_cal': result['N_cal'], **result['metrics']}
        master_df_updated.append(row)
if len(master_df_updated) > 0:
    master_df = pd.DataFrame(master_df_updated)
    master_df.to_csv(os.path.join(OUTPUT_DIR, 'master_metrics.csv'), index=False)

fig, ax = plt.subplots(figsize=(8, 5))
for cfg in ['Ref', '2a', '2b', '2c', '2d']:
    sub = master_df[master_df['config'] == cfg].sort_values('N_cal')
    if len(sub) == 0 or 'RMSD_samples_cal' not in sub.columns:
        continue
    ax.plot(sub['N_cal'], sub['RMSD_samples_cal'],
            marker=CONFIG_MARKERS[cfg], color=CONFIG_COLORS[cfg], linestyle='-',
            label=f'{cfg} cal', linewidth=1.5)
    if sub['RMSD_samples_val'].notna().any():
        ax.plot(sub['N_cal'], sub['RMSD_samples_val'],
                marker=CONFIG_MARKERS[cfg], color=CONFIG_COLORS[cfg], linestyle='--',
                label=f'{cfg} val', linewidth=1.0, alpha=0.7)

ax.set_xlabel('Calibration days (N)')
ax.set_ylabel('RMSD vs samples (m3/m3)')
ax.set_title('Figure 7.7: Sample-based RMSD')
ax.legend(fontsize=8)
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, 'fig7_7_RMSD_samples.png'), dpi=300)
plt.close()
print("Figure 7.7 saved.")

# %% [markdown]
# ## 3.9 Table 7.1: Days to stable prediction

# %%
threshold = 0.04
stable_days = {}
for cfg in ['Ref', '2a', '2b', '2c', '2d']:
    sub = master_df[master_df['config'] == cfg].sort_values('N_cal')
    stable = sub[sub['bcRMSD_val_sensor'] < threshold]
    if len(stable) > 0:
        stable_days[cfg] = stable['N_cal'].iloc[0]
    else:
        stable_days[cfg] = None

print("\n=== Table 7.1: Days to stable prediction (bcRMSD_val < 0.04) ===")
for cfg, days in stable_days.items():
    print(f"  {cfg}: {days if days is not None else 'Not reached'}")

# %% [markdown]
# ## 3.10 Table 4.1: Error statistics comparison

# %%
sentinel_err_path = os.path.join(BASE_DIR, '..', 'output', 'sentinel_error_params.json')
if os.path.exists(sentinel_err_path):
    with open(sentinel_err_path, 'r') as f:
        sent_err = json.load(f)

    print("\n=== Table 4.1: Error statistics comparison ===")
    print(f"{'Property':<30} {'In-situ sensor (Ch4)':<25} {'Sentinel-1 (this work)':<25}")
    print("-" * 80)
    print(f"{'Calibration (a, b)':<30} {'-0.006, 1.260 (fixed)':<25} {'a_S, b_S (DREAM)':<25}")
    print(f"{'MZ1 Deming (a, b)':<30} {'-':<25} {'0.054, 0.487':<25}")
    print(f"{'MZ2 Deming (a, b)':<30} {'-':<25} {'0.141, 0.296':<25}")
    print(f"{'sigma2_alpha':<30} {'0.001070':<25} {sent_err.get('sigma2_alpha', 'N/A'):<25}")
    print(f"{'sigma2_epsilon':<30} {'0.000998':<25} {sent_err.get('sigma2_epsilon', 'N/A'):<25}")
    print(f"{'ACOR':<30} {'0.518':<25} {sent_err.get('ACOR', 'N/A'):<25}")
    print(f"{'Demying slope (MZ1)':<30} {'~1.26':<25} {sent_err.get('calibration_verification', {}).get('deming_slope', 'N/A'):<25}")
    print(f"{'Temporal resolution':<30} {'daily':<25} {'2-6 days':<25}")
    print(f"{'Data window':<30} {'full season':<25} {'Apr 24 - Jun 5':<25}")

# %% [markdown]
# ## 3.11 Figures 4.1-4.3: Sentinel error characterization

# %%
if os.path.exists(sentinel_err_path):
    # Load theta timeseries for visualization
    theta_path = os.path.join(BASE_DIR, '..', 'output', 'theta_timeseries.csv')
    if os.path.exists(theta_path):
        theta_df = pd.read_csv(theta_path)

        # Figure 4.1: Sentinel SWI time series vs sensor VWC
        fig, ax = plt.subplots(figsize=(10, 5))
        for area_key, label in [('MZ1', 'MZ1'), ('MZ2', 'MZ2')]:
            sub = theta_df[theta_df['area'] == area_key]
            ax.plot(pd.to_datetime(sub['date_str']), sub['Theta'], marker='o', label=f'{label} Sentinel Theta', markersize=4)
        ax.set_xlabel('Date')
        ax.set_ylabel('Theta / SWI')
        ax.set_title('Figure 4.1: Sentinel-1 SWI time series')
        ax.legend()
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(os.path.join(OUTPUT_DIR, 'fig4_1_SWI_timeseries.png'), dpi=300)
        plt.close()
        print("Figure 4.1 saved.")

        # Figure 4.2: DpRVIc vs Theta scatter
        fig, ax = plt.subplots(figsize=(7, 6))
        for area_key, label, color in [('MZ1', 'MZ1', '#1f77b4'),
                                        ('MZ2', 'MZ2', '#ff7f0e')]:
            sub = theta_df[theta_df['area'] == area_key]
            ax.scatter(sub['DpRVIc'], sub['Theta'], alpha=0.7, label=label, color=color, s=30)
        ax.set_xlabel('DpRVIc')
        ax.set_ylabel('Theta')
        ax.set_title('Figure 4.2: DpRVIc vs Theta')
        ax.legend()
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(os.path.join(OUTPUT_DIR, 'fig4_2_DpRVIc_vs_Theta.png'), dpi=300)
        plt.close()
        print("Figure 4.2 saved.")

# Correlation / lag analysis figure from error params
if os.path.exists(sentinel_err_path):
    lag_corrs = sent_err.get('lag_correlations', {})
    if lag_corrs:
        fig, ax = plt.subplots(figsize=(7, 5))
        lags = [int(k) for k in lag_corrs.keys()]
        rhos = [float(v) for v in lag_corrs.values()]
        ax.bar(lags, rhos, color='steelblue', alpha=0.8)
        ax.set_xlabel('Lag (days)')
        ax.set_ylabel('Autocorrelation')
        ax.set_title(f'Figure 4.3: Sentinel error lag correlations (ACOR={sent_err.get("ACOR", "N/A"):.3f})')
        ax.axhline(y=0, color='k', linewidth=0.5)
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(os.path.join(OUTPUT_DIR, 'fig4_3_lag_correlations.png'), dpi=300)
        plt.close()
        print("Figure 4.3 saved.")

print("\nAll figures and tables generated.")

# %% [markdown]
# # SWC & Soil Water Balance figures (generated via SWB model)

# %%
import csv

def generate_swb_figures(N_cal_list=None):
    """Generate SWC and Soil Water Balance figures for each config/N using SWB model.
    
    Loads pre-computed CI.npy from disk (computed during DREAM post-processing).
    No ensemble runs — purely post-processing/plotting.
    """
    if N_cal_list is None:
        N_cal_list = CAL_DAYS_LIST
    
    for cfg in ['Ref', '2a', '2b', '2c', '2d']:
        if cfg not in FULL_CONFIGS:
            continue
        case_cfg = FULL_CONFIGS[cfg]['case']
        df_list_cfg = build_df_list(case_cfg)
        df_soil_case = df_soil_all[df_soil_all['year'].astype(str) == YEAR].copy()
        ini_cfg, opkomst_cfg, crop_name_cfg, soil_type_cfg, g0_cfg, forecast_cfg, irr_method_cfg, part_cfg = \
            get_initial_params(case_cfg, df_soil_case, df_crop, df_sensor_overview)
        
        for N_cal in N_cal_list:
            folder_name = f'{case_cfg}_ui_{YEAR}_{cfg}_N{N_cal}days'
            folder_path = os.path.join(OUTPUT_DIR, folder_name)
            ps_file = os.path.join(folder_path, f'ParSet_MaxLL_{case_cfg}{YEAR}{crop_name_cfg}.npy')
            ci_file = os.path.join(folder_path, 'CI.npy')
            if not os.path.exists(ps_file):
                continue
            
            print(f'  Generating SWB figures for {cfg} N={N_cal}...')
            ParSetMax = np.load(ps_file)
            ParSetMax_swim = ParSetMax[:-2]
            
            CI = np.load(ci_file, allow_pickle=True) if os.path.exists(ci_file) else np.empty(0)
            
            SWC_max, sw_max, g_max, sensor_max, covar_max, df_obs_max, df_all_max, df_parameters_max = SWB(
                ParSetMax_swim[0], ParSetMax_swim[1], ParSetMax_swim[2], ParSetMax_swim[3],
                ParSetMax_swim[4], ParSetMax_swim[5], ParSetMax_swim[6], ParSetMax_swim[7],
                ParSetMax_swim[8], ParSetMax_swim[9], ParSetMax_swim[10], ParSetMax_swim[11],
                sensor=True, cal='gen', sensor_cal=np.empty(0), CI=CI,
                show=[True, folder_path], case=case_cfg, year=YEAR,
                forecast=np.empty(0), df_list=df_list_cfg)
            
            os.makedirs(folder_path, exist_ok=True)
            pd.DataFrame(SWC_max).to_csv(os.path.join(folder_path, f'SWC_{case_cfg}{YEAR}{crop_name_cfg}.csv'), index=False)
            pd.DataFrame(g_max).to_csv(os.path.join(folder_path, f'g_list_{case_cfg}{YEAR}{crop_name_cfg}.csv'), index=False)
            df_all_max.to_csv(os.path.join(folder_path, f'df_all_{case_cfg}{YEAR}{crop_name_cfg}.csv'), index=False)
            
            par_names_swim = ['Kcb_ini','Kcb_mid','Kcb_end','L_ini','L_dev','L_mid',
                             'fc','log(Ksat)','CN','GWT_max','Zr_max','v_ini']
            if ParSetMax_swim.shape[0] > 12:
                par_names_swim = par_names_swim + ['a_S','b_S']
            parset_file = os.path.join(folder_path, f'ParSet_{case_cfg}{YEAR}{crop_name_cfg}.npy')
            if os.path.exists(parset_file):
                ParSet_full = np.load(parset_file)
                istart2 = int(0.5 * ParSet_full.shape[0])
                ParSet50b = ParSet_full[istart2:, :-2]
                excl2 = max(1, np.round(ParSet50b.shape[0]*0.025).astype(int))
                sort = ParSet50b.argsort(axis=0)
                ParSet50_sort = ParSet50b[sort, np.arange(sort.shape[1])]
                ParSet_95 = ParSet50_sort[excl2:-excl2]
                par_names_95 = par_names_swim
                ParSetMax_95_values = ParSetMax_swim
                with open(os.path.join(folder_path, f'ParSetMax-95_{case_cfg}{YEAR}{crop_name_cfg}.csv'), 'w', newline='') as f:
                    w = csv.writer(f, delimiter=',')
                    w.writerow(['Parameter','Max LLH','Min 95%','Max 95%'])
                    for i in range(min(len(par_names_95), ParSet_95.shape[1])):
                        w.writerow([par_names_95[i], ParSetMax_95_values[i],
                                   np.min(ParSet_95[:, i]), np.max(ParSet_95[:, i])])
            
            print(f'    -> Saved to {folder_path}')
    
    print('SWB figures generated.')


if args.compute_ci:
    for cfg in ['Ref', '2a', '2b', '2c', '2d']:
        if cfg not in FULL_CONFIGS:
            continue
        case_cfg = FULL_CONFIGS[cfg]['case']
        df_soil_case = df_soil_all[df_soil_all['year'].astype(str) == YEAR].copy()
        _, _, crop_name_ci, _, _, _, _, _ = get_initial_params(case_cfg, df_soil_case, df_crop, df_sensor_overview)
        df_list_ci = build_df_list(case_cfg)
        for N_cal in CAL_DAYS_LIST:
            folder_name = f'{case_cfg}_ui_{YEAR}_{cfg}_N{N_cal}days'
            folder_path = os.path.join(OUTPUT_DIR, folder_name)
            ci_file = os.path.join(folder_path, 'CI.npy')
            if os.path.exists(ci_file):
                print(f'  CI already exists for {cfg} N={N_cal}, skipping')
                continue
            if not os.path.exists(os.path.join(folder_path, f'ParSet_{case_cfg}{YEAR}{crop_name_ci}.npy')):
                continue
            print(f'  Computing CI for {cfg} N={N_cal}...')
            compute_and_save_ci(case_cfg, cfg, N_cal, None, folder_path,
                                crop_name_ci, case_cfg=case_cfg, df_list_cfg=df_list_ci)
    print('CI computation done.')
elif args.recompute_metrics:
    for cfg in ['Ref', '2a', '2b', '2c', '2d']:
        if cfg not in FULL_CONFIGS:
            continue
        case_cfg = FULL_CONFIGS[cfg]['case']
        df_soil_case = df_soil_all[df_soil_all['year'].astype(str) == YEAR].copy()
        _, _, crop_name_cfg, _, _, _, _, _ = get_initial_params(case_cfg, df_soil_case, df_crop, df_sensor_overview)
        df_list_cfg = build_df_list(case_cfg)
        g0_serial_cfg = ConvertToSerialDate(PLANTING_DATE)
        cal_start_serial_cfg = ConvertToSerialDate(OBS_START_DATE)
        for N_cal in CAL_DAYS_LIST:
            folder_name = f'{case_cfg}_ui_{YEAR}_{cfg}_N{N_cal}days'
            folder_path = os.path.join(OUTPUT_DIR, folder_name)
            ps_file = os.path.join(folder_path, f'ParSet_MaxLL_{case_cfg}{YEAR}{crop_name_cfg}.npy')
            if not os.path.exists(ps_file):
                continue
            print(f'  Recomputing metrics for {cfg} N={N_cal}...')
            ParSetMax = np.load(ps_file)
            metrics = compute_validation_metrics(case_cfg, cfg, N_cal, ParSetMax, df_list_cfg,
                                                  g0_serial_cfg, cal_start_serial_cfg, year=YEAR)
            metrics_df = pd.DataFrame(metrics, index=[0])
            metrics_df.to_csv(os.path.join(folder_path, f'metrics_{case_cfg}_{YEAR}_{cfg}_N{N_cal}days.csv'), index=False)
            print(f'    bcRMSD_val={metrics.get("bcRMSD_val_sensor", float("nan")):.4f} | '
                  f'OR={metrics.get("OR_sensor", float("nan")):.3f}')
    print('Metrics recomputation done.')
else:
    generate_swb_figures()
