import argparse
import json
import os
import numpy as np
import pandas as pd
from datetime import datetime
from collections import OrderedDict

PAR_NAMES_CROP = ['Kcb_ini', 'Kcb_mid', 'Kcb_end', 'L_ini', 'L_dev', 'L_mid']
PAR_NAMES_SOIL = ['fc', 'log(Ksat)', 'CN', 'GWT_max', 'Zr_max', 'v_ini']
PAR_NAMES_SENTINEL = ['a_S', 'b_S']

FULL_CONFIGS = OrderedDict({
    'Ref': {
        'case': 'MZ1', 'obsdata': 'Sensor+stalen',
        'Prior': 'LHS', 'sentinel_cal_on': False, 'n_params': 12,
        'description': 'MZ1 sensor+samples, 12 params, LHS prior',
    },
    '2a': {
        'case': 'MZ2', 'obsdata': 'Samples only',
        'Prior': 'LHS', 'sentinel_cal_on': False, 'n_params': 12,
        'description': 'MZ2 samples only, 12 params, LHS prior',
    },
    '2b': {
        'case': 'MZ2', 'obsdata': 'Sentinel+stalen',
        'Prior': 'LHS', 'sentinel_cal_on': True, 'n_params': 14,
        'description': 'MZ2 Sentinel+samples, 14 params (12 SWB + a_S + b_S), LHS prior',
    },
    '2c': {
        'case': 'MZ2', 'obsdata': 'Sentinel+stalen',
        'Prior': 'Transfer', 'sentinel_cal_on': True, 'n_params': 14,
        'description': 'MZ2 Sentinel+samples, 14 params, Transfer prior from MZ1',
    },
    '2d': {
        'case': 'MZ2', 'obsdata': 'Samples only',
        'Prior': 'Transfer', 'sentinel_cal_on': False, 'n_params': 12,
        'description': 'MZ2 samples only, 12 params, Transfer prior',
    },
})

SENSOR_ALPHA2 = 0.001070
SENSOR_EPSILON2 = 0.000998

CROP = 'ui'
YEAR = '2025'


def get_param_names(config_name, n_params):
    if n_params == 14:
        return PAR_NAMES_CROP + PAR_NAMES_SOIL + PAR_NAMES_SENTINEL
    else:
        return PAR_NAMES_CROP + PAR_NAMES_SOIL


def detect_configs_and_N(output_dir):
    folders = [d for d in os.listdir(output_dir)
               if os.path.isdir(os.path.join(output_dir, d))]
    results = []
    for folder in folders:
        config_name = None
        N_cal = None
        for cfg in FULL_CONFIGS:
            if f'_{cfg}_' in folder or folder.endswith(f'_{cfg}'):
                config_name = cfg
                break
        if config_name is None:
            continue
        parts = folder.split('_')
        for part in parts:
            if part.startswith('N') and part.endswith('days'):
                try:
                    N_cal = int(part[1:-4])
                except ValueError:
                    pass
        if config_name and N_cal:
            results.append((config_name, N_cal, folder))
    return results


def summarize_run(run_id):
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    OUTPUT_DIR = os.path.join(BASE_DIR, '..', 'output', 'dream_results')
    if run_id:
        output_dir = os.path.join(OUTPUT_DIR, run_id)
    else:
        output_dir = OUTPUT_DIR

    deming_file = os.path.join(BASE_DIR, '..', 'output', 'deming_priors_by_N.json')
    deming_priors = {}
    if os.path.exists(deming_file):
        with open(deming_file, 'r') as f:
            deming_priors = json.load(f)

    detected = detect_configs_and_N(output_dir)
    if not detected:
        print(f"No config folders found in {output_dir}")
        return

    summary = {
        'meta': {
            'run_id': run_id or 'default',
            'timestamp': datetime.now().isoformat(),
            'pipeline': 'SWIM2-DREAM-ZS v2.0 (N-cal dependent covariance)',
            'configs_run': sorted(set(c for c, n, f in detected)),
            'N_cal_values': sorted(set(n for c, n, f in detected)),
            'deming_priors_by_N': deming_priors,
            'sensor_covariance': {
                'alpha2': SENSOR_ALPHA2,
                'epsilon2': SENSOR_EPSILON2,
                'source': 'hardcoded (Chapter 4 in-situ sensor characterization)',
            },
        },
        'configs': {},
    }

    for config_name, N_cal, folder in sorted(detected, key=lambda x: (x[0], x[1])):
        cfg = FULL_CONFIGS[config_name]
        folder_path = os.path.join(output_dir, folder)
        case = cfg['case']

        if config_name not in summary['configs']:
            summary['configs'][config_name] = {
                'description': cfg['description'],
                'n_params': cfg['n_params'],
                'param_names': get_param_names(config_name, cfg['n_params']),
                'case': case,
                'obsdata': cfg['obsdata'],
                'prior': cfg['Prior'],
                'sentinel_cal_on': cfg['sentinel_cal_on'],
            }

        N_key = f'N{N_cal}'
        n_result = {}

        parset_files = [f for f in os.listdir(folder_path)
                        if f.startswith('ParSet_') and 'MaxLL' not in f]
        maxll_files = [f for f in os.listdir(folder_path)
                       if f.startswith('ParSet_MaxLL')]

        measurements_file = os.path.join(folder_path, 'Measurements_used_DREAM.csv')
        if os.path.exists(measurements_file):
            meas = pd.read_csv(measurements_file)
            n_result['n_observations'] = {
                'total': int(meas['N_total'].iloc[0]) if 'N_total' in meas.columns else None,
                'continuous': int(meas['N_sensor'].iloc[0]) if 'N_sensor' in meas.columns else None,
                'samples': int(meas['N_samples'].iloc[0]) if 'N_samples' in meas.columns else None,
            }

        if not parset_files:
            summary['configs'][config_name][N_key] = n_result
            continue

        parset_path = os.path.join(folder_path, parset_files[0])
        ParSet = np.load(parset_path, allow_pickle=True)
        istart = int(0.5 * ParSet.shape[0])
        ParSet50 = ParSet[istart:, :-2]
        n_params = ParSet50.shape[1]
        param_names = get_param_names(config_name, n_params)

        ParSetMax = np.load(os.path.join(folder_path, maxll_files[0]), allow_pickle=True) if maxll_files else ParSet[np.argmax(ParSet[:, -1]), :-2]

        p2_5 = np.percentile(ParSet50, 2.5, axis=0)
        p97_5 = np.percentile(ParSet50, 97.5, axis=0)
        pmean = np.mean(ParSet50, axis=0)
        pmedian = np.median(ParSet50, axis=0)
        psd = np.std(ParSet50, axis=0)

        p95_file = [f for f in os.listdir(folder_path) if f.startswith('ParSetMax-95')]

        parset_max_dict = {}
        posterior_summary = {}

        for i, pname in enumerate(param_names):
            if i < len(ParSetMax):
                parset_max_dict[pname] = float(ParSetMax[i])

            bound_info = {
                'mean': float(pmean[i]),
                'median': float(pmedian[i]),
                'sd': float(psd[i]),
                'p2.5': float(p2_5[i]),
                'p97.5': float(p97_5[i]),
            }

            posterior_summary[pname] = bound_info

        n_result['parset_max'] = parset_max_dict
        n_result['posterior_summary'] = posterior_summary
        n_result['parset_shape'] = list(ParSet.shape)

        if cfg['sentinel_cal_on']:
            N_str = str(N_cal)
            if N_str in deming_priors:
                dp = deming_priors[N_str]
                n_result['covariance_used'] = {
                    'type': 'sentinel_n_cal',
                    'alpha2': dp.get('alpha2'),
                    'epsilon2': dp.get('epsilon2'),
                    'n_matched_obs': dp.get('n_obs'),
                }
                n_result['deming_priors_used'] = {
                    'a': dp.get('a'), 'b': dp.get('b'),
                    'SE_a': dp.get('SE_a'), 'SE_b': dp.get('SE_b'),
                    'alpha2': dp.get('alpha2'), 'epsilon2': dp.get('epsilon2'),
                }
            else:
                n_result['covariance_used'] = {'type': 'sentinel_n_cal', 'note': f'N={N_cal} not in deming_priors'}
                n_result['deming_priors_used'] = None
        else:
            n_result['covariance_used'] = {
                'type': 'sensor',
                'alpha2': SENSOR_ALPHA2,
                'epsilon2': SENSOR_EPSILON2,
            }
            n_result['deming_priors_used'] = None

        metrics_file = os.path.join(folder_path, f'metrics_{case}_{YEAR}_{config_name}_N{N_cal}days.csv')
        if os.path.exists(metrics_file):
            m = pd.read_csv(metrics_file)
            n_result['metrics'] = {col: float(m[col].iloc[0]) for col in m.columns}
        else:
            n_result['metrics'] = None

        ci_file = os.path.join(folder_path, 'CI.npy')
        if os.path.exists(ci_file):
            ci = np.load(ci_file, allow_pickle=True)
            n_result['ci_available'] = bool(ci.size > 0 and ci.shape[0] > 0)
        else:
            n_result['ci_available'] = False

        covar_file = os.path.join(folder_path, 'Covar_DREAM.csv')
        if os.path.exists(covar_file):
            cov = pd.read_csv(covar_file).values
            n_result['covariance_shape'] = list(cov.shape)
            n_result['covariance_diag_mean'] = float(np.mean(np.diag(cov)))
            if cov.shape[0] > 1:
                n_result['covariance_offdiag_mean'] = float(
                    np.mean(cov[np.triu_indices(cov.shape[0], k=1)]))

        n_result['files'] = {
            'parset': os.path.basename(parset_files[0]),
            'parset_maxll': os.path.basename(maxll_files[0]) if maxll_files else None,
        }

        summary['configs'][config_name][N_key] = n_result

    metrics_to_rank = ['bcRMSD_val_sensor', 'bcNSE_val_sensor', 'OR_sensor',
                       'MAE_val_sensor', 'RMSE_val_sensor']
    rankings = {}
    for metric in metrics_to_rank:
        entries = []
        for cfg in summary['configs']:
            for N_key, N_data in summary['configs'][cfg].items():
                if not N_key.startswith('N'):
                    continue
                if N_data.get('metrics') and metric in N_data['metrics']:
                    val = N_data['metrics'][metric]
                    if val is not None and not (isinstance(val, float) and np.isnan(val)):
                        entries.append({'config': cfg, 'N': int(N_key[1:]), 'value': val})
        if entries:
            if 'RMSD' in metric or 'MAE' in metric or 'OR' in metric or 'bcRMSD' in metric:
                entries.sort(key=lambda x: x['value'])
            else:
                entries.sort(key=lambda x: -x['value'])
            rankings[metric] = entries
    summary['rankings'] = rankings

    diagnostics = {'warnings': []}
    for cfg in summary['configs']:
        cfg_data = summary['configs'][cfg]
        for N_key, N_data in cfg_data.items():
            if not N_key.startswith('N'):
                continue
            key = f'{cfg}_{N_key}'
            if N_data.get('covariance_used', {}).get('type') == 'sentinel_n_cal':
                alpha2 = N_data['covariance_used'].get('alpha2', 0)
                if alpha2 is not None and alpha2 < 1e-8:
                    diagnostics['warnings'].append(
                        f'{key}: alpha2={alpha2:.2e} (diagonal covariance only - no temporal correlation)')

    summary['diagnostics'] = diagnostics

    out_path = os.path.join(output_dir, 'run_summary.json')
    with open(out_path, 'w') as f:
        json.dump(summary, f, indent=2, default=str)
    print(f"Summary written to {out_path}")
    print(f"Configs: {summary['meta']['configs_run']}")
    print(f"N values: {summary['meta']['N_cal_values']}")
    for metric, entries in rankings.items():
        if entries:
            best = entries[0]
            print(f"  Best {metric}: {best['config']} N={best['N']} ({best['value']:.4f})")
    if diagnostics['warnings']:
        for w in diagnostics['warnings']:
            print(f"  WARNING: {w}")

    return summary


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Summarize DREAM production results as agent-readable JSON')
    parser.add_argument('--run-id', type=str, default=None,
                        help='Run ID subfolder (e.g. prod_newmatrix). Default: root output dir.')
    args = parser.parse_args()
    summarize_run(args.run_id)