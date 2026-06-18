"""
Parallel DREAM-ZS runner.

Orchestrates multiple (config, N_cal) runs across CPU cores using
subprocess calls to run_pipeline.py. Transfer scenarios (2c, 2d)
wait for their REF dependency to complete before starting.

Usage:
    python run_parallel.py                       # sequential (default)
    python run_parallel.py --workers 8           # 8 parallel workers
    python run_parallel.py --workers 8 --test    # quick test mode
    python run_parallel.py --workers 8 --run-id prod
"""

import argparse
import os
import subprocess
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed

BASE_DIR = os.path.dirname(os.path.realpath(__file__))
PIPELINE_SCRIPT = os.path.join(BASE_DIR, 'run_pipeline.py')
PYTHON = sys.executable

CONFIGS = ['Ref', '2a', '2b', '2c', '2d']

DEPENDENCIES = {
    'Ref': None,
    '2a': None,
    '2b': None,
    '2c': 'Ref',
    '2d': 'Ref',
}

CAL_DAYS_LIST = [10, 20]

SEPARATOR = '=' * 70


def run_one(config, N_cal, test, run_id, log_dir=None):
    """Run a single (config, N_cal) via subprocess.

    Streams stdout/stderr live to terminal. Optionally writes to a log file.

    Returns
    -------
    tuple : (config, N_cal, returncode, elapsed_seconds)
    """
    cmd = [PYTHON, PIPELINE_SCRIPT,
           '--configs', config, '--N-values', str(N_cal)]
    if test:
        cmd.append('--test')
    if run_id:
        cmd.extend(['--run-id', run_id])

    log_file = None
    if log_dir:
        os.makedirs(log_dir, exist_ok=True)
        log_path = os.path.join(log_dir, f'{config}_N{N_cal}.log')
        log_file = open(log_path, 'w')

    start = time.time()
    proc = subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        cwd=BASE_DIR,
    )
    for line in proc.stdout:
        print(line, end='')
        if log_file:
            log_file.write(line)

    proc.wait()
    elapsed = time.time() - start

    if log_file:
        log_file.close()

    return config, N_cal, proc.returncode, elapsed


def check_dependency(config, N_cal, completed, run_id=None):
    """Check if a run's dependency has completed successfully.

    For transfer scenarios (2c, 2d), checks that the REF output
    ParSet file exists on disk.
    """
    dep = DEPENDENCIES.get(config)
    if dep is None:
        return True
    year = '2025'
    search_dirs = []
    if run_id:
        search_dirs.append(os.path.join(
            BASE_DIR, '..', 'output', 'dream_results', run_id))
    search_dirs.append(os.path.join(
        BASE_DIR, '..', 'output', 'dream_results'))

    for base in search_dirs:
        ref_folder = os.path.join(base, f'MZ1_ui_{year}_Ref_N{N_cal}days')
        ref_ps = os.path.join(ref_folder, f'ParSet_MZ1{year}ui.npy')
        if os.path.exists(ref_ps):
            return True

    if (dep, N_cal) in completed and completed[(dep, N_cal)] == 0:
        print(f'  WARNING: {config} N={N_cal} dependency {dep} reported OK '
              f'but ParSet file not found')
    return False


def main():
    parser = argparse.ArgumentParser(
        description='Run DREAM-ZS scenarios in parallel or sequential mode.')
    parser.add_argument('--workers', type=int, default=4,
                        help='Number of parallel workers (1=sequential, default: 4). '
                             'Each worker uses ~400-500 MB RAM. With 16 GB total, '
                             '4 workers is safe; 6 may work; 12+ risks crashing.')
    parser.add_argument('--test', action='store_true',
                        help='Quick test mode: T=50, seq=3')
    parser.add_argument('--run-id', type=str, default=None,
                        help='Run ID subfolder for output (passed to run_pipeline.py)')
    parser.add_argument('--log-dir', type=str, default=None,
                        help='Directory for per-run log files (default: no logs)')
    parser.add_argument('--configs', type=str, default=None,
                        help='Comma-separated config names to run (e.g. Ref,2a,2b,2c,2d)')
    args = parser.parse_args()

    if args.configs:
        selected = [x.strip() for x in args.configs.split(',')]
        invalid = [c for c in selected if c not in CONFIGS]
        if invalid:
            print(f'ERROR: Unknown configs: {invalid}')
            print(f'Available: {CONFIGS}')
            return 1
        active_configs = [c for c in CONFIGS if c in selected]
    else:
        active_configs = CONFIGS

    all_runs = [(c, n) for c in active_configs for n in CAL_DAYS_LIST]
    total = len(all_runs)

    print(SEPARATOR)
    print('DREAM-ZS Parallel Runner')
    print(f'  Workers:  {args.workers}')
    print(f'  Mode:     {"TEST" if args.test else "PRODUCTION"}')
    print(f'  Configs:  {active_configs}')
    print(f'  N values: {CAL_DAYS_LIST}')
    print(f'  Total runs: {total}')
    print(SEPARATOR)

    results = {}
    completed = {}

    if args.workers == 1:
        print('\nRunning sequentially (1 worker)...\n')
        log_dir = args.log_dir

        for i, (config, N_cal) in enumerate(all_runs, 1):
            if not check_dependency(config, N_cal, completed, run_id=args.run_id):
                dep = DEPENDENCIES[config]
                print(f'  [{i}/{total}] {config} N={N_cal}: SKIPPED '
                      f'(dependency {dep} N={N_cal} not available)')
                results[(config, N_cal)] = ('SKIPPED', 0)
                continue

            print(f'\n  [{i}/{total}] {config} N={N_cal}: starting...')
            print('-' * 50)
            start = time.time()
            try:
                c, n, rc, elapsed = run_one(
                    config, N_cal, args.test, args.run_id, log_dir)
            except Exception as e:
                rc, elapsed = -1, time.time() - start
                print(f'  ERROR: {e}')

            status = 'OK' if rc == 0 else 'FAILED'
            print(f'\n  [{i}/{total}] {config} N={N_cal}: {status} '
                  f'({elapsed / 60:.1f} min)')
            print('-' * 50)

            completed[(config, N_cal)] = rc
            results[(config, N_cal)] = (status, elapsed)

    else:
        print(f'\nRunning with {args.workers} parallel workers...\n')

        phase1 = [(c, n) for c, n in all_runs if DEPENDENCIES[c] is None]
        phase2 = [(c, n) for c, n in all_runs if DEPENDENCIES[c] is not None]

        print(f'  Phase 1: {len(phase1)} independent runs (Ref, 2a, 2b)')
        print(f'  Phase 2: {len(phase2)} dependent runs (2c, 2d)')
        print()

        log_dir = args.log_dir

        with ProcessPoolExecutor(max_workers=args.workers) as pool:
            # Phase 1: independent runs
            print('--- Phase 1: Independent runs ---')
            futures = {}
            for config, N_cal in phase1:
                f = pool.submit(run_one, config, N_cal, args.test,
                                args.run_id, log_dir)
                futures[f] = (config, N_cal)

            for f in as_completed(futures):
                c, n, rc, elapsed = f.result()
                completed[(c, n)] = rc
                status = 'OK' if rc == 0 else 'FAILED'
                print(f'  {c} N={n}: {status} ({elapsed / 60:.1f} min)')
                results[(c, n)] = (status, elapsed)

            # Phase 2: dependent runs
            print('\n--- Phase 2: Dependent runs ---')
            phase2_submit = []
            phase2_skip = []
            for config, N_cal in phase2:
                if check_dependency(config, N_cal, completed, run_id=args.run_id):
                    phase2_submit.append((config, N_cal))
                else:
                    phase2_skip.append((config, N_cal))

            for config, N_cal in phase2_skip:
                dep = DEPENDENCIES[config]
                print(f'  {config} N={N_cal}: SKIPPED '
                      f'(dependency {dep} N={N_cal} output not found)')
                results[(config, N_cal)] = ('SKIPPED', 0)

            futures2 = {}
            for config, N_cal in phase2_submit:
                f = pool.submit(run_one, config, N_cal, args.test,
                                args.run_id, log_dir)
                futures2[f] = (config, N_cal)

            for f in as_completed(futures2):
                c, n, rc, elapsed = f.result()
                completed[(c, n)] = rc
                status = 'OK' if rc == 0 else 'FAILED'
                print(f'  {c} N={n}: {status} ({elapsed / 60:.1f} min)')
                results[(c, n)] = (status, elapsed)

    # Summary
    print(f'\n{SEPARATOR}')
    print('Summary')
    print(SEPARATOR)
    n_ok = sum(1 for s, _ in results.values() if s == 'OK')
    n_fail = sum(1 for s, _ in results.values() if s == 'FAILED')
    n_skip = sum(1 for s, _ in results.values() if s == 'SKIPPED')
    total_wall = sum(t for _, t in results.values() if t > 0)

    print(f'  OK:      {n_ok}')
    print(f'  FAILED:  {n_fail}')
    print(f'  SKIPPED: {n_skip}')
    if args.workers == 1:
        print(f'  Total wall time: {total_wall / 60:.1f} min')
    else:
        print(f'  Sum of run times: {total_wall / 60:.1f} min '
              f'(wall time is less with {args.workers} workers)')
    print()

    header = f'{"Config":<8} {"N_cal":<6} {"Status":<10} {"Time (min)":<10}'
    print(header)
    print('-' * len(header))
    for config in active_configs:
        for N_cal in CAL_DAYS_LIST:
            s, t = results.get((config, N_cal), ('?', 0))
            print(f'{config:<8} {N_cal:<6} {s:<10} {t / 60:.1f}')
    print()

    if n_fail > 0:
        print('Failed runs:')
        for (c, n), (s, t) in sorted(results.items()):
            if s == 'FAILED':
                print(f'  {c} N={n}')
    if n_skip > 0:
        print('Skipped runs (dependency not available):')
        for (c, n), (s, t) in sorted(results.items()):
            if s == 'SKIPPED':
                dep = DEPENDENCIES[c]
                print(f'  {c} N={n} (needs {dep} N={n})')

    print(SEPARATOR)
    if n_fail > 0 or n_skip > 0:
        print('To re-run specific scenarios:')
        print(f'  python {PIPELINE_SCRIPT} --configs <config> --N-values <N>')
    else:
        print('All runs completed successfully.')

    return 0 if n_fail == 0 else 1


if __name__ == '__main__':
    sys.exit(main())