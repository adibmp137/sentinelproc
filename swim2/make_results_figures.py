"""
Generate the Results-chapter figures for the thesis from the final_v4 DREAM run.
Outputs are written directly into paper/figures/ with res_* names.

Figures:
  res_scenario_metrics.png   - bcRMSD_val and bcNSE_val vs N_cal, all 5 scenarios (no SWB)
  res_posteriors_transfer.png- posterior histograms 2b (uniform) vs 2c (transfer), key params
  res_ref_calval.png         - REF N=30 SWC time series + 95% CI + cal/val windows
  res_mz2_scenarios.png      - MZ2 scenarios (2a,2b,2c,2d) N=30 SWC + CI vs sensor/samples
"""
import os, sys, json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

matplotlib.rcParams.update({
    'font.family': 'serif',
    'mathtext.fontset': 'dejavuserif',
    'font.size': 16,
    'axes.titlesize': 17,
    'axes.labelsize': 16,
    'xtick.labelsize': 15,
    'ytick.labelsize': 15,
    'legend.fontsize': 13,
    'lines.linewidth': 2.0,
})

BASE = os.path.dirname(os.path.abspath(__file__))
os.chdir(BASE)
sys.path.insert(0, BASE)
RUN = os.path.join('..', 'output', 'dream_results', 'final_v4')
FIGDIR = os.path.join('..', 'paper', 'figures')
os.makedirs(FIGDIR, exist_ok=True)
_DFP = None
_DFI = None

YEAR = '2025'
COLORS = {'Ref': '#1f77b4', '2a': '#ff7f0e', '2b': '#2ca02c', '2c': '#d62728', '2d': '#9467bd'}
MARK = {'Ref': 'o', '2a': 's', '2b': '^', '2c': 'D', '2d': 'v'}
LABEL = {'Ref': 'REF (MZ1 sensor)', '2a': 'A (samples)', '2b': 'B (Sentinel)',
         '2c': 'C (Sentinel+transfer)', '2d': 'D (samples+transfer)'}
NVALS = [10, 20, 30, 40]
CASE = {'Ref': 'MZ1', '2a': 'MZ2', '2b': 'MZ2', '2c': 'MZ2', '2d': 'MZ2'}


# -------------------------------------------------------------------------
# Figure 1: scenario metrics vs N (pure read from run_summary.json)
# -------------------------------------------------------------------------
def fig_scenario_metrics():
    d = json.load(open(os.path.join(RUN, 'run_summary.json')))
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(9.5, 4))
    for cfg in ['Ref', '2a', '2b', '2c', '2d']:
        if cfg not in d['configs']:
            continue
        Ns, rmsd, nse = [], [], []
        for N in NVALS:
            nd = d['configs'][cfg].get(f'N{N}')
            if not nd or not nd.get('metrics'):
                continue
            m = nd['metrics']
            Ns.append(N)
            rmsd.append(m.get('bcRMSD_val_sensor'))
            nse.append(m.get('bcNSE_val_sensor'))
        if not Ns:
            continue
        ax1.plot(Ns, rmsd, marker=MARK[cfg], color=COLORS[cfg], label=LABEL[cfg], lw=2.2, ms=8)
        ax2.plot(Ns, nse, marker=MARK[cfg], color=COLORS[cfg], label=LABEL[cfg], lw=2.2, ms=8)
    ax1.set_xlabel('Calibration window $N_{cal}$ (days)')
    ax1.set_ylabel('bcRMSD$_{val}$ (m$^3$/m$^3$)')
    ax1.set_title('(a) Validation error')
    ax1.set_xticks(NVALS)
    ax1.grid(alpha=0.3)
    ax2.axhline(0, color='k', lw=0.6, ls=':')
    ax2.set_xlabel('Calibration window $N_{cal}$ (days)')
    ax2.set_ylabel('bcNSE$_{val}$ (--)')
    ax2.set_title('(b) Validation efficiency')
    ax2.set_xticks(NVALS)
    ax2.grid(alpha=0.3)
    ax2.legend(fontsize=8, loc='lower center')
    plt.tight_layout()
    p = os.path.join(FIGDIR, 'res_scenario_metrics.png')
    plt.savefig(p, dpi=200)
    plt.savefig(p.replace('.png', '.pdf'))
    plt.close()
    print('saved', p)


# -------------------------------------------------------------------------
# Figure 2: posterior comparison 2b (uniform) vs 2c (transfer) at N=30
# -------------------------------------------------------------------------
PAR12 = [r'$K_{cb,\mathrm{ini}}$', r'$K_{cb,\mathrm{mid}}$', r'$K_{cb,\mathrm{end}}$',
         r'$L_{\mathrm{ini}}$', r'$L_{\mathrm{dev}}$', r'$L_{\mathrm{mid}}$', r'$f_c$',
         r'$\log K_{\mathrm{sat}}$', r'$CN$', r'$GWT_{\mathrm{max}}$', r'$Z_{r,\mathrm{max}}$',
         r'$v_0$']


def load_ps(cfg, N):
    folder = os.path.join(RUN, f'{CASE[cfg]}_ui_{YEAR}_{cfg}_N{N}days')
    f = os.path.join(folder, f'ParSet_{CASE[cfg]}{YEAR}ui.npy')
    if not os.path.exists(f):
        return None
    ps = np.load(f)
    return ps[int(0.5 * ps.shape[0]):, :-2]


def fig_posteriors_transfer():
    ps_b = load_ps('2b', 30)
    ps_c = load_ps('2c', 30)
    if ps_b is None or ps_c is None:
        print('posteriors: missing ParSet, skip')
        return
    fig, axes = plt.subplots(3, 4, figsize=(9.5, 6.2))
    for i, ax in enumerate(axes.flat):
        lo = min(ps_b[:, i].min(), ps_c[:, i].min())
        hi = max(ps_b[:, i].max(), ps_c[:, i].max())
        bins = np.linspace(lo, hi, 35)
        ax.hist(ps_b[:, i], bins=bins, color=COLORS['2b'], alpha=0.5, density=True,
                label='B (uniform)')
        ax.hist(ps_c[:, i], bins=bins, color=COLORS['2c'], alpha=0.5, density=True,
                label='C (transfer)')
        ax.set_title(PAR12[i], fontsize=15)
        ax.tick_params(labelsize=14)
        ax.set_yticks([])
    _h, _l = axes[0, 0].get_legend_handles_labels()
    fig.legend(_h, _l, loc='lower center', ncol=2, fontsize=13, frameon=True,
               bbox_to_anchor=(0.5, 1.0))
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    p = os.path.join(FIGDIR, 'res_posteriors_transfer.png')
    plt.savefig(p, dpi=200, bbox_inches='tight')
    plt.savefig(p.replace('.png', '.pdf'))
    plt.close()
    print('saved', p)


def fig_ref_diagnostics():
    ps = load_ps('Ref', 30)
    if ps is None:
        print('REF diagnostics: missing ParSet, skip')
        return
    # Posterior histograms
    fig, axes = plt.subplots(3, 4, figsize=(9.5, 6.2))
    for i, ax in enumerate(axes.flat):
        ax.hist(ps[:, i], bins=35, color='#1f77b4', alpha=0.7, density=True)
        ax.axvline(np.median(ps[:, i]), color='k', ls='--', lw=1)
        ax.set_title(PAR12[i], fontsize=15)
        ax.tick_params(labelsize=14)
        ax.set_yticks([])
    plt.tight_layout()
    p = os.path.join(FIGDIR, 'res_ref_posteriors.png')
    plt.savefig(p, dpi=200); plt.savefig(p.replace('.png', '.pdf')); plt.close(); print('saved', p)
    # Correlation matrix
    corr = np.corrcoef(ps.T)
    fig, ax = plt.subplots(figsize=(6.3, 5.5))
    im = ax.imshow(corr, cmap='RdBu_r', vmin=-1, vmax=1)
    ax.set_xticks(range(12)); ax.set_yticks(range(12))
    ax.set_xticklabels(PAR12, rotation=45, ha='right', fontsize=13)
    ax.set_yticklabels(PAR12, fontsize=13)
    for i in range(12):
        for j in range(12):
            if abs(corr[i, j]) > 0.3 and i != j:
                ax.text(j, i, f'{corr[i,j]:.1f}', ha='center', va='center', fontsize=11,
                        color='white' if abs(corr[i, j]) > 0.6 else 'black')
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label='Pearson correlation')
    plt.tight_layout()
    p = os.path.join(FIGDIR, 'res_ref_correlation.png')
    plt.savefig(p, dpi=200); plt.savefig(p.replace('.png', '.pdf')); plt.close(); print('saved', p)


# -------------------------------------------------------------------------
# SWB runner (shared) - adapted from make_figures.py
# -------------------------------------------------------------------------
def setup_swb():
    from swim2_data import (load_soildata, load_eto, load_precipitation,
                            load_irrigation, load_soilobs, load_sensor_overview)
    import SWB_model
    df_soil = load_soildata()
    df_soil_y = df_soil[df_soil['year'].astype(str) == YEAR].copy().reset_index(drop=True)
    df_eto = load_eto(); df_precip = load_precipitation(); df_irrig = load_irrigation()
    df_obs_all = load_soilobs(); df_obs_all = df_obs_all[df_obs_all['year'].astype(str) == YEAR]
    df_crop = pd.read_csv(os.path.join(BASE, 'crop_FAO.csv'), encoding='unicode_escape')
    SWB_model.df_sensor_teler = load_sensor_overview()
    global _DFP, _DFI
    _DFP, _DFI = df_precip, df_irrig
    return df_soil_y, df_eto, df_precip, df_irrig, df_obs_all, df_crop


def run_case(cfg, N, ctx):
    from SWB_model import SWB
    df_soil_y, df_eto, df_precip, df_irrig, df_obs_all, df_crop = ctx
    case = CASE[cfg]
    folder = os.path.join(RUN, f'{case}_ui_{YEAR}_{cfg}_N{N}days')
    pm_file = os.path.join(folder, f'ParSet_MaxLL_{case}{YEAR}ui.npy')
    if not os.path.exists(pm_file):
        return None
    PM = np.load(pm_file)
    df_obs = df_obs_all[df_obs_all['Sensornr'] == case].reset_index(drop=True)
    df_list = [df_precip[df_precip['year'].astype(str) == YEAR].reset_index(drop=True),
               df_irrig[df_irrig['year'].astype(str) == YEAR].reset_index(drop=True),
               df_eto[df_eto['year'].astype(str) == YEAR].reset_index(drop=True),
               df_soil_y, df_obs, df_crop]
    SWC, sw, g_list, _, _, df_obs_grp, _, _ = SWB(
        PM[0], PM[1], PM[2], PM[3], PM[4], PM[5], PM[6], PM[7], PM[8], PM[9], PM[10], PM[11],
        sensor=True, cal='gen', sensor_cal=np.empty(0), CI=np.empty(0),
        show=[False, ''], case=case, year=YEAR, forecast=np.empty(0), df_list=df_list)
    CI = np.empty(0)
    cif = os.path.join(folder, 'CI.npy')
    if os.path.exists(cif):
        CI = np.load(cif, allow_pickle=True)
    return dict(SWC=np.array(SWC), g_list=np.array(g_list, float), CI=CI, df_obs_grp=df_obs_grp)


def sensor_daily(case, g0):
    from swim2_data import load_sensordata, SENSOR_CAL_A, SENSOR_CAL_B
    from Sensordata import ConvertToSerialDate
    v = load_sensordata(case); v.sort_values('Datetime', inplace=True); v.reset_index(drop=True, inplace=True)
    cols = {c: SENSOR_CAL_A + SENSOR_CAL_B * v[c].values for c in ['vwc0 (m3/m3)', 'vwc1 (m3/m3)', 'vwc2 (m3/m3)']}
    serial = np.array([ConvertToSerialDate(v['Datetime'].iloc[i]) for i in range(len(v))])
    arr = np.vstack(list(cols.values())).T
    df = pd.DataFrame({'s': serial, 'v': np.nanmean(arr, axis=1)})
    df = df[(df['v'] > 0.01) & (df['v'] < 1.0)]
    df['day'] = np.floor(df['s']).astype(int)
    g = df.groupby('day', as_index=False)['v'].mean()
    return g['day'].values - int(g0), g['v'].values


def _multicolor_yr_label(ax, fragments, fontsize=12, xpad=1.15):
    """Right y-axis label whose fragments carry individual colours (rotated 90 deg).

    fragments: list of (text, colour). Built bottom-to-top so it reads upward like a
    normal rotated y-label, with each word coloured to match its bars."""
    from matplotlib.offsetbox import TextArea, VPacker, AnchoredOffsetbox
    boxes = [TextArea(t, textprops=dict(color=c, rotation=90, ha='left', va='bottom',
                                        fontsize=fontsize)) for t, c in fragments[::-1]]
    ybox = VPacker(children=boxes, align='center', pad=0, sep=2)
    anchor = AnchoredOffsetbox(loc='center left', child=ybox, pad=0, frameon=False,
                               bbox_to_anchor=(xpad, 0.5), bbox_transform=ax.transAxes,
                               borderpad=0)
    ax.add_artist(anchor)


_FORCING_LABEL = [('Precip.', '#2171b5'), (' / ', '#444444'),
                  ('irrig.', '#d94801'), (' (mm)', '#444444')]


def plot_ts(ax, r, case, cfg, N, ctx_dates, show_legend=False, label_xpad=1.15):
    from Sensordata import ConvertToSerialDate
    from swim2_data import PLANTING_DATE, OBS_START_DATE
    g0 = ConvertToSerialDate(PLANTING_DATE)
    cal0 = int(ConvertToSerialDate(OBS_START_DATE))
    # x-axis is re-based to the observation / calibration start (day 0 = first data);
    # the planting->obs spin-up is cropped by xlim below.
    xmax = 75 - (cal0 - int(g0))
    days = r['g_list'] - cal0
    ax.plot(days, r['SWC'], color=COLORS[cfg], lw=2.2, label='Model (MAP)')
    if r['CI'].size and r['CI'].shape[0] >= 2 and r['CI'].shape[1] <= len(days):
        ax.fill_between(days[:r['CI'].shape[1]], r['CI'][0], r['CI'][1],
                        alpha=0.30, color=COLORS[cfg], label='95% CI')
    sd, sv = sensor_daily(case, cal0)
    ax.plot(sd, sv, 'k.', ms=4.5, alpha=0.75, label='Sensor (15 cm)')
    grp = r['df_obs_grp']
    for i in range(len(grp)):
        dd = int(grp['Date'].iloc[i]) - cal0
        ax.plot(dd, grp['Mean30'].iloc[i], 'r^', ms=8, zorder=5,
                label='Ground sample' if i == 0 else '')
    c0 = 0; c1 = c0 + N - 1; v1 = c1 + 7
    ax.axvspan(c0, c1, alpha=0.10, color='green')
    ax.axvspan(c1 + 1, v1, alpha=0.12, color='orange')
    ax.axvline(c1 + 0.5, color='k', ls='--', lw=0.7)
    # forcing bars (precipitation + irrigation) on a muted secondary axis
    if _DFP is not None and _DFI is not None:
        gi = cal0
        ax2 = ax.twinx()
        pday = _DFP['Date'].values - gi
        pmm = _DFP[case].values.astype(float)
        iday = _DFI['Date'].values - gi
        imm = _DFI[case].values.astype(float)
        ax2.bar(pday, pmm, width=0.9, color='#4292c6', alpha=0.55, zorder=0)
        ax2.bar(iday, imm, width=0.9, color='#e6550d', alpha=0.70, zorder=0)
        pw = pmm[(pday >= 0) & (pday <= xmax)]
        iw = imm[(iday >= 0) & (iday <= xmax)]
        mx = max(pw.max() if pw.size else 1, iw.max() if iw.size else 1, 1)
        ax2.set_ylim(0, mx * 4)
        _multicolor_yr_label(ax2, _FORCING_LABEL, fontsize=13, xpad=label_xpad)
        ax2.tick_params(axis='y', labelsize=15, colors='#444444')
        ax.set_zorder(ax2.get_zorder() + 1)
        ax.patch.set_visible(False)
    ax.set_ylim(0, 0.58)
    ax.set_xlim(0, xmax)
    if show_legend:
        ax.legend(fontsize=12, loc='upper left', ncol=2, framealpha=0.9)


def fig_ref_calval(ctx):
    r = run_case('Ref', 30, ctx)
    if r is None:
        print('REF run missing, skip'); return
    fig, ax = plt.subplots(figsize=(8.5, 3.2))
    plot_ts(ax, r, 'MZ1', 'Ref', 30, None, show_legend=True, label_xpad=1.05)
    ax.set_xlabel('Days since calibration start (24 April 2025)'); ax.set_ylabel('SWC (m$^3$/m$^3$)')
    plt.tight_layout()
    p = os.path.join(FIGDIR, 'res_ref_calval.png'); plt.savefig(p, dpi=200, bbox_inches='tight'); plt.savefig(p.replace('.png', '.pdf')); plt.close()
    print('saved', p)


def fig_mz2_scenarios(ctx):
    fig, axes = plt.subplots(2, 2, figsize=(9.5, 5.6), sharex=True, sharey=True)
    order = ['2a', '2b', '2c', '2d']
    titles = {'2a': 'A: samples only, uniform prior', '2b': 'B: Sentinel+samples, uniform prior',
              '2c': 'C: Sentinel+samples, transfer prior', '2d': 'D: samples only, transfer prior'}
    for ax, cfg in zip(axes.flat, order):
        r = run_case(cfg, 30, ctx)
        if r is None:
            ax.set_visible(False); continue
        plot_ts(ax, r, 'MZ2', cfg, 30, None, show_legend=False)
        ax.set_title(titles[cfg], fontsize=15)
    for ax in axes[1, :]:
        ax.set_xlabel('Days since calibration start (24 April 2025)')
    for ax in axes[:, 0]:
        ax.set_ylabel('SWC (m$^3$/m$^3$)')
    _h, _l = axes.flat[0].get_legend_handles_labels()
    fig.legend(_h, _l, loc='lower center', ncol=4, fontsize=13, frameon=True,
               bbox_to_anchor=(0.5, 1.0))
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    p = os.path.join(FIGDIR, 'res_mz2_scenarios.png'); plt.savefig(p, dpi=200, bbox_inches='tight'); plt.savefig(p.replace('.png', '.pdf')); plt.close()
    print('saved', p)


def fig_all_scenarios(ctx):
    titles = {'Ref': 'REF (MZ1): sensor + samples', '2a': 'A (MZ2): samples, uniform prior',
              '2b': 'B (MZ2): Sentinel + samples, uniform prior',
              '2c': 'C (MZ2): Sentinel + samples, transfer prior',
              '2d': 'D (MZ2): samples, transfer prior'}
    for cfg in ['Ref', '2a', '2b', '2c', '2d']:
        fig, axes = plt.subplots(2, 2, figsize=(9.5, 5.6), sharex=True, sharey=True)
        for ax, N in zip(axes.flat, [10, 20, 30, 40]):
            r = run_case(cfg, N, ctx)
            if r is None:
                ax.set_visible(False); continue
            plot_ts(ax, r, CASE[cfg], cfg, N, None, show_legend=False)
            ax.set_title(f'$N_{{cal}}={N}$ days', fontsize=15)
        for ax in axes[1, :]:
            ax.set_xlabel('Days since calibration start (24 April 2025)')
        for ax in axes[:, 0]:
            ax.set_ylabel('SWC (m$^3$/m$^3$)')
        _h, _l = axes.flat[0].get_legend_handles_labels()
        fig.legend(_h, _l, loc='lower center', ncol=4, fontsize=13, frameon=True,
                   bbox_to_anchor=(0.5, 1.0))
        plt.tight_layout(rect=[0, 0, 1, 0.95])
        p = os.path.join(FIGDIR, f'res_allwin_{cfg}.png')
        plt.savefig(p, dpi=170, bbox_inches='tight'); plt.savefig(p.replace('.png', '.pdf')); plt.close(); print('saved', p)


if __name__ == '__main__':
    try:
        fig_scenario_metrics()
    except Exception as e:
        print('fig_scenario_metrics FAILED:', e)
    try:
        fig_posteriors_transfer()
    except Exception as e:
        print('fig_posteriors_transfer FAILED:', e)
    try:
        fig_ref_diagnostics()
    except Exception as e:
        print('fig_ref_diagnostics FAILED:', e)
    try:
        ctx = setup_swb()
        try:
            fig_ref_calval(ctx)
        except Exception as e:
            import traceback; traceback.print_exc(); print('fig_ref_calval FAILED:', e)
        try:
            fig_mz2_scenarios(ctx)
        except Exception as e:
            import traceback; traceback.print_exc(); print('fig_mz2_scenarios FAILED:', e)
        try:
            fig_all_scenarios(ctx)
        except Exception as e:
            import traceback; traceback.print_exc(); print('fig_all_scenarios FAILED:', e)
    except Exception as e:
        import traceback; traceback.print_exc(); print('SWB setup FAILED:', e)
    print('done')
