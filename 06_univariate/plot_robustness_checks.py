import argparse
import sys
from pathlib import Path

import numpy as np
import itertools
import matplotlib.pyplot as plt
from scipy import stats

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from resampling_stats import bootstrap_ci
from robustness_checks import (load_subject_region_stage_means, compute_subject_trends,
                               STAGE_ORDER, TREND_WEIGHTS)

'''
Figure companion to robustness_checks.py -- draws the LOSO / permutation /
bootstrap checks on the anterior caudate vs. putamen feedback learning-stage
effect, so "does this replicate at n=12" is answered visually, not just in a
summary table. One combined "robustness panel" figure: rows are the three
contrasts (aCAU trend, aPUT trend, aCAU-minus-aPUT), columns are

  1. per-subject data   -- the raw early/middle/final betas (region rows) or the
                           per-subject aCAU-vs-aPUT trend pair (difference row),
                           so the reader sees the effect isn't carried by one subject
  2. LOSO caterpillar   -- the one-sample t per leave-one-subject-out fold, vs.
                           the full-sample t and the +/- p=.05 critical value;
                           filled dot = fold preserves direction AND significance
  3. permutation null    -- all 2^n exact sign-flip permuted means, observed marked,
                           two-sided tail shaded, p annotated
  4. bootstrap            -- ~10k resampled means, 95% percentile CI band, 0 marked

Reads the same TSV as robustness_checks.py (written by group_level_all_ROI.ipynb's
striatal x feedback cell). Recomputes the per-subject trend scalars and the
resampling intermediates here rather than threading arrays back through
resampling_stats.py's summary API.

Example: python plot_robustness_checks.py \
    --csv=derivatives/nilearn/group_fwhm-0.00/univariate-results_network-tian-S2_contrast-fb-correct-vs-wrong.tsv \
    --sep=tab --region_a=aCAU --region_b=aPUT \
    --out_dir=derivatives/nilearn/group_fwhm-0.00/figures/
'''

# minimal, CVD-safe: neutral ink for marks/text, one accent for the observed
# effect, light grey for null mass. "preserved" is carried by fill (filled vs
# hollow), not colour alone.
INK = '#3d3d3d'
ACCENT = '#2f6db0'
NULL_GREY = '#c9c9c9'
MUTED = '#8a8a8a'

ALPHA = 0.05


def loo_fold_ts(values):
    ''' one-sample t against 0 for each leave-one-subject-out fold, plus
    whether that fold preserves the full-sample effect's sign and
    significance (matching resampling_stats.leave_one_out_stability). '''
    values = np.asarray(values, float)
    orig_t, _ = stats.ttest_1samp(values, 0)
    orig_sign = np.sign(orig_t)
    rows = []
    for i in range(len(values)):
        loo_t, loo_p = stats.ttest_1samp(np.delete(values, i), 0)
        rows.append((i, float(loo_t), np.sign(loo_t) == orig_sign and loo_p < ALPHA))
    return orig_t, rows


def sign_flip_null_means(values):
    ''' every 2^n sign-flip permuted mean (exact null for a one-sample
    scalar against 0). '''
    values = np.asarray(values, float)
    return np.array([np.mean(np.asarray(signs) * values)
                     for signs in itertools.product([1, -1], repeat=len(values))])


def _panel_subject_slopes(ax, means_df, region):
    sub_means = (means_df[means_df['region'] == region]
                 .pivot(index='participant_id', columns='learning_stage', values='beta')
                 .reindex(columns=STAGE_ORDER).dropna())
    x = np.arange(len(STAGE_ORDER))
    for _, row in sub_means.iterrows():
        ax.plot(x, row.values, color=MUTED, lw=0.8, alpha=0.6, zorder=2)
    ax.plot(x, sub_means.mean(axis=0).values, color=INK, lw=2.2, zorder=4,
            marker='o', ms=5)
    ax.axhline(0, color=NULL_GREY, lw=1, ls='--', zorder=1)
    ax.set_xticks(x)
    ax.set_xticklabels(STAGE_ORDER)
    ax.set_ylabel(f'{region}  fb betas')
    ax.set_title('per-subject', fontsize=9, color=MUTED)


def _panel_subject_pairs(ax, trend_a, trend_b, region_a, region_b):
    common = trend_a.index.intersection(trend_b.index)
    a, b = trend_a[common].values, trend_b[common].values
    for ai, bi in zip(a, b):
        ax.plot([0, 1], [ai, bi], color=MUTED, lw=0.8, alpha=0.6, zorder=2)
    ax.plot([0, 1], [a.mean(), b.mean()], color=INK, lw=2.2, marker='o', ms=5, zorder=4)
    ax.axhline(0, color=NULL_GREY, lw=1, ls='--', zorder=1)
    ax.set_xticks([0, 1])
    ax.set_xticklabels([f'{region_a}\ntrend', f'{region_b}\ntrend'])
    ax.set_ylabel('learning-stage trend')
    ax.set_title('per-subject', fontsize=9, color=MUTED)


def _panel_loso(ax, values):
    orig_t, rows = loo_fold_ts(values)
    rows_sorted = sorted(rows, key=lambda r: r[1])
    df = len(values) - 1
    t_crit = stats.t.ppf(1 - ALPHA / 2, df)
    n_pres = sum(r[2] for r in rows)

    ax.axvspan(-t_crit, t_crit, color=NULL_GREY, alpha=0.35, zorder=1)
    ax.axvline(orig_t, color=ACCENT, lw=1.6, zorder=2)
    for y, (_, t, preserved) in enumerate(rows_sorted):
        ax.plot(t, y, marker='o', ms=6, zorder=3,
                mfc=INK if preserved else 'white', mec=INK, mew=1.2)
    ax.set_yticks([])
    ax.set_ylabel('LOSO folds')
    ax.set_xlabel('one-sample t')
    ax.set_title(f'{n_pres}/{len(values)} folds preserved', fontsize=9, color=MUTED)


def _panel_permutation(ax, values):
    null_means = sign_flip_null_means(values)
    obs = float(np.mean(values))
    p = float(np.mean(np.abs(null_means) >= abs(obs) - 1e-12))

    lo, hi = null_means.min(), null_means.max()
    within = lo <= obs <= hi

    ax.hist(null_means, bins=40, color=NULL_GREY, zorder=1)
    # mark the two-sided "as extreme as observed" boundaries -- histogram mass
    # outside these lines is the p-value, without a muddy fill overlay
    if within:
        for b in (abs(obs), -abs(obs)):
            ax.axvline(b, color=MUTED, lw=1, ls=':', zorder=2)
    ax.axvline(obs, color=ACCENT, lw=1.8, zorder=3)

    pad = 0.10 * (hi - lo)
    ax.set_xlim(min(lo, obs) - pad, max(hi, obs) + pad)
    ax.annotate(f'observed {obs:.2f}', xy=(obs, 0.9), xycoords=('data', 'axes fraction'),
                xytext=(6 if obs < 0 else -6, 0), textcoords='offset points',
                ha='left' if obs < 0 else 'right', va='top', fontsize=8, color=ACCENT)
    ax.set_yticks([])
    ax.set_xlabel('sign-flip null mean')
    tail = '' if within else '  (beyond every permutation)'
    ax.set_title(f'exact p = {p:.4f}  ({len(null_means)} perms){tail}',
                 fontsize=9, color=MUTED)


def _panel_bootstrap(ax, values, n_boot, rng):
    boot = np.array([np.mean(rng.choice(values, size=len(values), replace=True))
                     for _ in range(n_boot)])
    lo, hi = np.percentile(boot, [2.5, 97.5])
    ax.hist(boot, bins=40, color=NULL_GREY, zorder=1)
    ax.axvspan(lo, hi, color=ACCENT, alpha=0.18, zorder=2)
    ax.axvline(0, color=INK, lw=1, ls='--', zorder=3)
    ax.axvline(float(np.mean(values)), color=ACCENT, lw=1.6, zorder=4)
    ax.set_yticks([])
    ax.set_xlabel('bootstrap mean')
    excl = 'excludes 0' if (lo > 0 or hi < 0) else 'includes 0'
    ax.set_title(f'95% CI [{lo:.2f}, {hi:.2f}] — {excl}', fontsize=9, color=MUTED)


def _draw_row(axes_row, label, values, col1_fn, n_boot, rng):
    ''' one contrast across the four robustness columns. col1_fn(ax) draws
    whatever raw per-subject view belongs in column 1. '''
    col1_fn(axes_row[0])
    _panel_loso(axes_row[1], values)
    _panel_permutation(axes_row[2], values)
    _panel_bootstrap(axes_row[3], values, n_boot, rng)
    axes_row[0].annotate(label, xy=(-0.42, 0.5), xycoords='axes fraction',
                         rotation=90, va='center', ha='center',
                         fontsize=11, fontweight='bold', color=INK)


def _finalize(fig, axes, out_dir, stem, suptitle):
    for ax in axes.flat:
        ax.spines[['top', 'right']].set_visible(False)
        ax.tick_params(labelsize=8)
    fig.suptitle(suptitle, fontsize=12, fontweight='bold')
    fig.tight_layout(rect=[0.03, 0, 1, 0.97])
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    for ext in ('png', 'svg'):
        fpath = out_dir / f'{stem}.{ext}'
        fig.savefig(fpath, bbox_inches='tight')
        print(f'saved {fpath}')
    return fig


def make_panel(csv_fpath, region_a, region_b, out_dir, sep=',', event_type=None,
               n_boot=10000, seed=0):
    ''' 3-row striatal panel: region_a trend, region_b trend, their difference. '''
    rng = np.random.default_rng(seed)
    means_df = load_subject_region_stage_means(csv_fpath, [region_a, region_b],
                                               event_type=event_type, sep=sep)
    trend_a = compute_subject_trends(means_df, region_a)
    trend_b = compute_subject_trends(means_df, region_b)
    common = trend_a.index.intersection(trend_b.index)
    trend_diff = (trend_a[common] - trend_b[common])

    fig, axes = plt.subplots(3, 4, figsize=(14, 9), dpi=300)
    _draw_row(axes[0], f'{region_a} trend', trend_a.values,
              lambda ax: _panel_subject_slopes(ax, means_df, region_a), n_boot, rng)
    _draw_row(axes[1], f'{region_b} trend', trend_b.values,
              lambda ax: _panel_subject_slopes(ax, means_df, region_b), n_boot, rng)
    _draw_row(axes[2], f'{region_a} − {region_b} trend', trend_diff.values,
              lambda ax: _panel_subject_pairs(ax, trend_a, trend_b, region_a, region_b),
              n_boot, rng)

    return _finalize(fig, axes, out_dir, f'robustness_panel_{region_a}_vs_{region_b}',
                     'Robustness of the anterior caudate vs. putamen feedback '
                     f'learning-stage effect (n={len(trend_a)})')


def make_spc_panel(csv_fpath, out_dir, sep=',', n_boot=10000, seed=0):
    ''' 1-row behavioral panel: SPC's relative-BIC change from early to final
    learning stage (from model_comparison.py's spc_relative_bic_early_late.csv).
    Negative = SPC fits relatively better than competitors late in learning. '''
    import pandas as pd
    rng = np.random.default_rng(seed)
    sep = '\t' if sep == 'tab' else sep
    df = pd.read_csv(csv_fpath, sep=sep)
    values = df['late_minus_early'].values

    def col1(ax):
        for _, row in df.iterrows():
            ax.plot([0, 1], [row['early'], row['final']], color=MUTED, lw=0.8,
                    alpha=0.6, zorder=2)
        ax.plot([0, 1], [df['early'].mean(), df['final'].mean()], color=INK,
                lw=2.2, marker='o', ms=5, zorder=4)
        ax.axhline(0, color=NULL_GREY, lw=1, ls='--', zorder=1)
        ax.set_xticks([0, 1])
        ax.set_xticklabels(['early', 'final'])
        ax.set_ylabel('SPC Δ-BIC vs. best competitor')
        ax.set_title('per-subject', fontsize=9, color=MUTED)

    fig, axes = plt.subplots(1, 4, figsize=(14, 3.4), dpi=300)
    _draw_row(axes, 'SPC early→final\nΔ-BIC', values, col1, n_boot, rng)
    return _finalize(fig, axes[None, :], out_dir, 'robustness_panel_spc_trend',
                     'Robustness of the SPC relative-fit learning trend '
                     f'(n={len(values)})')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Robustness-panel figures (LOSO / exact permutation / bootstrap) '
                    'for the aCAU vs. aPUT feedback learning-stage effect, or the SPC '
                    'behavioral learning trend',
        epilog=('Examples:\n'
                '  python plot_robustness_checks.py --mode=roi '
                '--csv=derivatives/nilearn/group_fwhm-0.00/univariate-results_network-tian-S2_contrast-fb-correct-vs-wrong.tsv '
                '--sep=tab --out_dir=derivatives/nilearn/group_fwhm-0.00/figures/\n'
                '  python plot_robustness_checks.py --mode=spc '
                '--csv=derivatives/behavior/spc_relative_bic_early_late.csv '
                '--out_dir=derivatives/behavior/figures/'),
        formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--mode', choices=['roi', 'spc'], default='roi',
                        help="'roi' (default): 3-row striatal panel from the ROI beta TSV. "
                             "'spc': 1-row behavioral panel from model_comparison.py's "
                             "spc_relative_bic_early_late.csv")
    parser.add_argument('--csv', required=True, help='input table (see --mode)')
    parser.add_argument('--sep', default=',',
                        help="field separator: 'tab' for the ROI .tsv, or a literal character")
    parser.add_argument('--event_type', default=None,
                        help='roi mode only: for a combined-contrast CSV that has an event_type column')
    parser.add_argument('--region_a', default='aCAU', help='roi mode only')
    parser.add_argument('--region_b', default='aPUT', help='roi mode only')
    parser.add_argument('--out_dir', required=True, help='directory to write the figure to')
    parser.add_argument('--n_boot', type=int, default=10000)
    parser.add_argument('--seed', type=int, default=0)
    args = parser.parse_args()

    if args.mode == 'roi':
        sep = '\t' if args.sep == 'tab' else args.sep
        make_panel(args.csv, args.region_a, args.region_b, args.out_dir,
                   sep=sep, event_type=args.event_type, n_boot=args.n_boot, seed=args.seed)
    else:
        make_spc_panel(args.csv, args.out_dir, sep=args.sep,
                       n_boot=args.n_boot, seed=args.seed)
