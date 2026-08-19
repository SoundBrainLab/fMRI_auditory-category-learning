import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from resampling_stats import leave_one_out_stability, exact_sign_flip_permutation_test, bootstrap_ci

''' Robustness/permutation checks for the key univariate feedback
learning-stage effect (Workstream 5 of the manuscript revision plan):
the anterior caudate vs. putamen learning-stage interaction (Table 8) --
Reviewer #2's "anterior caudate decrease vs. sustained putamen" story,
and the specific pattern Reviewer #1 doubted would replicate at n=12.

Scope is intentionally narrow: this checks the two ROIs that carry the
paper's headline claim, not every ROI/hemisphere/contrast -- doing that
would recreate the multiple-comparisons problem these checks exist to
address.

Input: a long-format, TAB-separated table with one row per participant x
region x hemisphere x learning_stage, and a `beta` column. This is not
a guess -- group_level_all_ROI.ipynb already writes exactly this file
for the striatal/feedback analysis (its cell 213), via:
    out_fname = f'univariate-results_network-{network_name}_contrast-{contrast_label}.tsv'
    roi_df_long.to_csv(os.path.join(group_out_dir, out_fname), sep='\t', ...)
which resolves (network_name='tian-S2', contrast_label='fb-correct-vs-wrong')
to:
    derivatives/nilearn/group_fwhm-0.00/univariate-results_network-tian-S2_contrast-fb-correct-vs-wrong.tsv
No new export needs to be added to the notebook -- this file already
gets written whenever that section runs. Columns confirmed by reading
the notebook: participant_id, region (e.g. aCAU/pCAU/aPUT/pPUT/
NAc-core/NAc-shell), hemisphere (lh/rh), learning_stage (values are
'early'/'middle'/'final' -- NOT 'earlythird' etc., which is only used
in the GLM output folder structure, not this dataframe), beta. No
event_type column is needed or expected, since the file is already
scoped to one contrast; pass --event_type only if working from a
different, combined-contrast CSV that has one.

One thing to confirm once real data is available, not assumed here:
the notebook restricts this section to `sub_list_nman` (non-Mandarin
native speakers) rather than the full participant list, so n may be
smaller than the manuscript's overall n=12 -- the checks below adapt to
however many subjects are actually in the file either way.

Example: python robustness_checks.py \
    --csv=derivatives/nilearn/group_fwhm-0.00/univariate-results_network-tian-S2_contrast-fb-correct-vs-wrong.tsv \
    --sep=tab --region_a=aCAU --region_b=aPUT \
    --out=derivatives/nilearn/group_fwhm-0.00/robustness_summary.csv
'''

STAGE_ORDER = ['early', 'middle', 'final']
TREND_WEIGHTS = np.array([-1, 0, 1])  # linear contrast across early/middle/final


def load_subject_region_stage_means(csv_fpath, regions, event_type=None, sep=','):
    ''' one mean beta per participant x region x learning_stage,
    averaging over hemisphere. `event_type` filtering only applies if
    that column is present -- the real per-contrast TSVs written by
    group_level_all_ROI.ipynb don't have one (see module docstring). '''
    df = pd.read_csv(csv_fpath, sep=sep)
    if event_type is not None and 'event_type' in df.columns:
        df = df[df['event_type'] == event_type]
    df = df[df['region'].isin(regions)]
    return (df.groupby(['participant_id', 'region', 'learning_stage'])['beta']
              .mean()
              .reset_index())


def compute_subject_trends(means_df, region, stage_order=STAGE_ORDER, weights=TREND_WEIGHTS):
    ''' per-subject linear learning-stage trend for one region: the dot
    product of that subject's [early, middle, late] means with `weights`.
    Subjects missing any stage for this region are dropped. Returns a
    pandas Series indexed by participant_id. '''
    trends = {}
    region_df = means_df[means_df['region'] == region]
    for pid, sub_df in region_df.groupby('participant_id'):
        stage_means = sub_df.set_index('learning_stage')['beta'].reindex(stage_order)
        if stage_means.isna().any():
            continue
        trends[pid] = float(np.dot(stage_means.values, weights))
    return pd.Series(trends, name=f'{region}_trend')


def monte_carlo_stage_permutation_test(means_df, region, stage_order=STAGE_ORDER,
                                       weights=TREND_WEIGHTS, n_perms=10000, rng=None):
    ''' Monte Carlo permutation test for the group-level learning-stage
    trend in one region: shuffle stage labels within each subject
    n_perms times, recompute the group-mean trend under each shuffle,
    and compare the observed trend to that null distribution. Used
    instead of exact enumeration since permuting labels across a 3-level
    factor isn't practical to enumerate exactly. '''
    rng = np.random.default_rng() if rng is None else rng

    region_df = means_df[means_df['region'] == region]
    subject_stage_values = []
    for pid, g in region_df.groupby('participant_id'):
        stage_means = g.set_index('learning_stage')['beta'].reindex(stage_order)
        if stage_means.isna().any():
            continue
        subject_stage_values.append(stage_means.values)

    observed_group_trend = np.mean([np.dot(v, weights) for v in subject_stage_values])

    null_trends = np.empty(n_perms)
    for px in range(n_perms):
        null_trends[px] = np.mean([np.dot(rng.permutation(v), weights)
                                   for v in subject_stage_values])

    p_value = float(np.mean(np.abs(null_trends) >= abs(observed_group_trend)))
    return {'observed_group_trend': observed_group_trend,
            'n_permutations': n_perms, 'p_value': p_value}


def run_robustness_checks(csv_fpath, region_a, region_b, out_fpath, event_type=None,
                          sep=',', n_perms=10000, n_boot=10000, seed=0):
    rng = np.random.default_rng(seed)
    means_df = load_subject_region_stage_means(csv_fpath, [region_a, region_b],
                                               event_type=event_type, sep=sep)

    trend_a = compute_subject_trends(means_df, region_a)
    trend_b = compute_subject_trends(means_df, region_b)
    common = trend_a.index.intersection(trend_b.index)
    trend_diff = trend_a[common] - trend_b[common]

    rows = []
    for label, values in [(f'{region_a}_trend', trend_a.values),
                          (f'{region_b}_trend', trend_b.values),
                          (f'{region_a}_minus_{region_b}_trend', trend_diff.values)]:
        loso = leave_one_out_stability(values)
        exact_perm = exact_sign_flip_permutation_test(values)
        boot = bootstrap_ci(values, n_boot=n_boot, rng=rng)
        rows.append({
            'contrast': label,
            'n_subjects': len(values),
            'original_t': loso['original_t'],
            'original_p': loso['original_p'],
            'loso_folds_preserved': loso['n_preserved'],
            'loso_n_folds': loso['n_folds'],
            'exact_permutation_p': exact_perm['p_value'],
            'bootstrap_mean': boot['mean'],
            'bootstrap_ci_lower': boot['ci_lower'],
            'bootstrap_ci_upper': boot['ci_upper'],
        })

    for region in [region_a, region_b]:
        mc = monte_carlo_stage_permutation_test(means_df, region, n_perms=n_perms, rng=rng)
        rows.append({
            'contrast': f'{region}_stage_montecarlo',
            'n_subjects': means_df[means_df['region'] == region]['participant_id'].nunique(),
            'monte_carlo_observed_trend': mc['observed_group_trend'],
            'monte_carlo_p': mc['p_value'],
        })

    summary = pd.DataFrame(rows)
    summary.to_csv(out_fpath, index=False)
    print(f'saved robustness summary to {out_fpath}')
    return summary


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Robustness/permutation checks on the anterior caudate vs. '
                    'putamen feedback learning-stage effect',
        epilog=('Example: python robustness_checks.py '
                '--csv=derivatives/nilearn/group_fwhm-0.00/univariate-results_network-tian-S2_contrast-fb-correct-vs-wrong.tsv '
                '--sep=tab --region_a=aCAU --region_b=aPUT '
                '--out=derivatives/nilearn/group_fwhm-0.00/robustness_summary.csv'))
    parser.add_argument('--csv', help='long-format ROI beta table, see module docstring for schema', type=str)
    parser.add_argument('--sep', help="field separator: 'tab' for the real .tsv files, or a literal "
                        "character like ',' (default: ',')", type=str, default=',')
    parser.add_argument('--event_type', help='only needed for a combined-contrast CSV that has an '
                        'event_type column; the real per-contrast .tsv files do not', type=str, default=None)
    parser.add_argument('--region_a', help='first region label, e.g. aCAU', type=str, default='aCAU')
    parser.add_argument('--region_b', help='second region label, e.g. aPUT', type=str, default='aPUT')
    parser.add_argument('--out', help='output summary CSV path', type=str)
    parser.add_argument('--n_perms', help='Monte Carlo permutations', type=int, default=10000)
    parser.add_argument('--n_boot', help='bootstrap resamples', type=int, default=10000)
    parser.add_argument('--seed', help='random seed for Monte Carlo/bootstrap', type=int, default=0)
    args = parser.parse_args()

    sep = '\t' if args.sep == 'tab' else args.sep
    run_robustness_checks(args.csv, args.region_a, args.region_b, args.out,
                          event_type=args.event_type, sep=sep,
                          n_perms=args.n_perms, n_boot=args.n_boot, seed=args.seed)
