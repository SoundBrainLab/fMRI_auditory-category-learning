import argparse
import itertools

import numpy as np
import pandas as pd
from scipy import stats

''' Robustness/permutation checks for the key univariate feedback
learning-stage effect (Workstream 5 of the manuscript revision plan):
the anterior caudate vs. putamen learning-stage interaction (Table 8) --
Reviewer #2's "anterior caudate decrease vs. sustained putamen" story,
and the specific pattern Reviewer #1 doubted would replicate at n=12.

Scope is intentionally narrow: this checks the two ROIs that carry the
paper's headline claim, not every ROI/hemisphere/contrast -- doing that
would recreate the multiple-comparisons problem these checks exist to
address.

Input: a long-format CSV with one row per participant x region x
hemisphere x learning_stage x event_type, and a `beta` column -- the
same schema group_level_all_ROI.ipynb already builds for its
AnovaRM/pairwise_tests calls (columns: participant_id, region,
hemisphere, learning_stage, event_type, beta). That notebook has
several near-duplicate analysis blocks (likely per-network iterations),
so exporting the right one to CSV needs to be wired up by hand once
real data exists -- this script does not assume where that export
happens, only what its output looks like.

Example: python robustness_checks.py \
    --csv=derivatives/nilearn/group_level/roi_values_long.csv \
    --event_type=feedback --region_a=aCAU --region_b=aPUT \
    --out=derivatives/nilearn/group_level/robustness_summary.csv
'''

STAGE_ORDER = ['earlythird', 'middlethird', 'latethird']
TREND_WEIGHTS = np.array([-1, 0, 1])  # linear contrast across early/middle/late


def load_subject_region_stage_means(csv_fpath, event_type, regions):
    ''' one mean beta per participant x region x learning_stage,
    averaging over hemisphere '''
    df = pd.read_csv(csv_fpath)
    df = df[(df['event_type'] == event_type) & (df['region'].isin(regions))]
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


def leave_one_out_stability(values, alpha=0.05):
    ''' refit a one-sample t-test against 0, leaving out each subject in
    turn. Reports how many of the N folds preserve the original effect's
    direction and significance -- a stable effect across all folds is a
    concrete answer to "would this replicate"; one that flips when a
    particular subject drops out is exactly the fragility reviewers
    doubted. '''
    values = np.asarray(values, dtype=float)
    n = len(values)
    orig_t, orig_p = stats.ttest_1samp(values, 0)
    orig_sign = np.sign(orig_t)

    n_preserved = 0
    for i in range(n):
        loo_t, loo_p = stats.ttest_1samp(np.delete(values, i), 0)
        if np.sign(loo_t) == orig_sign and loo_p < alpha:
            n_preserved += 1

    return {'original_t': orig_t, 'original_p': orig_p,
            'n_folds': n, 'n_preserved': n_preserved}


def exact_sign_flip_permutation_test(values):
    ''' exact two-sided permutation test for a one-sample scalar against
    0, enumerating all 2^n possible sign flips. Feasible up to about
    n=20 subjects (2^20 = ~1M); at n=12 (2^12 = 4096) this is exact
    rather than a Monte Carlo approximation -- a real upside of small n,
    not just a limitation. '''
    values = np.asarray(values, dtype=float)
    n = len(values)
    if n > 20:
        raise ValueError('exact enumeration infeasible above ~20 subjects; use Monte Carlo instead')

    observed = np.mean(values)
    n_perms = 0
    n_as_extreme = 0
    for signs in itertools.product([1, -1], repeat=n):
        permuted_mean = np.mean(np.asarray(signs) * values)
        if abs(permuted_mean) >= abs(observed) - 1e-12:
            n_as_extreme += 1
        n_perms += 1

    return {'observed_mean': observed, 'n_permutations': n_perms,
            'p_value': n_as_extreme / n_perms}


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


def bootstrap_ci(values, n_boot=10000, ci=0.95, rng=None):
    ''' percentile bootstrap CI on the mean of a per-subject scalar --
    defends an effect size without leaning on parametric normality
    assumptions at n=12. '''
    rng = np.random.default_rng() if rng is None else rng
    values = np.asarray(values, dtype=float)
    boot_means = np.array([np.mean(rng.choice(values, size=len(values), replace=True))
                           for _ in range(n_boot)])
    alpha = 1 - ci
    return {'mean': float(np.mean(values)),
            'ci_lower': float(np.percentile(boot_means, 100 * alpha / 2)),
            'ci_upper': float(np.percentile(boot_means, 100 * (1 - alpha / 2))),
            'n_boot': n_boot}


def run_robustness_checks(csv_fpath, event_type, region_a, region_b, out_fpath,
                          n_perms=10000, n_boot=10000, seed=0):
    rng = np.random.default_rng(seed)
    means_df = load_subject_region_stage_means(csv_fpath, event_type, [region_a, region_b])

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
                '--csv=derivatives/nilearn/group_level/roi_values_long.csv '
                '--event_type=feedback --region_a=aCAU --region_b=aPUT '
                '--out=derivatives/nilearn/group_level/robustness_summary.csv'))
    parser.add_argument('--csv', help='long-format ROI beta CSV, see module docstring for schema', type=str)
    parser.add_argument('--event_type', help='e.g. feedback or sound', type=str, default='feedback')
    parser.add_argument('--region_a', help='first region label, e.g. aCAU', type=str, default='aCAU')
    parser.add_argument('--region_b', help='second region label, e.g. aPUT', type=str, default='aPUT')
    parser.add_argument('--out', help='output summary CSV path', type=str)
    parser.add_argument('--n_perms', help='Monte Carlo permutations', type=int, default=10000)
    parser.add_argument('--n_boot', help='bootstrap resamples', type=int, default=10000)
    parser.add_argument('--seed', help='random seed for Monte Carlo/bootstrap', type=int, default=0)
    args = parser.parse_args()

    run_robustness_checks(args.csv, args.event_type, args.region_a, args.region_b,
                          args.out, n_perms=args.n_perms, n_boot=args.n_boot, seed=args.seed)
