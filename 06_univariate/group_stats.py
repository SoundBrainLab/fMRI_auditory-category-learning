import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pingouin as pg
from scipy import stats
from statsmodels.stats.multitest import multipletests

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from resampling_stats import (leave_one_out_stability, exact_sign_flip_permutation_test, bootstrap_ci,
                              leave_one_out_two_groups, group_label_permutation_test,
                              bootstrap_ci_two_groups)
from robustness_checks import STAGE_ORDER, TREND_WEIGHTS

''' Two-group (non-Mandarin learners vs. Mandarin native listeners) ROI
statistics for group_level_two_group.ipynb. Reads the long tables written
by extract_roi_values.py -- one row per participant x region x hemisphere
x learning stage, with a participant_group column -- so the notebook stays
a thin sequence of calls and every analysis treats both groups the same way.

Region-level analyses average over hemisphere first (as the notebook's
"Compare contrasts" section does); the per-subject learning-stage trend is
the same [-1, 0, 1] contrast robustness_checks.py uses, so the learners'
values here reproduce that script's.
'''

GROUPS = ('non-Mandarin', 'Mandarin')
NETWORKS = ('tian-S2', 'auditory', 'pfc')
# resp stays in the first-level model as a nuisance regressor but isn't
# reported: it follows the sound by <1 s, too close to separate at TR=3 s
CONTRASTS = ('sound', 'fbcorrectvswrong', 'fbcorrect')

# the auditory and PFC networks both draw these two labels from the same
# atlas region; keep the auditory copy so each region is tested once
DUPLICATE_REGIONS = {('pfc', 'IFGo'): ('auditory', 'ParsOp'),
                     ('pfc', 'IFGt'): ('auditory', 'ParsTri')}


def fdr(p_values):
    ''' Benjamini-Hochberg q-values, NaN-safe '''
    p = np.asarray(p_values, dtype=float)
    q = np.full_like(p, np.nan)
    ok = ~np.isnan(p)
    if ok.any():
        q[ok] = multipletests(p[ok], method='fdr_bh')[1]
    return q


def load_roi_values(roi_dir, networks=NETWORKS, contrasts=CONTRASTS, drop_duplicate_regions=True):
    ''' all extract_roi_values.py tables in roi_dir as one long dataframe,
    with `network` and `contrast` columns added '''
    tables = []
    for network in networks:
        for contrast in contrasts:
            fpath = os.path.join(roi_dir, f'roi-values_network-{network}_contrast-{contrast}.tsv')
            if not os.path.exists(fpath):
                print(f'not found, skipping: {fpath}')
                continue
            table = pd.read_csv(fpath, sep='\t')
            table['network'], table['contrast'] = network, contrast
            tables.append(table)
    if not tables:
        raise FileNotFoundError(f'no roi-values tables found in {roi_dir}')
    df = pd.concat(tables, ignore_index=True)

    if drop_duplicate_regions:
        is_dup = pd.Series(False, index=df.index)
        for (network, region), kept in DUPLICATE_REGIONS.items():
            is_dup |= (df.network == network) & (df.region == region)
        df = df[~is_dup].reset_index(drop=True)
    return df


def coverage(df):
    ''' participants per contrast x group x learning stage -- the first
    thing to check, since natives lack fbcorrectvswrong in some stages '''
    return (df.groupby(['contrast', 'participant_group', 'learning_stage'])
              .participant_id.nunique()
              .unstack('learning_stage')[list(STAGE_ORDER)])


def bilateral_mean(df):
    ''' average over hemisphere: one value per participant x region x stage '''
    keys = ['network', 'contrast', 'participant_id', 'participant_group', 'region', 'learning_stage']
    return df.groupby(keys, as_index=False)['beta'].mean()


def baseline_tests(df, by_stage=False):
    ''' one-sample t-test against 0 per group x contrast x network x region
    (stages averaged first unless by_stage), FDR-corrected across regions
    within each group x contrast x network (x stage) '''
    bilat = bilateral_mean(df)
    unit = ['network', 'contrast', 'participant_group', 'region'] + (['learning_stage'] if by_stage else [])
    per_subject = bilat.groupby(unit + ['participant_id'], as_index=False)['beta'].mean()

    rows = []
    for key, grp in per_subject.groupby(unit):
        values = grp['beta'].values
        t, p = stats.ttest_1samp(values, 0)
        rows.append(dict(zip(unit, key), n=len(values), mean=values.mean(),
                         t=t, df=len(values) - 1, p=p, d=values.mean() / values.std(ddof=1)))
    out = pd.DataFrame(rows)
    family = [c for c in unit if c != 'region']
    out['q'] = out.groupby(family)['p'].transform(fdr)
    return out


def group_difference_tests(df, contrasts=('sound', 'fbcorrect'), by_stage=False,
                           groups=GROUPS, n_perms=10000, seed=0):
    ''' learners - natives per contrast x network x region (x stage):
    Welch t, Cohen's d (pooled SD), group-label permutation p. Stages are
    averaged per participant first unless by_stage. FDR across regions
    within each contrast x network (x stage), as in baseline_tests.
    fbcorrectvswrong is left out by default -- natives mostly lack it. '''
    rng = np.random.default_rng(seed)
    bilat = bilateral_mean(df)
    bilat = bilat[bilat.contrast.isin(contrasts)]
    unit = ['network', 'contrast', 'region'] + (['learning_stage'] if by_stage else [])
    per_subject = bilat.groupby(unit + ['participant_id', 'participant_group'], as_index=False)['beta'].mean()

    rows = []
    for key, grp in per_subject.groupby(unit):
        a = grp.loc[grp.participant_group == groups[0], 'beta'].values
        b = grp.loc[grp.participant_group == groups[1], 'beta'].values
        if len(a) < 2 or len(b) < 2:
            continue
        t, p = stats.ttest_ind(a, b, equal_var=False)
        pooled_sd = np.sqrt(((len(a) - 1) * a.var(ddof=1) + (len(b) - 1) * b.var(ddof=1)) / (len(a) + len(b) - 2))
        perm = group_label_permutation_test(a, b, n_perms=n_perms, rng=rng) if n_perms else {'p_value': np.nan}
        rows.append(dict(zip(unit, key), n_a=len(a), n_b=len(b), mean_a=a.mean(), mean_b=b.mean(),
                         difference=a.mean() - b.mean(), t=t, p=p, d=(a.mean() - b.mean()) / pooled_sd,
                         permutation_p=perm['p_value']))
    out = pd.DataFrame(rows).rename(columns={'n_a': f'n_{groups[0]}', 'n_b': f'n_{groups[1]}',
                                             'mean_a': f'mean_{groups[0]}', 'mean_b': f'mean_{groups[1]}'})
    family = [c for c in unit if c != 'region']
    out['q'] = out.groupby(family)['p'].transform(fdr)
    return out


def subject_trends(df):
    ''' per-subject linear learning-stage trend (weights [-1, 0, 1] across
    early/middle/final) per contrast x network x region, hemispheres
    averaged. Participants missing a stage for that contrast are dropped. '''
    bilat = bilateral_mean(df)
    wide = bilat.pivot_table(index=['network', 'contrast', 'participant_id', 'participant_group', 'region'],
                             columns='learning_stage', values='beta')
    wide = wide.reindex(columns=list(STAGE_ORDER)).dropna()
    trends = wide.values @ TREND_WEIGHTS
    return wide.index.to_frame(index=False).assign(trend=trends)


def differential_trend(trends, contrast, region_a, region_b):
    ''' per-subject region_a-minus-region_b trend (e.g. aCAU - aPUT), shaped
    like subject_trends' output so it can go straight into
    trend_group_difference '''
    sub = trends[trends.contrast == contrast].set_index(['participant_id', 'participant_group', 'region'])['trend']
    a = sub.xs(region_a, level='region')
    b = sub.xs(region_b, level='region')
    diff = (a - b).dropna().reset_index()
    return diff.assign(contrast=contrast, region=f'{region_a}-minus-{region_b}', network='tian-S2')


def group_by_stage_anova(df, contrasts=None):
    ''' mixed ANOVA (learning_stage within, participant_group between) per
    contrast x network x region on hemisphere-averaged values, using only
    participants with all three stages. FDR per contrast x effect across
    all regions. '''
    bilat = bilateral_mean(df)
    if contrasts is not None:
        bilat = bilat[bilat.contrast.isin(contrasts)]
    rows = []
    for (network, contrast, region), grp in bilat.groupby(['network', 'contrast', 'region']):
        complete = grp.groupby('participant_id').learning_stage.nunique() == len(STAGE_ORDER)
        grp = grp[grp.participant_id.isin(complete[complete].index)]
        if grp.participant_group.nunique() < 2:
            continue
        aov = pg.mixed_anova(data=grp, dv='beta', within='learning_stage',
                             between='participant_group', subject='participant_id')
        # pingouin has renamed these across versions (DF1/ddof1, p-unc/p_unc)
        col = {name: next(c for c in options if c in aov.columns)
               for name, options in {'df1': ('DF1', 'ddof1'), 'df2': ('DF2', 'ddof2'),
                                     'p': ('p_unc', 'p-unc')}.items()}
        n_per_group = grp.groupby('participant_group').participant_id.nunique().to_dict()
        for _, r in aov.iterrows():
            rows.append({'network': network, 'contrast': contrast, 'region': region,
                         'effect': {'participant_group': 'group', 'learning_stage': 'stage',
                                    'Interaction': 'group x stage'}[r['Source']],
                         'F': r['F'], 'df1': r[col['df1']], 'df2': r[col['df2']], 'p': r[col['p']],
                         'np2': r['np2'], 'n_per_group': n_per_group})
    out = pd.DataFrame(rows)
    out['q'] = out.groupby(['contrast', 'effect'])['p'].transform(fdr)
    return out


def trend_group_difference(trends, contrast, region, groups=GROUPS, n_perms=10000, n_boot=10000, seed=0):
    ''' the planned two-group test on one region's learning-stage trend:
    within each group (one-sample t, exact sign-flip, LOSO, bootstrap) and
    between groups (Welch t, group-label permutation, LOSO, bootstrap CI
    on the difference). Returns one row per test as a dataframe. '''
    rng = np.random.default_rng(seed)
    sub = trends[(trends.contrast == contrast) & (trends.region == region)]
    values = {g: sub.loc[sub.participant_group == g, 'trend'].values for g in groups}

    rows = []
    for g in groups:
        loso = leave_one_out_stability(values[g])
        exact = exact_sign_flip_permutation_test(values[g])
        boot = bootstrap_ci(values[g], n_boot=n_boot, rng=rng)
        rows.append({'test': f'{g} trend vs 0', 'n': len(values[g]), 'estimate': boot['mean'],
                     't': loso['original_t'], 'p': loso['original_p'], 'permutation_p': exact['p_value'],
                     'loso_preserved': f"{loso['n_preserved']}/{loso['n_folds']}",
                     'ci_lower': boot['ci_lower'], 'ci_upper': boot['ci_upper']})

    a, b = values[groups[0]], values[groups[1]]
    loso = leave_one_out_two_groups(a, b)
    perm = group_label_permutation_test(a, b, n_perms=n_perms, rng=rng)
    boot = bootstrap_ci_two_groups(a, b, n_boot=n_boot, rng=rng)
    rows.append({'test': f'{groups[0]} - {groups[1]} (Welch)', 'n': len(a) + len(b),
                 'estimate': boot['difference'], 't': loso['original_t'], 'p': loso['original_p'],
                 'permutation_p': perm['p_value'],
                 'loso_preserved': f"{loso['n_preserved']}/{loso['n_folds']}",
                 'ci_lower': boot['ci_lower'], 'ci_upper': boot['ci_upper']})
    return pd.DataFrame(rows).assign(contrast=contrast, region=region)


def learner_contrast_agreement(trends, region, group='non-Mandarin'):
    ''' within learners, does fbcorrect (vs. baseline) show the same
    learning-stage trend as the headline fbcorrectvswrong? The two-group
    feedback comparison rests on fbcorrect, so this has to hold first
    (TWO_GROUP_MANUSCRIPT_PLAN.md, R3). '''
    sub = trends[(trends.region == region) & (trends.participant_group == group)]
    wide = sub.pivot(index='participant_id', columns='contrast', values='trend')
    wide = wide[['fbcorrectvswrong', 'fbcorrect']].dropna()
    rows = []
    for contrast in ['fbcorrectvswrong', 'fbcorrect']:
        t, p = stats.ttest_1samp(wide[contrast], 0)
        rows.append({'contrast': contrast, 'n': len(wide), 'mean_trend': wide[contrast].mean(), 't': t, 'p': p})
    r, p_r = stats.pearsonr(wide['fbcorrectvswrong'], wide['fbcorrect'])
    rows.append({'contrast': 'correlation between the two', 'n': len(wide), 'r': r, 'p': p_r})
    return pd.DataFrame(rows).assign(region=region, group=group)
