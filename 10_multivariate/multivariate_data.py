import os
import sys
from glob import glob
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / '06_univariate'))
from group_stats import GROUPS, STAGE_ORDER

''' Loaders for the per-subject multivariate outputs FLT2 writes (GLMsingle
single-trial betas on fMRIPrep 25.2.5, T1w; see FLT2's dimensionality_roi.py).
Each loader returns a long table in the same schema as
06_univariate/extract_roi_values.py -- participant_id, participant_group,
region, hemisphere, learning_stage, network, contrast, beta -- so the
two-group functions in group_stats.py (bilateral_mean, subject_trends,
group_by_stage_anova, trend_difference_tests, trend_group_difference, ...)
run on it unchanged. The measure goes in `beta` and its name in `contrast`
(e.g. contrast='PR'), since those functions read the value from `beta`.

The per-subject processing stays in FLT2; only group-level analysis lives
here (TWO_GROUP_MANUSCRIPT_PLAN.md R5). RSA and decoding loaders get added
once FLT2 writes per-stage outputs for them.
'''

DERIV_DIR = '/ix1/bchandrasekaran/krs228/data/FLT/data_denoised/derivatives'
GLMSINGLE_DIRNAME = 'glmsingle_fmriprep-25.2.5_space-T1w'
IGNORE_SUBS = ['sub-FLT27',  # did not participate
               'sub-FLT07']  # bad QA 11/14/23 (as in extract_roi_values.py)

# a-priori composites (FLT2 notebook): primary = core/belt, association = the rest
COMPOSITES = {'Primary': ('HG', 'PT', 'PP'),
              'Association': ('STGa', 'STGp', 'ParsOp', 'ParsTri')}


def load_participants(bidsroot):
    ''' participants.tsv restricted to the two analysed groups, minus exclusions '''
    participants = pd.read_csv(os.path.join(bidsroot, 'participants.tsv'), sep='\t')
    participants = participants[participants.group.isin(GROUPS)
                                & ~participants.participant_id.isin(IGNORE_SUBS)]
    return participants[['participant_id', 'group']].rename(columns={'group': 'participant_group'})


def load_dimensionality(dim_dir, participants, measure='PR', network='auditory'):
    ''' FLT2's per-stage dimensionality CSVs (stage-<stage>/sub-<id>_dimensionality_<measure>.csv,
    one row per participant, one column per ROI such as 'L-HG') as a long table.
    measure: 'PR', 'Dsheng', or 'ntrials' (trials per stage, for coverage checks) '''
    rows = []
    for stage in STAGE_ORDER:
        for fpath in sorted(glob(os.path.join(dim_dir, f'stage-{stage}', f'sub-*_dimensionality_{measure}.csv'))):
            participant_id = os.path.basename(fpath).split('_')[0]
            values = pd.read_csv(fpath).iloc[0]
            for region_hemi, value in values.items():
                rows.append({'participant_id': participant_id, 'learning_stage': stage,
                             'region_hemi': region_hemi, 'region': region_hemi[2:],
                             'hemisphere': region_hemi[0], 'beta': value})
    if not rows:
        raise FileNotFoundError(f'no stage-*/sub-*_dimensionality_{measure}.csv files in {dim_dir}')
    df = pd.DataFrame(rows).merge(participants, on='participant_id', how='inner')
    return df.assign(network=network, contrast=measure)


def load_nvoxels(dim_dir, participants):
    ''' voxels per ROI (whole session; PR is capped at min(n_stimuli, n_voxels)) '''
    rows = []
    for fpath in sorted(glob(os.path.join(dim_dir, 'sub-*_dimensionality_nvoxels.csv'))):
        values = pd.read_csv(fpath).iloc[0]
        rows += [{'participant_id': os.path.basename(fpath).split('_')[0],
                  'region_hemi': region_hemi, 'n_voxels': n} for region_hemi, n in values.items()]
    return pd.DataFrame(rows).merge(participants, on='participant_id', how='inner')


def add_composites(df, composites=COMPOSITES):
    ''' append composite regions: per participant x hemisphere x stage, the
    mean over the member regions (only where every member is present). They
    get network '<network>-composite' so FDR families (per network) keep the
    planned composites apart from the exploratory per-region tests. '''
    keys = ['participant_id', 'participant_group', 'hemisphere', 'learning_stage', 'network', 'contrast']
    out = [df]
    for name, members in composites.items():
        sub = df[df.region.isin(members)]
        complete = sub.groupby(keys).region.transform('nunique') == len(members)
        mean = sub[complete].groupby(keys, as_index=False)['beta'].mean()
        out.append(mean.assign(region=name, region_hemi=mean.hemisphere + '-' + name,
                               network=mean.network + '-composite'))
    return pd.concat(out, ignore_index=True)
