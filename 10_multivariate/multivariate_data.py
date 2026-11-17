import os
import sys
from glob import glob
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / '06_univariate'))
from group_stats import GROUPS, STAGE_ORDER

''' Loaders for the per-subject multivariate outputs FLT2 writes (GLMsingle
single-trial betas on fMRIPrep 25.2.5, T1w; see FLT2's dimensionality_roi.py
and rsa_roi.py).
Each loader returns a long table in the same schema as
06_univariate/extract_roi_values.py -- participant_id, participant_group,
region, hemisphere, learning_stage, network, contrast, beta -- so the
two-group functions in group_stats.py (bilateral_mean, subject_trends,
group_by_stage_anova, trend_difference_tests, trend_group_difference, ...)
run on it unchanged. The measure goes in `beta` and its name in `contrast`
(e.g. contrast='PR'), since those functions read the value from `beta`.

The per-subject processing stays in FLT2; only group-level analysis lives
here (TWO_GROUP_MANUSCRIPT_PLAN.md R5). The RSA functions import rsatoolbox
only when called, so the dimensionality notebook doesn't need it. A decoding
loader gets added once FLT2 writes per-stage decoding outputs.
'''

DERIV_DIR = '/ix1/bchandrasekaran/krs228/data/FLT/data_denoised/derivatives'
GLMSINGLE_DIRNAME = 'glmsingle_fmriprep-25.2.5_space-T1w'
IGNORE_SUBS = ['sub-FLT27',  # did not participate
               'sub-FLT07']  # bad QA 11/14/23 (as in extract_roi_values.py)

# a-priori composites (FLT2 notebook): primary = core/belt, association = the rest
COMPOSITES = {'Primary': ('HG', 'PT', 'PP'),
              'Association': ('STGa', 'STGp', 'ParsOp', 'ParsTri')}

# stimulus labels are 'di<tone>_<talker>' (e.g. 'di3_bN'): 4 tones x 4 talkers
STIMULUS_FACTORS = {'tone': lambda stim: stim.split('_')[0], 'talker': lambda stim: stim.split('_')[1]}


# networks for FLT2's masked ROIs that aren't auditory cortex: the subcortical auditory atlas
# (`subcort-aud`) and the whole caudate/putamen labels in the `dseg` atlas
OTHER_NETWORKS = {'IC': 'subcort-aud', 'MGN': 'subcort-aud', 'CN': 'subcort-aud', 'SOC': 'subcort-aud',
                  'Caud': 'striatum-whole', 'Put': 'striatum-whole'}


def split_region_hemi(region_hemi):
    ''' 'L-HG' -> ('HG', 'L', 'auditory'); 'aCAU-lh' -> ('aCAU', 'lh', 'tian-S2'), as in
    extract_roi_values.py; 'L-IC' -> ('IC', 'L', 'subcort-aud') '''
    if region_hemi.endswith(('-lh', '-rh')):
        return region_hemi[:-3], region_hemi[-2:], 'tian-S2'
    region = region_hemi[2:]
    return region, region_hemi[0], OTHER_NETWORKS.get(region, 'auditory')


def load_participants(bidsroot):
    ''' participants.tsv restricted to the two analysed groups, minus exclusions '''
    participants = pd.read_csv(os.path.join(bidsroot, 'participants.tsv'), sep='\t')
    participants = participants[participants.group.isin(GROUPS)
                                & ~participants.participant_id.isin(IGNORE_SUBS)]
    return participants[['participant_id', 'group']].rename(columns={'group': 'participant_group'})


def load_dimensionality(dim_dir, participants, measure='PR'):
    ''' FLT2's per-stage dimensionality CSVs (stage-<stage>/sub-<id>_dimensionality_<measure>.csv,
    one row per participant, one column per ROI such as 'L-HG') as a long table.
    measure: 'PR', 'Dsheng', or 'ntrials' (trials per stage, for coverage checks) '''
    rows = []
    for stage in STAGE_ORDER:
        for fpath in sorted(glob(os.path.join(dim_dir, f'stage-{stage}', f'sub-*_dimensionality_{measure}.csv'))):
            participant_id = os.path.basename(fpath).split('_')[0]
            values = pd.read_csv(fpath).iloc[0]
            for region_hemi, value in values.items():
                region, hemisphere, network = split_region_hemi(region_hemi)
                rows.append({'participant_id': participant_id, 'learning_stage': stage,
                             'region_hemi': region_hemi, 'region': region,
                             'hemisphere': hemisphere, 'network': network, 'beta': value})
    if not rows:
        raise FileNotFoundError(f'no stage-*/sub-*_dimensionality_{measure}.csv files in {dim_dir}')
    df = pd.DataFrame(rows).merge(participants, on='participant_id', how='inner')
    return df.assign(contrast=measure)


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


def model_rdms(stimuli):
    ''' categorical model RDMs (0 = same category, 1 = different) for each
    factor in STIMULUS_FACTORS, over the given stimulus order '''
    from rsatoolbox.rdm import RDMs
    models = {}
    for factor, label_of in STIMULUS_FACTORS.items():
        labels = np.array([label_of(stim) for stim in stimuli])
        matrix = (labels[:, None] != labels[None, :]).astype(float)
        models[factor] = RDMs(matrix[None], dissimilarity_measure='categorical',
                              rdm_descriptors={'model': [factor]},
                              pattern_descriptors={'stimulus': list(stimuli)})
    return models


def load_subject_rdms(rdm_dir, participants, stages=STAGE_ORDER):
    ''' FLT2's sub-<id>_stage-<stage>_rdms.hdf5 files as {(participant_id, stage): RDMs},
    analysed participants only '''
    from rsatoolbox.rdm import load_rdm
    rdms = {}
    for participant_id in participants.participant_id:
        for stage in stages:
            fpath = os.path.join(rdm_dir, f'{participant_id}_stage-{stage}_rdms.hdf5')
            if os.path.exists(fpath):
                rdms[(participant_id, stage)] = load_rdm(fpath, file_type='hdf5')
    if not rdms:
        raise FileNotFoundError(f'no sub-*_stage-*_rdms.hdf5 files in {rdm_dir}')
    return rdms


def model_fits(rdms, participants, method='tau-a'):
    ''' each subject x stage x ROI RDM compared with each model RDM, as a long table in
    the extract_roi_values.py schema (contrast = '<model>_fit', value in `beta`).
    Kendall's tau-a by default, the usual choice for categorical models (many tied
    model distances; Nili et al. 2014). '''
    from rsatoolbox.rdm import compare
    group_of = participants.set_index('participant_id').participant_group
    rows = []
    for (participant_id, stage), sub_rdms in rdms.items():
        models = model_rdms(sub_rdms.pattern_descriptors['stimulus'])
        for i, region_hemi in enumerate(sub_rdms.rdm_descriptors['ROI']):
            region, hemisphere, network = split_region_hemi(str(region_hemi))
            for factor, model in models.items():
                fit = compare(model, sub_rdms[i], method=method)[0, 0]
                rows.append({'participant_id': participant_id,
                             'participant_group': group_of[participant_id],
                             'learning_stage': stage, 'region_hemi': str(region_hemi),
                             'region': region, 'hemisphere': hemisphere, 'network': network,
                             'contrast': f'{factor}_fit', 'beta': fit})
    return pd.DataFrame(rows)


def group_mean_rdms(rdms, participants):
    ''' {(group, stage, region_hemi): mean dissimilarity vector, stimuli}: for plotting
    and the noise ceiling. Checks every subject has the same stimulus order. '''
    group_of = participants.set_index('participant_id').participant_group
    vectors, stimuli = {}, None
    for (participant_id, stage), sub_rdms in rdms.items():
        order = list(sub_rdms.pattern_descriptors['stimulus'])
        assert stimuli is None or order == stimuli, f'{participant_id}: stimulus order differs'
        stimuli = order
        for i, region_hemi in enumerate(sub_rdms.rdm_descriptors['ROI']):
            key = (group_of[participant_id], stage, str(region_hemi))
            vectors.setdefault(key, {})[participant_id] = sub_rdms[i].get_vectors()[0]
    return vectors, stimuli


def noise_ceiling(rdms, participants, method='tau-a'):
    ''' per group x stage x ROI: how well the group's own mean RDM predicts each
    member's RDM. Upper bound: mean fit to the mean of all members (including the
    subject); lower bound: mean fit to the mean of the others (leave one out). A
    model fit near the lower bound is as good as the data's reliability allows. '''
    from rsatoolbox.rdm import RDMs, compare
    as_rdm = lambda v: RDMs(v[None], dissimilarity_measure='crossnobis')
    vectors, _ = group_mean_rdms(rdms, participants)
    rows = []
    for (group, stage, region_hemi), by_subject in vectors.items():
        if len(by_subject) < 3:
            continue
        stacked = np.vstack(list(by_subject.values()))
        upper, lower = [], []
        for i in range(len(stacked)):
            others = np.delete(stacked, i, axis=0).mean(axis=0)
            upper.append(compare(as_rdm(stacked.mean(axis=0)), as_rdm(stacked[i]), method=method)[0, 0])
            lower.append(compare(as_rdm(others), as_rdm(stacked[i]), method=method)[0, 0])
        region, hemisphere, network = split_region_hemi(region_hemi)
        rows.append({'participant_group': group, 'learning_stage': stage, 'region_hemi': region_hemi,
                     'region': region, 'hemisphere': hemisphere, 'network': network,
                     'n': len(stacked), 'lower': np.mean(lower), 'upper': np.mean(upper)})
    return pd.DataFrame(rows)

