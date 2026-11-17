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
only when called, so the dimensionality notebook doesn't need it.
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


def load_decoding(dec_dir, participants, held_out='mean'):
    ''' FLT2's decoding_generalization.py output (sub-<id>_decoding.tsv) in the
    extract_roi_values.py schema: accuracy minus chance in `beta`, the analysis in
    `contrast` ('tone_across_talker', 'talker_across_tone'; '<analysis>_held-out-run' for
    the conservative scheme), stages early/middle/final (the whole-session rows are
    dropped). held_out='mean' keeps the mean over folds. '''
    files = sorted(glob(os.path.join(dec_dir, 'sub-*_decoding.tsv')))
    if not files:
        raise FileNotFoundError(f'no sub-*_decoding.tsv files in {dec_dir}')
    dec = pd.concat([pd.read_csv(f, sep='\t') for f in files], ignore_index=True)
    dec = dec[(dec.held_out == held_out) & dec.stage.isin(STAGE_ORDER)]
    split = dec.roi.map(split_region_hemi)
    dec = dec.assign(region_hemi=dec.roi, region=split.str[0], hemisphere=split.str[1], network=split.str[2],
                     learning_stage=dec.stage, beta=dec.accuracy - dec.chance,
                     contrast=np.where(dec.scheme == 'same-runs', dec.analysis, dec.analysis + '_held-out-run'))
    return dec.drop(columns='participant_group', errors='ignore').merge(participants, on='participant_id', how='inner')[
        ['participant_id', 'participant_group', 'region_hemi', 'region', 'hemisphere', 'network',
         'learning_stage', 'contrast', 'beta', 'n_voxels']]


def rdm_vectors(rdms):
    ''' {(participant_id, stage, region_hemi): dissimilarity vector}, checking one stimulus order '''
    vectors, stimuli = {}, None
    for (participant_id, stage), sub_rdms in rdms.items():
        order = list(sub_rdms.pattern_descriptors['stimulus'])
        assert stimuli is None or order == stimuli, f'{participant_id}: stimulus order differs'
        stimuli = order
        for i, region_hemi in enumerate(sub_rdms.rdm_descriptors['ROI']):
            vectors[(participant_id, stage, str(region_hemi))] = sub_rdms[i].get_vectors()[0]
    return vectors


def _rdm_row(participant_id, group, region_hemi, label, contrast, value):
    region, hemisphere, network = split_region_hemi(region_hemi)
    return {'participant_id': participant_id, 'participant_group': group, 'region_hemi': region_hemi,
            'region': region, 'hemisphere': hemisphere, 'network': network,
            'learning_stage': label, 'contrast': contrast, 'beta': value}


def stage_stability(rdms, participants, pairs=(('early', 'middle'), ('middle', 'final'), ('early', 'final'))):
    ''' within each participant and ROI, the Pearson correlation between two stages'
    cross-validated RDMs: how much the representational geometry changes. The stage pair
    (e.g. 'early-final') goes in `learning_stage`, so group_difference_tests(..., by_stage=True)
    tests each pair. Noisier RDMs correlate less, so check against mean_distance. '''
    group_of = participants.set_index('participant_id').participant_group
    vectors = rdm_vectors(rdms)
    rows = []
    for (participant_id, stage, region_hemi), v in vectors.items():
        for a, b in pairs:
            if stage == a and (participant_id, b, region_hemi) in vectors:
                r = np.corrcoef(v, vectors[(participant_id, b, region_hemi)])[0, 1]
                rows.append(_rdm_row(participant_id, group_of[participant_id], region_hemi,
                                     f'{a}-{b}', 'stage_stability', r))
    return pd.DataFrame(rows)


def native_likeness(rdms, participants, natives='Mandarin'):
    ''' per participant x stage x ROI, the Pearson correlation of the participant's RDM
    with the natives' template: their mean RDM over all stages (natives' geometry shouldn't
    change), each native left out of their own template. Control, 'peer_likeness': the
    correlation with the mean RDM of the other members of the participant's own group at the
    same stage. If learners' native-likeness rises only as much as their peer-likeness, the
    rise is reliability (cleaner RDMs correlate with everything more), not convergence on the
    native geometry. '''
    group_of = participants.set_index('participant_id').participant_group
    vectors = rdm_vectors(rdms)
    native_mean = {}
    for (p, stage, roi), v in vectors.items():
        if group_of[p] == natives:
            native_mean.setdefault(roi, {}).setdefault(p, []).append(v)
    native_mean = {roi: {p: np.mean(vs, axis=0) for p, vs in by_p.items()} for roi, by_p in native_mean.items()}

    rows = []
    for (participant_id, stage, roi), v in vectors.items():
        group = group_of[participant_id]
        natives_others = [m for p, m in native_mean.get(roi, {}).items() if p != participant_id]
        peers = [w for (p, st, r), w in vectors.items()
                 if r == roi and st == stage and p != participant_id and group_of[p] == group]
        if len(natives_others) >= 2:
            rows.append(_rdm_row(participant_id, group, roi, stage, 'native_likeness',
                                 np.corrcoef(v, np.mean(natives_others, axis=0))[0, 1]))
        if len(peers) >= 2:
            rows.append(_rdm_row(participant_id, group, roi, stage, 'peer_likeness',
                                 np.corrcoef(v, np.mean(peers, axis=0))[0, 1]))
    return pd.DataFrame(rows)


def mean_distance(rdms, participants):
    ''' mean cross-validated distance per participant x stage x ROI: overall
    discriminability, for checking that stability or native-likeness differences
    aren't just differences in signal strength '''
    group_of = participants.set_index('participant_id').participant_group
    return pd.DataFrame([_rdm_row(p, group_of[p], roi, stage, 'mean_distance', v.mean())
                         for (p, stage, roi), v in rdm_vectors(rdms).items()])

