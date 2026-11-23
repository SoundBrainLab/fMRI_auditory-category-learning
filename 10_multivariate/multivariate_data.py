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

# composites: primary = core/belt; association = superior temporal; IFG = inferior frontal.
# FLT2's notebook grouped STG and IFG as one 'Association' composite; IFG is split out because
# pars opercularis / triangularis aren't auditory association cortex (all three stay in the
# planned FDR family, rsa_two_group.ipynb section 9)
COMPOSITES = {'Primary': ('HG', 'PT', 'PP'),
              'Association': ('STGa', 'STGp'),
              'IFG': ('ParsOp', 'ParsTri')}

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
    model distances; Nili et al. 2014). Pairs that are NaN in a data RDM (rsa_roi.py
    --trials=correct, stimuli without enough correct trials) are left out of that fit. '''
    from rsatoolbox.rdm import compare
    group_of = participants.set_index('participant_id').participant_group
    rows = []
    for (participant_id, stage), sub_rdms in rdms.items():
        models = model_rdms(sub_rdms.pattern_descriptors['stimulus'])
        for i, region_hemi in enumerate(sub_rdms.rdm_descriptors['ROI']):
            region, hemisphere, network = split_region_hemi(str(region_hemi))
            data = sub_rdms[i].get_vectors()
            for factor, model in models.items():
                model_vector = model.get_vectors().copy()
                model_vector[np.isnan(data)] = np.nan
                fit = compare(model_vector, data, method=method)[0, 0]
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


def load_outcome_decoding(dec_dir, participants, center='stimulus'):
    ''' FLT2's decoding_outcome.py output (sub-<id>_decoding-outcome_center-<center>.tsv):
    correct vs. wrong, balanced accuracy minus chance (0.5) in `beta`, contrast
    'outcome_center-<center>', with the trial counts (n_correct, n_wrong). Stages
    early/middle/final; stages without enough trials of each outcome are NaN and dropped. '''
    files = sorted(glob(os.path.join(dec_dir, f'sub-*_decoding-outcome_center-{center}.tsv')))
    if not files:
        raise FileNotFoundError(f'no sub-*_decoding-outcome_center-{center}.tsv files in {dec_dir}')
    dec = pd.concat([pd.read_csv(f, sep='\t') for f in files], ignore_index=True)
    dec = dec[dec.stage.isin(STAGE_ORDER)].dropna(subset=['balanced_accuracy'])
    split = dec.roi.map(split_region_hemi)
    dec = dec.assign(region_hemi=dec.roi, region=split.str[0], hemisphere=split.str[1], network=split.str[2],
                     learning_stage=dec.stage, beta=dec.balanced_accuracy - dec.chance,
                     contrast=f'outcome_center-{center}')
    return dec.merge(participants, on='participant_id', how='inner')[
        ['participant_id', 'participant_group', 'region_hemi', 'region', 'hemisphere', 'network',
         'learning_stage', 'contrast', 'beta', 'n_correct', 'n_wrong']]


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
    return pd.DataFrame([_rdm_row(p, group_of[p], roi, stage, 'mean_distance', np.nanmean(v))
                         for (p, stage, roi), v in rdm_vectors(rdms).items()])



# ---- behavioral model RDMs: does a learner's own accuracy/confusion explain their RDM? ----

# response keys -> the tone they label (04_behavior/create_behav_confusion_matrices.py renumbers
# keys 7, 6, 1, 2 as tones 1-4); checked against each trial's correct key in load_behavior
KEY_TO_TONE = {'7': 'di1', '6': 'di2', '1': 'di3', '2': 'di4'}
STAGE_OF_RUN = {1: 'early', 2: 'early', 3: 'middle', 4: 'middle', 5: 'final', 6: 'final'}


def load_behavior(bidsroot, participants, task='tonecat'):
    ''' one row per tonecat trial from the BIDS events files: stimulus ('di1_aN'), the tone
    the response labels (None if no response), the correct key's tone, feedback
    ('fb_correct' / 'fb_wrong' / 'fb_noresp'), and the sound and feedback onsets.

    Events label sounds 'sound_di1-aN'; GLMsingle (event_type='stimulus') turns the hyphen
    into '_', so the RDMs say 'di1_aN' -- converted here to match.
    The events files hold separate sound, resp and fb rows (04_behavior/convert_behav_to_bids.py);
    sorted by onset, each trial is sound -> resp -> fb, so resp and fb rows are assigned to
    the preceding sound. Missed responses have no onset (sorted last) and are skipped; the
    trial's fb_noresp row still marks them. '''
    group_of = participants.set_index('participant_id').participant_group
    rows = []
    for participant_id in participants.participant_id:
        pattern = os.path.join(bidsroot, participant_id, 'func', f'{participant_id}_task-{task}_run-*_events.tsv')
        for fpath in sorted(glob(pattern)):
            run = int(os.path.basename(fpath).split('_run-')[1][:2])
            events = pd.read_csv(fpath, sep='\t', index_col=0).sort_values('onset', na_position='last')
            trial = None
            for onset, trial_type, correct_key in zip(events.onset, events.trial_type.astype(str),
                                                      events.get('correct_key', pd.Series(np.nan, index=events.index))):
                if trial_type.startswith('sound_'):
                    if trial is not None:
                        rows.append(trial)
                    trial = {'participant_id': participant_id, 'participant_group': group_of[participant_id],
                             'run': run, 'learning_stage': STAGE_OF_RUN.get(run), 'stimulus': trial_type[len('sound_'):].replace('-', '_'),
                             'sound_onset': onset, 'response_tone': None, 'correct_tone': None,
                             'feedback': None, 'fb_onset': np.nan}
                elif trial is None or pd.isna(onset):
                    continue
                elif trial_type.startswith('resp_'):
                    key = trial_type[len('resp_'):].removesuffix('.0')
                    trial['response_tone'] = KEY_TO_TONE.get(key)
                    if pd.notna(correct_key):
                        trial['correct_tone'] = KEY_TO_TONE.get(str(int(correct_key)))
                elif trial_type.startswith('fb_'):
                    trial['feedback'], trial['fb_onset'] = trial_type, onset
            if trial is not None:
                rows.append(trial)
    trials = pd.DataFrame(rows)
    trials['tone'] = trials.stimulus.str.split('_').str[0]
    trials['correct'] = trials.feedback.map({'fb_correct': 1.0, 'fb_wrong': 0.0})  # fb_noresp -> NaN
    trials['fb_delay'] = trials.fb_onset - trials.sound_onset
    return trials


def stimulus_behavior(trials, noresp='drop'):
    ''' per participant x stage x stimulus: accuracy and the proportions of responses
    labelling each tone. noresp='drop' leaves missed responses out; 'wrong' counts them
    as incorrect (and as no tone). '''
    trials = trials.copy()
    if noresp == 'wrong':
        trials['correct'] = trials.correct.fillna(0.0)
    else:
        trials = trials[trials.feedback != 'fb_noresp']
    tones = sorted(KEY_TO_TONE.values())
    for tone in tones:
        trials[f'resp_{tone}'] = (trials.response_tone == tone).astype(float)
    keys = ['participant_id', 'participant_group', 'learning_stage', 'stimulus']
    return trials.groupby(keys, as_index=False).agg(
        n=('correct', 'size'), accuracy=('correct', 'mean'),
        **{f'resp_{tone}': (f'resp_{tone}', 'mean') for tone in tones})


def behavior_models(stim_behavior, stimuli, groups=('non-Mandarin',)):
    ''' {(participant_id, stage): {model name: dissimilarity vector}} over `stimuli` (the RDMs'
    stimulus order), for participants in `groups`:
      accuracy_own     |acc_s - acc_t| from the participant's own accuracy that stage
      confusion_own    Euclidean distance between the stimuli's response-proportion vectors
      accuracy_group, confusion_group   the same from the mean of the other members of the
                       participant's group that stage (shared stimulus difficulty)
    A model with no variance (e.g. a participant at ceiling) is left out. '''
    from scipy.spatial.distance import pdist
    resp_cols = [c for c in stim_behavior.columns if c.startswith('resp_')]
    table = stim_behavior.set_index(['participant_id', 'learning_stage', 'stimulus'])
    models = {}
    for (participant_id, stage), sub in stim_behavior.groupby(['participant_id', 'learning_stage']):
        group = sub.participant_group.iloc[0]
        if group not in groups:
            continue
        peers = stim_behavior[(stim_behavior.participant_group == group) & (stim_behavior.learning_stage == stage)
                              & (stim_behavior.participant_id != participant_id)]
        sources = {'own': table.loc[(participant_id, stage)],
                   'group': peers.groupby('stimulus')[['accuracy'] + resp_cols].mean()}
        out = {}
        for source, values in sources.items():
            values = values.reindex(stimuli)
            if values[['accuracy'] + resp_cols].isna().any().any():
                continue   # a stimulus with no (responded) trials
            for name, matrix in [('accuracy', values[['accuracy']].values), ('confusion', values[resp_cols].values)]:
                vector = pdist(matrix, 'euclidean')
                if np.ptp(vector) > 0:
                    out[f'{name}_{source}'] = vector
        models[(participant_id, stage)] = out
    return models


def behavior_model_fits(rdms, participants, models, method='tau-a'):
    ''' each participant x stage x ROI RDM compared with that participant's behavioral
    models (contrast '<model>_fit'), as model_fits does for the tone/talker models '''
    from rsatoolbox.rdm import RDMs, compare
    group_of = participants.set_index('participant_id').participant_group
    rows = []
    for (participant_id, stage, region_hemi), v in rdm_vectors(rdms).items():
        data = RDMs(v[None], dissimilarity_measure='crossnobis')
        for name, m in models.get((participant_id, stage), {}).items():
            fit = compare(RDMs(m[None], dissimilarity_measure='euclidean'), data, method=method)[0, 0]
            rows.append(_rdm_row(participant_id, group_of[participant_id], region_hemi, stage, f'{name}_fit', fit))
    return pd.DataFrame(rows)


def behavior_model_regression(rdms, participants, models, behavioral=('accuracy_own',)):
    ''' unique contributions: per participant x stage x ROI, the rank-transformed RDM regressed on
    the rank-transformed tone and talker models plus the given behavioral models (all
    z-scored; ordinary least squares). Coefficients as contrast '<model>_beta'. Tone,
    confusion and accuracy are correlated, so a simple fit can't separate them. '''
    from scipy.stats import rankdata
    group_of = participants.set_index('participant_id').participant_group
    vectors = rdm_vectors(rdms)
    stimuli = next(iter(rdms.values())).pattern_descriptors['stimulus']
    fixed = {f: m.get_vectors()[0] for f, m in model_rdms(stimuli).items()}
    z = lambda x: (rankdata(x) - rankdata(x).mean()) / rankdata(x).std()
    rows = []
    for (participant_id, stage, region_hemi), v in vectors.items():
        sub_models = models.get((participant_id, stage), {})
        if not all(b in sub_models for b in behavioral):
            continue
        predictors = {**fixed, **{b: sub_models[b] for b in behavioral}}
        X = np.column_stack([np.ones(len(v))] + [z(p) for p in predictors.values()])
        coef = np.linalg.lstsq(X, z(v), rcond=None)[0][1:]
        for name, c in zip(predictors, coef):
            rows.append(_rdm_row(participant_id, group_of[participant_id], region_hemi, stage, f'{name}_beta', c))
    return pd.DataFrame(rows)
