import argparse
import os
import re
import sys
from glob import glob

import numpy as np
import pandas as pd

from extract_roi_values import REGION_DICT, IGNORE_SUBS, mask_path, split_region_hemi

''' Generalized psychophysiological interaction (gPPI), ROI to ROI: does
coupling between a striatal seed and auditory cortex depend on the task
event, and does that differ between learners and natives or change with
learning?

One model per participant x learning stage x seed, fit to the mean time
series of every target ROI. The task part of the design is the univariate
model's, unchanged (univariate_glm.py --event_type=feedback, the
`scrubbed-splitfb` design): sound, resp, fb_correct, fb_wrong, fb_noresp,
cosine drift, motion + aCompCor confounds, FD/DVARS scrubbing, and the
same early/middle/final run pairs. Added to it:
  - seed: the seed ROI's mean time series, cleaned of drift and confounds
    (not of task), z-scored within run
  - ppi_<condition>: seed x that condition's HRF-convolved regressor, one
    term per task condition (generalized PPI, McLaren et al. 2012), except
    resp: it follows the sound by <1 s, so seed x resp is nearly a copy of
    seed x sound (VIF in the hundreds on simulated runs with this timing).
    resp stays in the model as a task regressor; ppi_sound covers the
    sound + response period.
The PPI terms are built from the BOLD seed without deconvolution: at TR = 3 s
deconvolution adds noise and assumptions for little gain in an event-related
design (O'Reilly et al. 2012; Di et al. 2017 compare the two).

Contrasts (ids without separators, as in univariate_glm.py):
  seed                -- task-independent coupling (sanity check)
  ppisound            -- coupling during the sound
  ppifbcorrect        -- coupling during correct feedback, both groups
  ppifbwrong
  ppifbcorrectvswrong -- learners only (natives have ~1 wrong trial per run)
Feedback falls one TR after the sound, so ppisound and the ppifb terms
share some variance; the QC table records each PPI term's VIF.

Target time series are z-scored within run, so effects are in SD of the
target per SD of the seed. Runs within a stage are combined by fixed effects
(nilearn's averaging of run contrasts, as in the univariate model).

Per participant:  python ppi_roi.py --sub=FLT02 --bidsroot=... --fmriprep_dir=...
  -> derivatives/nilearn/<fmriprep tag>/scrubbed-splitfb/ppi_roi/sub-<id>_ppi.tsv (+ _ppi-qc.tsv)
Then once:        python ppi_roi.py --gather --bidsroot=...
  -> group_space-T1w_fwhm-0.00/roi-values_ppi-seed-<seed>/roi-values_network-<net>_contrast-<c>.tsv,
     the extract_roi_values.py schema, so group_stats.load_roi_values reads it.
'''

STAGE_RUNS = {'early': [0, 1], 'middle': [2, 3], 'final': [4, 5]}   # as univariate_glm.py's grouped mode
SEEDS = {'aCAU': ('aCAU-lh', 'aCAU-rh'), 'aPUT': ('aPUT-lh', 'aPUT-rh')}   # bilateral (union of hemispheres)
SEED_NETWORK = 'tian-S2'
NO_PPI_TERM = ('resp',)   # task regressor only; see above
# contrast id -> {design column: weight}
CONTRASTS = {'seed': {'seed': 1},
             'ppisound': {'ppi_sound': 1},
             'ppifbcorrect': {'ppi_fb_correct': 1},
             'ppifbwrong': {'ppi_fb_wrong': 1},
             'ppifbcorrectvswrong': {'ppi_fb_correct': 1, 'ppi_fb_wrong': -1}}
VARIANT_TAG = 'scrubbed-splitfb'   # the univariate model this one extends


def feedback_events(run_events):
    ''' univariate_glm.py's event_type=feedback relabeling: drop events that
    didn't happen, collapse sound_*/resp_* to 'sound'/'resp', keep the
    feedback outcomes split. Keep in sync with prep_models_and_args. '''
    events = run_events.dropna(subset=['onset']).reset_index(drop=True).copy()
    is_nuisance = events['trial_type'].str.startswith(('sound_', 'resp_'))
    events.loc[is_nuisance, 'trial_type'] = events.loc[is_nuisance, 'trial_type'].str.split('_', expand=True)[0]
    return events


def task_design(model, events, n_scans):
    ''' the run's task + drift design, built as FirstLevelModel.fit builds it '''
    from nilearn.glm.first_level import make_first_level_design_matrix
    start_time = model.slice_time_ref * model.t_r
    frame_times = np.linspace(start_time, (n_scans - 1) * model.t_r + start_time, n_scans)
    return make_first_level_design_matrix(frame_times, events, hrf_model=model.hrf_model,
                                          drift_model=model.drift_model, high_pass=model.high_pass,
                                          drift_order=model.drift_order, fir_delays=model.fir_delays,
                                          min_onset=model.min_onset)


def condition_columns(design):
    return [c for c in design.columns if not c.startswith('drift_') and c != 'constant']


def zscore(x):
    x = np.asarray(x, dtype=float)
    return (x - x.mean(axis=0)) / x.std(axis=0)


def ppi_design(task, confounds, seed_ts):
    ''' gPPI design for one run, retained volumes only (task, confounds and
    seed_ts already restricted to them). The seed is cleaned of drift and
    confounds before it's multiplied, so the interaction terms don't carry
    motion; task stays in it, since the task regressors are in the model. '''
    task, confounds = task.reset_index(drop=True), confounds.reset_index(drop=True)
    seed_ts = np.asarray(seed_ts, dtype=float)
    conditions = condition_columns(task)
    nuisance =pd.concat([task.drop(columns=conditions), confounds], axis=1).values
    seed = seed_ts - nuisance @ np.linalg.lstsq(nuisance, seed_ts, rcond=None)[0]
    seed = zscore(seed)
    ppi = pd.DataFrame({f'ppi_{c}': seed * task[c].values for c in conditions if c not in NO_PPI_TERM},
                       index=task.index)
    return pd.concat([task, confounds, pd.Series(seed, index=task.index, name='seed'), ppi], axis=1)


def vif(design, columns):
    ''' variance inflation factor of each named column, from the inverse
    correlation matrix of the non-constant columns '''
    X = design.loc[:, design.std() > 0]
    inv = np.linalg.pinv(np.corrcoef(X.values, rowvar=False))
    return {c: inv[X.columns.get_loc(c), X.columns.get_loc(c)] for c in columns if c in X.columns}


def contrast_vector(design, weights):
    ''' None if any column the contrast names is missing from this run's design
    (e.g. no fb_wrong trials), as univariate_glm._estimable_contrasts skips it '''
    if any(col not in design.columns for col in weights):
        return None
    return np.array([weights.get(col, 0.0) for col in design.columns])


def fit_stage(runs, contrasts=CONTRASTS, noise_model='ar1'):
    ''' runs: list of (design DataFrame, Y array n_retained x n_targets).
    Returns {contrast_id: (effect, t)} per target, combining runs by fixed
    effects; a contrast missing from any run is skipped for the stage. '''
    from nilearn.glm.first_level import run_glm
    try:
        from nilearn.glm.contrasts import compute_fixed_effect_contrast
    except ImportError:   # older nilearn
        from nilearn.glm.contrasts import _compute_fixed_effect_contrast as compute_fixed_effect_contrast

    fits = [run_glm(Y, design.values, noise_model=noise_model) for design, Y in runs]
    out = {}
    for contrast_id, weights in contrasts.items():
        con_vals = [contrast_vector(design, weights) for design, _ in runs]
        if any(c is None for c in con_vals):
            continue
        con = compute_fixed_effect_contrast([f[0] for f in fits], [f[1] for f in fits], con_vals, 't')
        out[contrast_id] = (np.ravel(con.effect_size()), np.ravel(con.stat()))
    return out


def roi_timeseries(img, roi_masks, mask_cache):
    ''' mean time series per ROI (roi name -> list of mask paths, unioned).
    Masks are resampled to the BOLD grid (nearest neighbour), as
    extract_roi_values.py resamples them to the stat maps; voxels with no
    signal (zero variance, e.g. outside the BOLD field of view) are dropped. '''
    from nilearn.image import resample_to_img, index_img
    data = img.get_fdata(dtype=np.float32)
    grid = (img.shape[:3], tuple(np.round(img.affine, 4).ravel()))
    ts, n_voxels = {}, {}
    for name, paths in roi_masks.items():
        key = (grid, name)
        if key not in mask_cache:
            ref = index_img(img, 0)
            union = np.zeros(img.shape[:3], dtype=bool)
            for path in paths:
                resampled = resample_to_img(path, ref, interpolation='nearest',
                                            force_resample=True, copy_header=True)
                union |= np.asarray(resampled.dataobj) > 0
            mask_cache[key] = union
        voxels = data[mask_cache[key]]
        voxels = voxels[voxels.std(axis=1) > 0]
        n_voxels[name] = len(voxels)
        ts[name] = voxels.mean(axis=0) if len(voxels) else np.full(img.shape[3], np.nan)
    return ts, n_voxels


def load_subject(sub_id, task, space, bidsroot, fmriprep_dir):
    ''' run images, feedback-relabeled events, confounds and sample masks, as
    univariate_glm.prep_models_and_args loads them (same scrubbing thresholds;
    None sample masks normalized to keep-all index arrays) '''
    from nilearn.glm.first_level import first_level_from_bids
    from nilearn.interfaces.fmriprep import load_confounds_strategy
    models, run_imgs, events, _ = first_level_from_bids(bidsroot, task, space, [sub_id], smoothing_fwhm=0.0,
                                                        derivatives_folder=fmriprep_dir, minimize_memory=False)
    model, run_imgs, events = models[0], run_imgs[0], [feedback_events(e) for e in events[0]]
    confounds, sample_masks = load_confounds_strategy(img_files=run_imgs, denoise_strategy='scrubbing',
                                                      fd_threshold=0.9, std_dvars_threshold=1.5)
    sample_masks = [np.arange(len(c)) if m is None else np.asarray(m) for c, m in zip(confounds, sample_masks)]
    return model, run_imgs, events, confounds, sample_masks


def run_subject(sub_id, args):
    import nibabel as nib
    model, run_imgs, events, confounds, sample_masks = load_subject(sub_id, args.task, args.space,
                                                                    args.bidsroot, args.fmriprep_dir)
    masks_dir = os.path.join(args.bidsroot, 'derivatives', 'nilearn', 'masks')
    sub_label = f'sub-{sub_id}'

    seed_masks = {s: [mask_path(masks_dir, sub_label, args.space, SEED_NETWORK, rh) for rh in SEEDS[s]]
                  for s in args.seeds}
    seed_rois = {rh for s in args.seeds for rh in SEEDS[s]}
    targets = [(network, rh) for network in args.target_networks for rh in REGION_DICT[network]
               if rh not in seed_rois]
    target_masks = {rh: [mask_path(masks_dir, sub_label, args.space, network, rh)] for network, rh in targets}
    missing = [p for paths in list(seed_masks.values()) + list(target_masks.values()) for p in paths
               if not os.path.exists(p)]
    if missing:
        raise FileNotFoundError(f'missing masks: {missing}')
    if len(run_imgs) != 6:
        print(f'NOTE: {sub_id} has {len(run_imgs)} runs; stages use whatever runs exist')

    # every ROI's time series, one run at a time (one 4D image in memory at once)
    mask_cache, per_run = {}, []
    for rx, img_path in enumerate(run_imgs):
        img = nib.load(img_path)
        ts, n_vox = roi_timeseries(img, {**seed_masks, **target_masks}, mask_cache)
        task = task_design(model, events[rx], img.shape[3])
        per_run.append({'ts': ts, 'n_voxels': n_vox, 'task': task,
                        'confounds': confounds[rx].fillna(0), 'keep': sample_masks[rx],
                        'events': events[rx]})
        del img
        print(f'run {rx + 1}: {task.shape[0]} volumes, {len(sample_masks[rx])} retained', flush=True)

    target_names = [rh for _, rh in targets]
    target_network = dict((rh, network) for network, rh in targets)
    rows, qc = [], []
    for seed_name in args.seeds:
        for stage, run_idx in STAGE_RUNS.items():
            runs = []
            for rx in [r for r in run_idx if r < len(per_run)]:
                r = per_run[rx]
                keep = r['keep']
                task = r['task'].iloc[keep].reset_index(drop=True)
                conf = r['confounds'].iloc[keep].reset_index(drop=True)
                design = ppi_design(task, conf, r['ts'][seed_name][keep])
                Y = zscore(np.column_stack([r['ts'][t][keep] for t in target_names]))
                runs.append((design, Y))
                ppi_cols = [c for c in design.columns if c.startswith('ppi_')]
                counts = r['events'].trial_type.value_counts()
                qc.append({'participant_id': sub_label, 'seed': seed_name, 'learning_stage': stage,
                           'run': rx + 1, 'n_volumes': len(r['task']), 'n_retained': len(keep),
                           'n_regressors': design.shape[1],
                           'residual_dof': len(keep) - np.linalg.matrix_rank(design.values),
                           'seed_n_voxels': r['n_voxels'][seed_name],
                           **{f'n_{c}': counts.get(c, 0) for c in ('fb_correct', 'fb_wrong', 'fb_noresp')},
                           **{f'vif_{c}': v for c, v in vif(design, ppi_cols).items()}})
            if not runs:
                continue
            for contrast_id, (effect, t) in fit_stage(runs).items():
                for name, e, tv in zip(target_names, effect, t):
                    region, hemisphere = split_region_hemi(name, target_network[name])
                    rows.append({'participant_id': sub_label, 'seed': seed_name, 'contrast': contrast_id,
                                 'network': target_network[name], 'region_hemi': name, 'region': region,
                                 'hemisphere': hemisphere, 'learning_stage': stage, 'beta': e, 't': tv})
            print(f'{seed_name} {stage}: fit {len(runs)} runs', flush=True)

    out_dir = os.path.join(args.bidsroot, 'derivatives', 'nilearn', fmriprep_tag(args.fmriprep_dir),
                           VARIANT_TAG, 'ppi_roi')
    os.makedirs(out_dir, exist_ok=True)
    pd.DataFrame(rows).to_csv(os.path.join(out_dir, f'{sub_label}_ppi.tsv'), sep='\t', index=False)
    pd.DataFrame(qc).to_csv(os.path.join(out_dir, f'{sub_label}_ppi-qc.tsv'), sep='\t', index=False)
    print(f'saved {len(rows)} rows to {out_dir}')


def gather(args):
    ''' per-subject outputs -> one extract_roi_values.py-style table per seed x network x contrast '''
    nilearn_dir = os.path.join(args.bidsroot, 'derivatives', 'nilearn')
    in_dir = os.path.join(nilearn_dir, fmriprep_tag(args.fmriprep_dir), VARIANT_TAG, 'ppi_roi')
    participants = pd.read_csv(os.path.join(args.bidsroot, 'participants.tsv'), sep='\t')
    participants = participants[~participants.participant_id.isin(IGNORE_SUBS)]
    group_of = participants.set_index('participant_id').group

    df = pd.concat([pd.read_csv(f, sep='\t') for f in sorted(glob(os.path.join(in_dir, 'sub-*_ppi.tsv')))],
                   ignore_index=True)
    df = df[df.participant_id.isin(group_of.index)].assign(participant_group=lambda d: d.participant_id.map(group_of))
    qc = pd.concat([pd.read_csv(f, sep='\t') for f in sorted(glob(os.path.join(in_dir, 'sub-*_ppi-qc.tsv')))],
                   ignore_index=True)
    qc = qc[qc.participant_id.isin(group_of.index)].assign(participant_group=lambda d: d.participant_id.map(group_of))
    print(df.groupby(['seed', 'participant_group']).participant_id.nunique().to_string())

    group_dir = os.path.join(nilearn_dir, 'group_space-%s_fwhm-0.00' % args.space)
    columns = ['participant_id', 'participant_group', 'region', 'beta', 'region_hemi', 'hemisphere',
               'learning_stage']
    for seed_name, by_seed in df.groupby('seed'):
        out_dir = os.path.join(group_dir, f'roi-values_ppi-seed-{seed_name}')
        os.makedirs(out_dir, exist_ok=True)
        for (network, contrast), table in by_seed.groupby(['network', 'contrast']):
            table[columns].to_csv(os.path.join(out_dir, f'roi-values_network-{network}_contrast-{contrast}.tsv'),
                                  sep='\t', index=False)
        qc[qc.seed == seed_name].to_csv(os.path.join(out_dir, 'ppi-qc.tsv'), sep='\t', index=False)
        print(f'wrote {out_dir}')


def fmriprep_tag(fmriprep_dir):
    ''' as univariate_glm._fmriprep_tag '''
    match = re.search(r'fmriprep-[\d.]+', os.path.basename(os.path.normpath(fmriprep_dir)))
    return match.group(0) if match else os.path.basename(os.path.normpath(fmriprep_dir))


def main(argv=None):
    parser = argparse.ArgumentParser(description='ROI-to-ROI generalized PPI per learning stage')
    parser.add_argument('--sub', type=str, help='participant id without sub- (e.g. FLT02)')
    parser.add_argument('--gather', action='store_true', help='combine per-subject outputs for group_stats')
    parser.add_argument('--bidsroot', required=True, type=str)
    parser.add_argument('--fmriprep_dir', type=str,
                        help='default: <bidsroot>/derivatives/denoised_fmriprep-25.2.5')
    parser.add_argument('--task', default='tonecat', type=str)
    parser.add_argument('--space', default='T1w', type=str)
    parser.add_argument('--seeds', nargs='+', default=list(SEEDS), choices=list(SEEDS))
    parser.add_argument('--target_networks', nargs='+', default=['auditory'], choices=list(REGION_DICT))
    args = parser.parse_args(argv)
    args.fmriprep_dir = args.fmriprep_dir or os.path.join(args.bidsroot, 'derivatives',
                                                          'denoised_fmriprep-25.2.5')
    if args.gather:
        gather(args)
    elif args.sub:
        run_subject(args.sub, args)
    else:
        parser.error('give --sub or --gather')
    return 0


if __name__ == '__main__':
    sys.exit(main())
