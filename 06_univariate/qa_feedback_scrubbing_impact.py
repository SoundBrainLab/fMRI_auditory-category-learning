import os
import sys
import argparse

import numpy as np
import pandas as pd

'''
Quantify how much motion/DVARS scrubbing actually touches feedback trials,
per learning-stage run-group -- built to test a specific hypothesis raised
while investigating why the reprocessed striatal fb_correct-vs-wrong results
differ substantially from the original manuscript numbers: scrubbing was not
part of the original analysis (aCompCor disabled, no censoring), and feedback
trials are a much thinner, more imbalanced condition than the sound-stimulus
conditions (correct-response trials typically outnumber incorrect ones, and
that imbalance should get worse as learning progresses within a session --
exactly the run-groups ('final' especially) the learning-stage claim rests
on). If scrubbing removes even a handful of already-scarce fb_wrong trials'
volumes, the late-stage contrast could become unstable in a way the original
(unscrubbed) analysis never faced.

This does not change the actual GLM in univariate_glm.py -- it mirrors that
script's confound-loading call exactly (same load_confounds_strategy params)
and cross-references trial onsets against the resulting sample_mask, purely
as a diagnostic to size the risk empirically instead of by assumption.

Usage:
    python qa_feedback_scrubbing_impact.py --sub=FLT02 --task=tonecat \\
        --space=T1w --t_acq=2 --t_r=3 \\
        --bidsroot=/PATH/TO/BIDS/DIR/ --fmriprep_dir=/PATH/TO/FMRIPREP/DIR/ \\
        --out_dir=/PATH/TO/OUTPUT/DIR/
'''

# mirrors univariate_glm.py's nilearn_glm_grouped_runs -- keep in sync if that
# script's run_group_dict changes (not imported directly since univariate_glm.py
# runs its argparse at module level and isn't currently import-safe)
RUN_GROUP_DICT = {'early': [0, 1],
                  'middle': [2, 3],
                  'final': [4, 5]}

# same denoise_strategy/thresholds as univariate_glm.py's prep_models_and_args --
# any change there should be mirrored here so this stays a faithful diagnostic
DENOISE_STRATEGY = 'scrubbing'
FD_THRESHOLD = 0.9
STD_DVARS_THRESHOLD = 1.5


def trials_affected_by_scrubbing(run_events, sample_mask, n_vols, t_r, hrf_window_trs=2):
    ''' for each feedback trial (fb_correct/fb_wrong), flag whether any TR in
    its post-onset HRF window [onset_tr, onset_tr + hrf_window_trs) was
    scrubbed. hrf_window_trs=2 at t_r=3s covers ~0-6s post-onset, the rising
    edge of a canonical HRF -- a trial whose response has no chance to be
    estimated cleanly even if its exact onset TR survived. Returns a
    DataFrame with one row per feedback trial. '''
    retained = set(range(n_vols)) if sample_mask is None else set(int(i) for i in sample_mask)
    scrubbed_trs = set(range(n_vols)) - retained

    fb_events = run_events[run_events['trial_type'].isin(['fb_correct', 'fb_wrong'])]
    rows = []
    for _, row in fb_events.iterrows():
        onset_tr = int(row['onset'] // t_r)
        window = range(onset_tr, onset_tr + hrf_window_trs)
        affected = any(tr in scrubbed_trs for tr in window if tr < n_vols)
        rows.append({
            'trial_type': row['trial_type'],
            'onset': row['onset'],
            'onset_tr': onset_tr,
            'scrubbed': affected,
        })
    return pd.DataFrame(rows)


def main():
    parser = argparse.ArgumentParser(
        description=('Quantify scrubbing impact on feedback trials, by learning-stage '
                     'run-group, for one subject'),
        epilog=('Example: python qa_feedback_scrubbing_impact.py --sub=FLT02 '
                '--task=tonecat --space=T1w --t_acq=2 --t_r=3 '
                '--bidsroot=/PATH/TO/BIDS/DIR/ '
                '--fmriprep_dir=/PATH/TO/FMRIPREP/DIR/ '
                '--out_dir=/PATH/TO/OUTPUT/DIR/'))
    parser.add_argument('--sub', help='participant id', type=str)
    parser.add_argument('--task', help='task id', type=str)
    parser.add_argument('--space', help='space label', type=str, default='T1w')
    parser.add_argument('--t_acq', help='BOLD acquisition time', type=float)
    parser.add_argument('--t_r', help='BOLD repetition time', type=float)
    parser.add_argument('--hrf_window_trs',
                        help='number of TRs post-onset checked for scrubbing (default 2)',
                        type=int, default=2)
    parser.add_argument('--bidsroot', help='top-level directory of the BIDS dataset', type=str)
    parser.add_argument('--fmriprep_dir', help='directory of the fMRIprep preprocessed dataset',
                        type=str)
    parser.add_argument('--out_dir', help='directory to write the per-subject summary CSV to',
                        type=str, default='.')
    args = parser.parse_args()

    if len(sys.argv) < 2:
        parser.print_help()
        sys.exit(1)

    from nilearn.glm.first_level import first_level_from_bids
    from nilearn.interfaces.fmriprep import load_confounds_strategy

    print(f'sub-{args.sub}: discovering runs via first_level_from_bids...', flush=True)
    _, models_run_imgs, models_events, _ = first_level_from_bids(
        args.bidsroot, args.task, args.space, [args.sub],
        derivatives_folder=args.fmriprep_dir)
    run_imgs = models_run_imgs[0]
    run_events_list = models_events[0]
    print(f'sub-{args.sub}: found {len(run_imgs)} runs', flush=True)

    if len(run_imgs) != 6:
        print(f'WARNING: expected 6 runs, found {len(run_imgs)} -- '
              f'RUN_GROUP_DICT indices below may not mean what you think', flush=True)

    per_run_rows = []
    per_trial_frames = []
    for rx, (imgs, run_events) in enumerate(zip(run_imgs, run_events_list)):
        print(f'sub-{args.sub}: run {rx} -- loading confounds...', flush=True)
        run_events = run_events.dropna(subset=['onset']).reset_index(drop=True)

        run_confounds, run_sample_mask = load_confounds_strategy(
            img_files=imgs,
            denoise_strategy=DENOISE_STRATEGY,
            fd_threshold=FD_THRESHOLD,
            std_dvars_threshold=STD_DVARS_THRESHOLD,
        )
        n_vols = len(run_confounds)
        n_retained = n_vols if run_sample_mask is None else len(run_sample_mask)
        print(f'sub-{args.sub}: run {rx} -- done ({n_retained}/{n_vols} volumes retained)', flush=True)

        trial_df = trials_affected_by_scrubbing(
            run_events, run_sample_mask, n_vols, args.t_r, args.hrf_window_trs)
        trial_df['run_index'] = rx
        per_trial_frames.append(trial_df)

        per_run_rows.append({
            'run_index': rx,
            'n_vols': n_vols,
            'n_retained': n_retained,
            'retained_frac': n_retained / n_vols,
            'n_fb_correct': int((trial_df['trial_type'] == 'fb_correct').sum()),
            'n_fb_wrong': int((trial_df['trial_type'] == 'fb_wrong').sum()),
        })

    per_trial_df = pd.concat(per_trial_frames, ignore_index=True)
    per_run_df = pd.DataFrame(per_run_rows)

    # map run_index -> learning_stage via the same grouping univariate_glm.py fits on
    run_to_stage = {rx: stage for stage, run_idxs in RUN_GROUP_DICT.items() for rx in run_idxs}
    per_trial_df['learning_stage'] = per_trial_df['run_index'].map(run_to_stage)

    summary = (per_trial_df.groupby(['learning_stage', 'trial_type'])
               .agg(n_trials=('scrubbed', 'size'), n_affected=('scrubbed', 'sum'))
               .reset_index())
    summary['pct_affected'] = 100 * summary['n_affected'] / summary['n_trials']
    # keep learning-stage order meaningful rather than alphabetical
    stage_order = list(RUN_GROUP_DICT.keys())
    summary['learning_stage'] = pd.Categorical(summary['learning_stage'],
                                               categories=stage_order, ordered=True)
    summary = summary.sort_values(['learning_stage', 'trial_type']).reset_index(drop=True)

    print(f'\nsub-{args.sub}: per-run retained fraction after scrubbing')
    print(per_run_df.to_string(index=False))
    print(f'\nsub-{args.sub}: feedback trials affected by scrubbing '
         f'(any TR in onset..onset+{args.hrf_window_trs} TRs scrubbed), by learning stage')
    print(summary.to_string(index=False))

    os.makedirs(args.out_dir, exist_ok=True)
    per_run_fpath = os.path.join(args.out_dir, f'sub-{args.sub}_task-{args.task}_feedback-scrubbing-per-run.csv')
    summary_fpath = os.path.join(args.out_dir, f'sub-{args.sub}_task-{args.task}_feedback-scrubbing-summary.csv')
    per_run_df.to_csv(per_run_fpath, index=False)
    summary.to_csv(summary_fpath, index=False)
    print(f'\nsaved {per_run_fpath}')
    print(f'saved {summary_fpath}')


if __name__ == '__main__':
    main()
