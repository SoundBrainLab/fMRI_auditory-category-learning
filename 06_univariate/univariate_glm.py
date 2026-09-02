import os
import re
import sys
import argparse

import numpy as np
import pandas as pd

''' Set up and interpret command line arguments '''
parser = argparse.ArgumentParser(
                description='Subject-level modeling of fmriprep-preprocessed data',
                epilog=('Example: python univariate_glm.py --sub=FLT02 '
                        '--task=tonecat --space=MNI152NLin2009cAsym '
                        '--fwhm=3 --event_type=sound --grouping=none '
                        '--t_acq=2 --t_r=3 '
                        '--bidsroot=/PATH/TO/BIDS/DIR/ '
                        '--fmriprep_dir=/PATH/TO/FMRIPREP/DIR/')
                )

parser.add_argument("--sub",
                    help="participant id", type=str)
parser.add_argument("--task",
                    help="task id", type=str)
parser.add_argument("--space",
                    help="space label", type=str)
parser.add_argument("--fwhm",
                    help="spatial smoothing full-width half-max",
                    type=float)
parser.add_argument("--event_type",
                    help="what to model (options: `stimulus` or `trial` or `sound` or `feedback`)",
                    type=str)
parser.add_argument("--grouping",
                    help=("run-grouping strategy: `none` fits all runs together "
                          "in a single GLM; `grouped` fits separate GLMs per "
                          "early/middle/late run-pair (learning stage)"),
                    type=str, choices=['none', 'grouped'], default='none')
parser.add_argument("--t_acq",
                    help=("BOLD acquisition time (if different from "
                          "repetition time [TR], as in sparse designs)"),
                    type=float)
parser.add_argument("--t_r",
                    help="BOLD repetition time",
                    type=float)
parser.add_argument("--bidsroot",
                    help="top-level directory of the BIDS dataset",
                    type=str)
parser.add_argument("--fmriprep_dir",
                    help="directory of the fMRIprep preprocessed dataset",
                    type=str)
parser.add_argument("--no_scrubbing",
                    help=("skip FD/DVARS-based volume censoring while keeping the same "
                          "motion/aCompCor confound regressors -- for isolating scrubbing's "
                          "specific effect via a side-by-side comparison. Outputs land in a "
                          "separate 'noscrub' subdirectory, not overwriting the default "
                          "scrubbed outputs"),
                    action='store_true', default=False)

args = parser.parse_args()

if len(sys.argv) < 2:
    parser.print_help()
    print(' ')
    sys.exit(1)

subject_id = args.sub
task_label = args.task
space_label = args.space
fwhm = args.fwhm
event_type = args.event_type
grouping = args.grouping
t_acq = args.t_acq
t_r = args.t_r
bidsroot = args.bidsroot
fmriprep_dir = args.fmriprep_dir
no_scrubbing = args.no_scrubbing


def _variant_tag(no_scrubbing, event_type):
    ''' single directory level describing every way this run deviates from
    the original scrubbed/uncollapsed pipeline -- rather than one nested
    directory per axis of variation. Deviating axes are joined with '_' so
    every combination gets exactly one flat, self-documenting segment:
        baseline (scrubbed, uncollapsed)      -> 'scrubbed_model-full'
        --no_scrubbing only                   -> 'noscrub'
        feedback's collapsed-nuisance design  -> 'collapsed-nuisance'
        both at once                          -> 'noscrub_collapsed-nuisance'
    'scrubbed_model-full' matches where the pre-existing baseline outputs
    (everything computed before --no_scrubbing/the collapsed-nuisance design
    existed) were moved to on disk -- see REVISION_PLAN.md/group_level_all_ROI.ipynb's
    l1_dir, which must stay in sync with this. Only feedback's design changed
    (sound/stimulus/trial already collapsed nuisance conditions the same way,
    or don't need to), so 'collapsed-nuisance' only ever appears for
    event_type='feedback'; sound/stimulus/trial baseline runs always land in
    'scrubbed_model-full' since neither axis here applies to them. '''
    parts = []
    if no_scrubbing:
        parts.append('noscrub')
    if event_type == 'feedback':
        parts.append('collapsed-nuisance')
    return '_'.join(parts) if parts else 'scrubbed_model-full'


variant_tag = _variant_tag(no_scrubbing, event_type)


def _fmriprep_tag(fmriprep_dir):
    ''' extract a `fmriprep-X.Y.Z` tag from --fmriprep_dir (e.g.
    .../derivatives/denoised_fmriprep-25.2.5/ -> 'fmriprep-25.2.5') to
    namespace nilearn outputs by fmriprep version. Without this, outputs
    are keyed only by bidsroot/space/fwhm/event_type -- nothing about the
    path reflects which fmriprep run produced the inputs. That's fine for
    template spaces (e.g. MNI152NLin2009cAsym) but not for T1w: fmriprep
    re-derives native T1w space from scratch every run (different
    recon-all/registration per version), so "T1w" from one fmriprep
    version and "T1w" from another are different spaces that would
    otherwise silently overwrite each other's results in derivatives/nilearn/. '''
    match = re.search(r'fmriprep-[\d.]+', os.path.basename(os.path.normpath(fmriprep_dir)))
    return match.group(0) if match else os.path.basename(os.path.normpath(fmriprep_dir))


fmriprep_tag = _fmriprep_tag(fmriprep_dir)


# ## nilearn modeling: first level
# based on: https://nilearn.github.io/auto_examples/04_glm_first_level/
# plot_bids_features.html#sphx-glr-auto-examples-04-glm-first-level-plot-bids-features-py

def _save_confound_qc(subject_id, task_label, event_type, bidsroot, fmriprep_tag, variant_tag,
                      models_confounds, models_sample_masks, min_retained_frac=0.5):
    ''' record per-run scrubbing stats -- for review, not for excluding data.
    We keep every run regardless of retained fraction (n=12 means every run
    counts), but still want low-quality runs visible somewhere. '''
    rows = []
    for confounds, sample_masks in zip(models_confounds, models_sample_masks):
        for rx, (conf, mask) in enumerate(zip(confounds, sample_masks)):
            n_vols = len(conf)
            n_retained = n_vols if mask is None else len(mask)
            retained_frac = n_retained / n_vols
            rows.append({
                'run_index': rx,
                'n_vols': n_vols,
                'n_retained': n_retained,
                'retained_frac': retained_frac,
                'low_quality': retained_frac < min_retained_frac,
            })

    qc_dir = os.path.join(bidsroot, 'derivatives', 'nilearn',
                          *(p for p in (fmriprep_tag, variant_tag) if p), 'qc')
    os.makedirs(qc_dir, exist_ok=True)
    qc_fpath = os.path.join(
        qc_dir, f'sub-{subject_id}_task-{task_label}_event-{event_type}_confound-scrubbing-qc.csv')
    pd.DataFrame(rows).to_csv(qc_fpath, index=False)
    print(f'saved confound-scrubbing QC to {qc_fpath}')

    n_low_quality = sum(r['low_quality'] for r in rows)
    if n_low_quality:
        print(f'NOTE: {n_low_quality} run(s) below {min_retained_frac:.0%} volumes-retained '
              f'threshold after scrubbing -- see {qc_fpath} (not excluded from analysis)')


def prep_models_and_args(subject_id=None, task_id=None, fwhm=None, bidsroot=None,
                         deriv_dir=None, event_type=None, t_r=None, t_acq=None, space_label='T1w',
                         no_scrubbing=False, fmriprep_tag=None, variant_tag=None):
    from nilearn.glm.first_level import first_level_from_bids
    from nilearn.interfaces.fmriprep import load_confounds_strategy
    data_dir = bidsroot

    task_label = task_id
    fwhm_sub = fwhm

    print(data_dir, task_label, space_label)

    # slice_time_ref intentionally left unset (not e.g. the old manual
    # 0.5*t_acq/t_r sparse-acquisition correction): confirmed on real output
    # (sub-FLT02's statmap.json) that first_level_from_bids successfully
    # infers it from the BOLD JSON sidecar's StartTime field (0.3233,
    # matching the old manual formula's 0.3333 to within ~3%) rather than
    # falling back to the wrong default of 0.0 -- see REVISION_PLAN.md for
    # how to re-check this if it's ever in doubt (grep run logs for
    # "slice_time_ref' not provided", which only appears if inference fails)
    models, models_run_imgs, \
            models_events, _ = first_level_from_bids(data_dir, task_label,
                                                     space_label, [subject_id],
                                                     smoothing_fwhm=fwhm,
                                                     derivatives_folder=deriv_dir,
                                                     # save_glm_to_bids (used by both
                                                     # grouping modes) unconditionally
                                                     # tries to save model-level maps
                                                     # (R-squared, residuals) that
                                                     # require this -- without it every
                                                     # save_glm_to_bids call raises and
                                                     # gets silently swallowed by
                                                     # grouped_runs' try/except
                                                     minimize_memory=False,
                                                    )

    # load confounds with FD/DVARS-based scrubbing (motion + compcor regressors
    # bundled automatically), replacing the old manual confound-column list
    # (which had aCompCor disabled and no motion scrubbing/censoring). Mirrors
    # the validated setup in sitek/SSP's univariate_fmri/univariate_first-level.py.
    # Low-quality runs are logged (see _save_confound_qc) but NOT excluded --
    # every run counts at n=12.
    models_confounds = []
    models_sample_masks = []
    for run_imgs in models_run_imgs:
        run_confounds, run_sample_masks = load_confounds_strategy(
            img_files=run_imgs,
            denoise_strategy='scrubbing',
            fd_threshold=0.9,
            std_dvars_threshold=1.5,
        )
        if no_scrubbing:
            # keep the same confound regressors (motion + aCompCor, still
            # sourced from the 'scrubbing' strategy above) but discard the
            # censoring mask itself -- isolates scrubbing's specific effect
            # by holding everything else in the denoising strategy constant.
            # Explicit "keep every volume" index arrays, NOT a list of None:
            # confirmed empirically that FirstLevelModel.fit() accepts either
            # a single bare None (no masking for the whole call) or a list of
            # real index arrays, but rejects a list containing None entries
            # (nilearn's check_run_sample_masks tries to iterate each one).
            # Per-run indexing elsewhere (grouped_runs slicing a subset of
            # runs into its own model.fit call) needs this to stay a real
            # per-run list, so a single overall None isn't an option here.
            run_sample_masks = [np.arange(len(conf)) for conf in run_confounds]
        else:
            # load_confounds_strategy documents returning None (not an index
            # array) for any individual run where zero volumes needed
            # censoring -- perfectly plausible for a single low-motion run
            # even when a sibling run in the same run-group had some
            # scrubbing. That produces the exact same list-of-[array, None]
            # mix FirstLevelModel.fit() rejects (see the no_scrubbing comment
            # above): confirmed on real data (sub-FLT20's 'early' run-group,
            # run 1 at 100% retained -> None, run 0 partially scrubbed ->
            # array) raising 'NoneType' object is not iterable and silently
            # dropping that entire run-group via grouped_runs' try/except.
            # None means "nothing scrubbed" here too, so this normalization
            # changes nothing about which volumes are used -- it only avoids
            # the mixed-type crash.
            run_sample_masks = [
                (np.arange(len(conf)) if mask is None else mask)
                for conf, mask in zip(run_confounds, run_sample_masks)
            ]
        models_confounds.append(run_confounds)
        models_sample_masks.append(run_sample_masks)

    _save_confound_qc(subject_id, task_label, event_type, bidsroot, fmriprep_tag, variant_tag,
                      models_confounds, models_sample_masks)

    ''' create events '''
    for sx, sub_events in enumerate(models_events):
        for mx, run_events in enumerate(sub_events):
            # drop rows where the event didn't actually occur on that trial
            # (e.g. feedback isn't delivered on a missed-response trial, so
            # its row is present in events.tsv with onset left blank) --
            # nilearn's design-matrix builder rejects any NaN onset outright,
            # regardless of event_type, so this has to happen before any of
            # the relabeling below
            run_events = run_events.dropna(subset=['onset']).reset_index(drop=True)

            # stimulus events
            if event_type == 'stimulus':
                name_groups = run_events.groupby('trial_type')['trial_type']
                suffix = name_groups.cumcount() + 1
                repeats = name_groups.transform('size')

                run_events['trial_type'] = run_events['trial_type']
                run_events['trial_type'] = run_events['trial_type'].str.replace('-','_')

            # trial-specific events
            if event_type == 'trial':
                name_groups = run_events.groupby('trial_type')['trial_type']
                suffix = name_groups.cumcount() + 1
                repeats = name_groups.transform('size')

                run_events['trial_type'] = run_events['trial_type'] + \
                                                    '_trial' + suffix.map(str)
                run_events['trial_type'] = run_events['trial_type'].str.replace('-','_')

            # combine all sound events
            elif event_type == 'sound':
                orig_stim_list = sorted([str(s) for s in run_events['trial_type'].unique()
                                         if str(s) not in ['nan', 'None', 'null']])
                #print('original stim list: ', orig_stim_list)

                run_events['trial_type'] = run_events.trial_type.str.split('_', expand=True)[0]

            # collapse sound/response conditions into single nuisance
            # regressors ('sound' mode already does this same split-on-
            # underscore collapse; feedback needs the same treatment for the
            # same reason, minus collapsing away the feedback outcomes
            # themselves, which are the actual contrast of interest and stay
            # as their own separate conditions). Cuts the design matrix from
            # ~24 task regressors to 5 (fb_correct/fb_wrong/fb_noresp/resp/
            # sound), meaningfully easing the residual-DOF strain scrubbing
            # already puts on these short runs.
            elif event_type == 'feedback':
                is_nuisance = run_events['trial_type'].str.startswith(('sound_', 'resp_'))
                run_events.loc[is_nuisance, 'trial_type'] = (
                    run_events.loc[is_nuisance, 'trial_type'].str.split('_', expand=True)[0])

            # re-assign to models_events
            models_events[sx][mx] = run_events

        # create stimulus list from updated events.tsv file
        stim_list = sorted([str(s) for s in run_events['trial_type'].unique() if str(s) not in ['nan', 'None']])

    # feedback is modeled as a single differential contrast rather than one
    # contrast per condition -- express that as a {contrast_id: contrast_def}
    # dict here so save_glm_to_bids's `contrasts` argument (which accepts
    # either a list of expressions or an id->expression dict) is the only
    # thing that varies by event_type; the GLM-fitting code downstream
    # doesn't need to know event_type exists. `fbcorrectvswrong` (rather
    # than a dashed label) is used as the id since it survives BIDS
    # entity-value sanitization unchanged.
    if event_type == 'feedback':
        stim_list = {'fbcorrectvswrong': 'fb_correct - fb_wrong'}

    #model_and_args = zip(models, models_run_imgs, models_events, models_confounds)
    return stim_list, models, models_run_imgs, models_events, models_confounds, models_sample_masks


# ### Grouping: none -- fit all runs together in a single GLM
def nilearn_glm_across_runs(stim_list, task_label, models, models_run_imgs,
                            models_events, models_confounds, models_sample_masks,
                            space_label, event_type, bidsroot, fmriprep_tag, variant_tag):
    from nilearn.glm import save_glm_to_bids

    bidsderiv_sub_dir = None
    for midx in range(len(models)):
        model = models[midx]
        imgs = models_run_imgs[midx]
        events = models_events[midx]
        confounds = models_confounds[midx]
        sample_masks = models_sample_masks[midx]

        # fit once regardless of how many contrasts get computed from it --
        # the design matrix doesn't change per contrast
        print('fitting GLM')
        model.fit(imgs, events, confounds, sample_masks=sample_masks)

        # one contrast per condition, but a single save_glm_to_bids call
        # handles the whole stim_list at once (it accepts a list or dict of
        # contrast definitions -- see prep_models_and_args for how feedback's
        # single differential contrast fits into that) instead of looping
        # compute_contrast + manual saving ourselves per stimulus
        print('computing and saving contrasts for all conditions')
        bidsderiv_sub_dir = os.path.join(bidsroot, 'derivatives', 'nilearn',
                                         *(p for p in (fmriprep_tag, variant_tag) if p),
                                         'bids-deriv_level-1_fwhm-%.02f'%model.smoothing_fwhm,
                                         f'sub-{model.subject_label}_space-{space_label}',
                                         f'run-all_event-{event_type}')
        os.makedirs(bidsderiv_sub_dir, exist_ok=True)

        out_prefix = f"sub-{model.subject_label}_task-{task_label}_fwhm-{model.smoothing_fwhm}"
        save_glm_to_bids(model,
                         contrasts=stim_list,
                         out_dir=bidsderiv_sub_dir,
                         prefix=out_prefix,
                        )
        print(f'Saved model outputs to {bidsderiv_sub_dir}')
    return bidsderiv_sub_dir


# ### Grouping: grouped -- fit separate GLMs per early/middle/late run-pair
def nilearn_glm_grouped_runs(stim_list, task_label, models, models_run_imgs,
                            models_events, models_confounds, models_sample_masks,
                            space_label, event_type, bidsroot, fmriprep_tag, variant_tag):
    from nilearn.glm import save_glm_to_bids

    #run_group_dict = {'firsthalf': [0, 1, 2],
    #                  'secondhalf': [3, 4, 5]}
    run_group_dict = {'early': [0, 1],
                      'middle': [2, 3],
                      'final': [4, 5]}

    bidsderiv_sub_dir = None
    for midx in range(len(models)):
        model = models[midx]
        imgs = models_run_imgs[midx]
        events = models_events[midx]
        confounds = models_confounds[midx]
        sample_masks = models_sample_masks[midx]

        print(model.subject_label)

        for run_group in run_group_dict:
            imgs_grouped = [imgs[x] for x in run_group_dict[run_group]]
            events_grouped = [events[x] for x in run_group_dict[run_group]]
            confounds_grouped = [confounds[x] for x in run_group_dict[run_group]]
            sample_masks_grouped = [sample_masks[x] for x in run_group_dict[run_group]]

            try:
                # fit once per run-group regardless of how many contrasts
                # get computed from it, instead of re-fitting per stimulus
                print('fitting GLM on ', imgs_grouped)
                model.fit(imgs_grouped, events_grouped, confounds_grouped,
                          sample_masks=sample_masks_grouped)

                print('computing and saving contrasts for all conditions')
                bidsderiv_sub_dir = os.path.join(bidsroot, 'derivatives', 'nilearn',
                                                 *(p for p in (fmriprep_tag, variant_tag) if p),
                                                 'bids-deriv_level-1_fwhm-%.02f'%model.smoothing_fwhm,
                                                 f'sub-{model.subject_label}_space-{space_label}',
                                                 f'{run_group}_event-{event_type}')
                os.makedirs(bidsderiv_sub_dir, exist_ok=True)

                out_prefix = f"sub-{model.subject_label}_task-{task_label}_fwhm-{model.smoothing_fwhm}"
                save_glm_to_bids(model,
                                 contrasts=stim_list,
                                 out_dir=bidsderiv_sub_dir,
                                 prefix=out_prefix,
                                )

                # save_glm_to_bids has returned without raising here, but a
                # prior run silently produced zero files for one run-group
                # (FLT20/early) despite reaching this same print statement --
                # verify from inside this process, at write time, that
                # anything actually landed on disk
                n_written = sum(len(files) for _, _, files in os.walk(bidsderiv_sub_dir))
                if n_written == 0:
                    print(f'WARNING: save_glm_to_bids() returned normally but '
                         f'{bidsderiv_sub_dir} contains zero files immediately after. '
                         f'Diagnostic state for {run_group}:')
                    print(f'  imgs_grouped: {imgs_grouped}')
                    print(f'  sample_masks_grouped types/lens: '
                         f'{[(type(m).__name__, None if m is None else len(m)) for m in sample_masks_grouped]}')
                    print(f'  confounds_grouped shapes: '
                         f'{[getattr(c, "shape", None) for c in confounds_grouped]}')
                    print(f'  events_grouped shapes: '
                         f'{[getattr(e, "shape", None) for e in events_grouped]}')
                    print(f'  model.labels_ type: {type(getattr(model, "labels_", None))}')
                else:
                    print(f'Saved model outputs to {bidsderiv_sub_dir} ({n_written} files)')
            except Exception as e:
                print(f'could not run {run_group}: {e}')
    return bidsderiv_sub_dir


''' run the pipeline '''

print('Running subject ', subject_id)
stim_list, models, models_run_imgs, models_events, \
           models_confounds, models_sample_masks = prep_models_and_args(subject_id,
                                                                   task_label,
                                                                   fwhm, bidsroot,
                                                                   fmriprep_dir,
                                                                   event_type,
                                                                   t_r, t_acq,
                                                                   space_label,
                                                                   no_scrubbing,
                                                                   fmriprep_tag,
                                                                   variant_tag)

if grouping == 'grouped':
    nilearn_glm_grouped_runs(stim_list, task_label, models, models_run_imgs,
                             models_events, models_confounds, models_sample_masks,
                             space_label, event_type, bidsroot, fmriprep_tag, variant_tag)
else:
    nilearn_glm_across_runs(stim_list, task_label, models, models_run_imgs,
                            models_events, models_confounds, models_sample_masks,
                            space_label, event_type, bidsroot, fmriprep_tag, variant_tag)
