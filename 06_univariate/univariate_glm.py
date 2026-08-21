import os
import re
import sys
import argparse

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

def _save_confound_qc(subject_id, task_label, event_type, bidsroot, fmriprep_tag,
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

    qc_dir = os.path.join(bidsroot, 'derivatives', 'nilearn', fmriprep_tag, 'qc')
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
                         deriv_dir=None, event_type=None, t_r=None, t_acq=None, space_label='T1w'):
    from nilearn.glm.first_level import first_level_from_bids
    from nilearn.interfaces.fmriprep import load_confounds_strategy
    data_dir = bidsroot

    task_label = task_id
    fwhm_sub = fwhm

    # correct the fmriprep-given slice reference (middle slice, or 0.5)
    # to account for sparse acquisition (silent gap during auditory presentation paradigm)
    # fmriprep is explicitly based on slice timings, while nilearn is based on t_r
    # and since images are only collected during a portion of the overall t_r
    # (which includes the silent gap), we need to account for this
    # BUT it's also included in the header, so maybe we don't need this. 2026-08-20
    slice_time_ref = 0.5 * t_acq / t_r
    

    print(data_dir, task_label, space_label)

    models, models_run_imgs, \
            models_events, _ = first_level_from_bids(data_dir, task_label,
                                                     space_label, [subject_id],
                                                     smoothing_fwhm=fwhm,
                                                     derivatives_folder=deriv_dir,
                                                    #slice_time_ref=slice_time_ref
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
        models_confounds.append(run_confounds)
        models_sample_masks.append(run_sample_masks)

    _save_confound_qc(subject_id, task_label, event_type, bidsroot, _fmriprep_tag(deriv_dir),
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
                            space_label, event_type, bidsroot, fmriprep_tag):
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
        bidsderiv_sub_dir = os.path.join(bidsroot, 'derivatives', 'nilearn', fmriprep_tag,
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
                            space_label, event_type, bidsroot, fmriprep_tag):
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
                bidsderiv_sub_dir = os.path.join(bidsroot, 'derivatives', 'nilearn', fmriprep_tag,
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
                print(f'Saved model outputs to {bidsderiv_sub_dir}')
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
                                                                   space_label)

if grouping == 'grouped':
    nilearn_glm_grouped_runs(stim_list, task_label, models, models_run_imgs,
                             models_events, models_confounds, models_sample_masks,
                             space_label, event_type, bidsroot, fmriprep_tag)
else:
    nilearn_glm_across_runs(stim_list, task_label, models, models_run_imgs,
                            models_events, models_confounds, models_sample_masks,
                            space_label, event_type, bidsroot, fmriprep_tag)
