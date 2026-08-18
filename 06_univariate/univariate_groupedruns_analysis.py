import os
import sys
import json
import argparse

import numpy as np
import matplotlib.pyplot as plt
import nibabel as nib

from glob import glob
from nilearn import plotting

''' Set up and interpret command line arguments '''
parser = argparse.ArgumentParser(
                description='Subject-level modeling of fmriprep-preprocessed data',
                epilog=('Example: python univariate_analysis.py --sub=FLT02 '
                        '--task=tonecat --space=MNI152NLin2009cAsym '
                        '--fwhm=3 --event_type=sound --t_acq=2 --t_r=3 '
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
                    help="what to model (options: `sound` or `stimulus` or `feedback`)", 
                    type=str)
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
space_label=args.space
fwhm = args.fwhm
event_type=args.event_type
t_acq = args.t_acq
t_r = args.t_r
bidsroot = args.bidsroot
fmriprep_dir = args.fmriprep_dir


# ## nilearn modeling: first level
# based on: https://nilearn.github.io/auto_examples/04_glm_first_level/
# plot_bids_features.html#sphx-glr-auto-examples-04-glm-first-level-plot-bids-features-py

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
    slice_time_ref = 0.5 * t_acq / t_r

    print(data_dir, task_label, space_label)

    models, models_run_imgs, \
            models_events, _ = first_level_from_bids(data_dir, task_label,
                                                                    space_label, [subject_id],
                                                                    smoothing_fwhm=fwhm,
                                                                    derivatives_folder=deriv_dir,
                                                                    slice_time_ref=slice_time_ref)

    # load confounds with FD/DVARS-based scrubbing (motion + compcor regressors
    # bundled automatically), replacing the old manual confound-column list
    # (which had aCompCor disabled and no motion scrubbing/censoring). Mirrors
    # the validated setup in sitek/SSP's univariate_fmri/univariate_first-level.py.
    #
    # NOTE: unlike univariate_analysis.py / univariate_analysis_fb-correct-vs-wrong.py,
    # this script does NOT auto-drop low-quality runs -- nilearn_glm_grouped_runs
    # below selects runs by fixed position (run_group_dict), so silently dropping
    # a run would shift indices and pair the wrong runs together. Runs below
    # threshold are only warned about here; excluding one requires manually
    # updating run_group_dict to match.
    min_retained_frac = 0.5
    models_confounds = []
    models_sample_masks = []
    for run_imgs in models_run_imgs:
        run_confounds, run_sample_masks = load_confounds_strategy(
            img_files=run_imgs,
            denoise_strategy='scrubbing',
            fd_threshold=0.9,
            std_dvars_threshold=1.5,
        )
        for rx, (conf, mask) in enumerate(zip(run_confounds, run_sample_masks)):
            n_vols = len(conf)
            n_retained = n_vols if mask is None else len(mask)
            retained_frac = n_retained / n_vols
            if retained_frac < min_retained_frac:
                print('WARNING: run %d retains only %.0f%% of volumes after '
                      'scrubbing (< %.0f%% threshold) -- NOT auto-excluded, see '
                      'note above; check run_group_dict manually' % (
                          rx, 100 * retained_frac, 100 * min_retained_frac))
        models_confounds.append(run_confounds)
        models_sample_masks.append(run_sample_masks)

    ''' create events '''
    for sx, sub_events in enumerate(models_events):        
        for mx, run_events in enumerate(sub_events):
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

            # all sound events
            elif event_type == 'sound':
                orig_stim_list = sorted([str(s) for s in run_events['trial_type'].unique() 
                                         if str(s) not in ['nan', 'None', 'null']])
                #print('original stim list: ', orig_stim_list)

                run_events['trial_type'] = run_events.trial_type.str.split('_', expand=True)[0]

            # re-assign to models_events
            models_events[sx][mx] = run_events
            
        # create stimulus list from updated events.tsv file
        stim_list = sorted([str(s) for s in run_events['trial_type'].unique() if str(s) not in ['nan', 'None']])
    
    #model_and_args = zip(models, models_run_imgs, models_events, models_confounds)
    return stim_list, models, models_run_imgs, models_events, models_confounds, models_sample_masks


# ### Across-runs GLM
def nilearn_glm_grouped_runs(stim_list, task_label, models, models_run_imgs, \
                            models_events, models_confounds, models_sample_masks, space_label,
                            event_type):
    from nilearn.reporting import make_glm_report

    #run_group_dict = {'firsthalf': [0, 1, 2],
    #                  'secondhalf': [3, 4, 5]}
    run_group_dict = {'earlythird': [0, 1],
                      'middlethird': [2, 3],
                      'latethird': [4, 5]}

    for midx in range(len(models)):
        # only run a single contrast if feedback condition
        if event_type == 'feedback':
            stim_list = ['fb-correct-vs-wrong']

        for sx, stim in enumerate(stim_list):
            if event_type == 'feedback':
                contrast_label = 'fb_correct - fb_wrong'
                contrast_desc  = stim
            else:
                contrast_label = stim
                contrast_desc  = stim


            model = models[midx]
            imgs = models_run_imgs[midx]
            events = models_events[midx]
            confounds = models_confounds[midx]
            sample_masks = models_sample_masks[midx]

            print(model.subject_label)

            #for rx in range(len(imgs)):
            for run_group in run_group_dict:
                imgs_grouped = [imgs[x] for x in run_group_dict[run_group]]
                events_grouped = [events[x] for x in run_group_dict[run_group]]
                confounds_grouped = [confounds[x] for x in run_group_dict[run_group]]
                sample_masks_grouped = [sample_masks[x] for x in run_group_dict[run_group]]

                try:
                    # fit the GLM
                    print('fitting GLM on ', imgs_grouped)
                    model.fit(imgs_grouped, events_grouped, confounds_grouped,
                              sample_masks=sample_masks_grouped);

                    # compute the contrast of interest
                    print('computing contrast of interest')
                    summary_statistics = model.compute_contrast(contrast_label, output_type='all')
                    zmap = summary_statistics['z_score']
                    statmap = summary_statistics['effect_size']

                    # save z map
                    print('saving z-map')
                    nilearn_sub_dir = os.path.join(bidsroot, 'derivatives', 'nilearn', 
                                                   'level-1_fwhm-%.02f'%model.smoothing_fwhm, 
                                                   'sub-%s_space-%s'%(model.subject_label, space_label),
                                                   'grouped_runs', run_group)
                    if not os.path.exists(nilearn_sub_dir):
                        os.makedirs(nilearn_sub_dir)

                    analysis_prefix = 'sub-%s_task-%s_fwhm-%.02f_space-%s_contrast-%s'%(model.subject_label,
                                                                                        task_label, 
                                                                                        model.smoothing_fwhm,
                                                                                        space_label, 
                                                                                        contrast_desc)
                    zmap_fpath = os.path.join(nilearn_sub_dir,
                                            analysis_prefix+'_map-zscore.nii.gz')
                    nib.save(zmap, zmap_fpath)
                    print('saved z map to ', zmap_fpath)

                    # also save beta maps
                    statmap_fpath = os.path.join(nilearn_sub_dir,
                                                analysis_prefix+'_map-beta.nii.gz')
                    nib.save(statmap, statmap_fpath)
                    print('saved beta map to ', statmap_fpath)

                    # save report
                    print('saving report')
                    report_fpath = os.path.join(nilearn_sub_dir,
                                                analysis_prefix+'_report.html')
                    report = make_glm_report(model=model,
                                            contrasts=contrast_label)
                    report.save_as_html(report_fpath)
                    print('saved report to ', report_fpath)
                except:
                    print('could not run for ', contrast_label)
    return zmap_fpath, statmap_fpath, contrast_label

''' run the pipeline '''

print('Running subject ', subject_id)
# Univariate analysis: MNI space, 3 mm, across-run GLM
stim_list, models, models_run_imgs, models_events, \
           models_confounds, models_sample_masks = prep_models_and_args(subject_id,
                                                                   task_label,
                                                                   fwhm, bidsroot,
                                                                   fmriprep_dir,
                                                                   event_type,
                                                                   t_r, t_acq,
                                                                   space_label)
# Across-run GLM
zmap_fpath, statmap_fpath, \
            contrast_label = nilearn_glm_grouped_runs(stim_list, task_label,
                                                     models, models_run_imgs,
                                                     models_events, models_confounds,
                                                     models_sample_masks, space_label,
                                                     event_type)
