#!/bin/bash

#SBATCH --time=12:00:00

# --grouping=grouped, not none: the manuscript's feedback results are
# learning-stage breakdowns (Table 8, early/middle/late thirds), matching
# run_univariate_groupedruns_denoised.sh's grouping -- just a different
# event_type/contrast.
python univariate_glm.py --sub=$1 \
                              --task=tonecat \
                              --space=MNI152NLin2009cAsym \
                              --fwhm=0.00 \
                              --event_type=feedback \
                              --grouping=grouped \
                              --t_acq=2 --t_r=3 \
                              --bidsroot=/bgfs/bchandrasekaran/krs228/data/FLT/data_denoised/ \
                              --fmriprep_dir=/bgfs/bchandrasekaran/krs228/data/FLT/data_denoised/derivatives/denoised_fmriprep-22.1.1/

