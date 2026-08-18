#!/bin/bash

#SBATCH --time=12:00:00

# NOTE: previously called univariate_analysis_fb-correct-vs-wrong.py with
# --event_type=stimulus. That script never actually branched on event_type
# (it always computed the feedback contrast unconditionally), but the
# consolidated univariate_glm.py does branch on event_type=='feedback' to
# pick the right output-saving path, so this must say `feedback` now.
python univariate_glm.py --sub=$1 \
                              --task=tonecat \
                              --space=MNI152NLin2009cAsym \
                              --fwhm=0.00 \
                              --event_type=feedback \
                              --grouping=none \
                              --t_acq=2 --t_r=3 \
                              --bidsroot=/bgfs/bchandrasekaran/krs228/data/FLT/data_denoised/ \
                              --fmriprep_dir=/bgfs/bchandrasekaran/krs228/data/FLT/data_denoised/derivatives/denoised_fmriprep-22.1.1/

