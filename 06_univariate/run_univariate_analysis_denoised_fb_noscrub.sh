#!/bin/bash

#SBATCH --time=12:00:00

# Same feedback analysis as run_univariate_analysis_denoised_fb.sh, but with
# --no_scrubbing: same confound regressors (motion + aCompCor), no FD/DVARS
# volume censoring. Outputs land in a separate .../noscrub/ subdirectory
# (see univariate_glm.py's _denoise_tag) so they never overwrite the default
# scrubbed outputs -- built specifically to isolate how much of the striatal
# fb_correct-vs-wrong difference from the original manuscript is attributable
# to scrubbing, versus everything else (fmriprep version, T1w space, etc.).

python univariate_glm.py --sub=$1 \
      --task=tonecat \
      --space=T1w \
      --fwhm=0.00 \
      --event_type=feedback \
      --grouping=grouped \
      --no_scrubbing \
      --t_acq=2 --t_r=3 \
      --bidsroot=/ix1/bchandrasekaran/krs228/data/FLT/data_denoised/ \
      --fmriprep_dir=/ix1/bchandrasekaran/krs228/data/FLT/data_denoised/derivatives/denoised_fmriprep-25.2.5/
