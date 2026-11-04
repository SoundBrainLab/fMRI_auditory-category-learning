#!/bin/bash

#SBATCH --time=4:00:00

# Per-ROI mean first-level effects for every participant (both groups),
# network, contrast and learning stage -> one long table per network x
# contrast under derivatives/nilearn/group_space-T1w_fwhm-0.00/roi-values_scrubbed-splitfb/.
# Run after loop_univariate_groupedruns.sh has finished.

python extract_roi_values.py \
      --bidsroot=/ix1/bchandrasekaran/krs228/data/FLT/data_denoised/ \
      --variant_tag=scrubbed-splitfb
