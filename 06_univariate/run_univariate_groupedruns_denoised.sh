#!/bin/bash

#SBATCH --time=6:00:00

python univariate_glm.py --sub=$1 \
              --task=tonecat \
              --space=T1w \
              --fwhm=0.00 \
              --event_type=sound \
              --grouping=grouped \
              --t_acq=2 --t_r=3 \
              --bidsroot=/ix1/bchandrasekaran/krs228/data/FLT/data_denoised/ \
              --fmriprep_dir=/ix1/bchandrasekaran/krs228/data/FLT/data_denoised/derivatives/denoised_fmriprep-25.2.5/
