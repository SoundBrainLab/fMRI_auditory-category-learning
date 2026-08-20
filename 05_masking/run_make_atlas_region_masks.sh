#!/bin/bash
#SBATCH --time=4:00:00
#SBATCH -c 2

sub=$1
atlas=$2

python make_atlas_region_masks.py --sub=$sub \
--space=T1w \
--fwhm=0.00 \
--atlas_label=$atlas \
--bidsroot=/ix1/bchandrasekaran/krs228/data/FLT/data_denoised/ \
--fmriprep_dir=/ix1/bchandrasekaran/krs228/data/FLT/data_denoised/derivatives/denoised_fmriprep-25.2.5/
