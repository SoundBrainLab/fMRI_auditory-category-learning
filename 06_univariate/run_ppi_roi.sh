#!/bin/bash

#SBATCH --time=4:00:00
#SBATCH --mem=32G

# ROI-to-ROI gPPI (aCAU and aPUT seeds -> auditory ROIs), one participant;
# after all participants finish: python ppi_roi.py --gather --bidsroot=<same>
python ppi_roi.py --sub=$1 \
      --task=tonecat \
      --space=T1w \
      --bidsroot=/ix1/bchandrasekaran/krs228/data/FLT/data_denoised/ \
      --fmriprep_dir=/ix1/bchandrasekaran/krs228/data/FLT/data_denoised/derivatives/denoised_fmriprep-25.2.5/
