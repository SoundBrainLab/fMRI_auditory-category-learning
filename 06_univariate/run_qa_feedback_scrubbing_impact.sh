#!/bin/bash

#SBATCH --time=00:30:00
#SBATCH --mem=8G

# no GLM fitting or 4D image loading here (just events.tsv + confounds.tsv
# parsing per run), so this should take well under a minute per subject --
# generous walltime/mem above just in case of slow network-filesystem I/O

python qa_feedback_scrubbing_impact.py --sub=$1 \
      --task=tonecat \
      --space=T1w \
      --t_acq=2 --t_r=3 \
      --bidsroot=/ix1/bchandrasekaran/krs228/data/FLT/data_denoised/ \
      --fmriprep_dir=/ix1/bchandrasekaran/krs228/data/FLT/data_denoised/derivatives/denoised_fmriprep-25.2.5/ \
      --out_dir=/ix1/bchandrasekaran/krs228/data/FLT/data_denoised/derivatives/nilearn/fmriprep-25.2.5/qc/
