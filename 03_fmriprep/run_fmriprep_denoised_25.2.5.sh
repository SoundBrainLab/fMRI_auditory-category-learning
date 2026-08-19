#!/bin/bash
#SBATCH --time=3-00
#SBATCH --mem=64G
#SBATCH --cpus-per-task=8

# Preprocess single-subject FLT data using fmriprep
# in a Singularity container
# Updated to fmriprep 25.2.5 (current LTS line, support through Oct 2029)
# to pick up fieldmap/SDC masking fixes (23.2.0) and brain-mask
# dilation fixes (24.0.0) since the 22.1.1 run.
# Runs to a SEPARATE output tree from the 22.1.1 derivatives -- do not
# point this at the same out_dir/work_dir as run_fmriprep_denoised.sh.
# Confirm 25.2.5 is still the latest 25.2.x LTS patch before building
# the container; bump fmriprep_version below if a newer patch exists.

module add freesurfer
module add fsl
module add afni
module add ants
module add singularity/3.8.3

#conda activate py3

# define paths
software_path=/bgfs/bchandrasekaran/krs228/software/
project_path=/bgfs/bchandrasekaran/krs228/data/FLT/
data_dir=$project_path/data_denoised/

fmriprep_version=25.2.5
analysis_desc="denoised_fmriprep-$fmriprep_version"
work_dir=/bgfs/bchandrasekaran/krs228/work/${analysis_desc}
out_dir=$data_dir/derivatives/${analysis_desc}/

# singularity
sing_dir=$software_path/singularity_images/
sing_img=$sing_dir/${fmriprep_version}.simg

# define inputs
fs_license=$software_path/license.txt
sub=$1

# NOTE: intentionally NOT reusing the 22.1.1-era FreeSurfer outputs here.
# The bundled FreeSurfer version has moved on since 22.1.1 (7.2 -> 7.3.2+),
# and mixing FreeSurfer versions across a derivatives tree risks subtle
# inconsistencies. Omitting --fs-subjects-dir lets fmriprep run recon-all
# fresh under this run's own <out_dir>/sourcedata/freesurfer/ (the
# --output-layout bids default). This adds several hours/subject of
# recon-all compute versus the old script -- budget for it.

# copy from SBATCH arguments
mem=64000
nprocs=8
omp_n=4

# BEFORE RUNNING FOR THE FIRST TIME:
# build the fmriprep container to a singularity image
# (will only build from head node; no unsquashfs when running from nodes)
#singularity build $sing_img docker://nipreps/fmriprep:25.2.5

# run fmriprep
singularity run --cleanenv -B /bgfs:/bgfs $sing_img \
  $data_dir $out_dir participant \
  --participant-label $sub \
  --fs-license-file $fs_license \
  --work-dir $work_dir \
  --skip-bids-validation \
  --output-layout bids \
  -vv \
  --mem $mem \
  --nprocs $nprocs --omp-nthreads $omp_n \
  --output-spaces T1w func fsnative MNI152NLin2009cAsym
