#!/bin/bash
#SBATCH --time=1:00:00
#SBATCH --mem=8G

# Warp group-level MNI-space atlases into a single subject's native T1w
# space, using the subject-specific nonlinear transform fMRIPrep already
# computes as part of its normal anatomical workflow. Requires the
# fMRIPrep 25.2.5 anat derivatives to exist first
# (03_fmriprep/run_fmriprep_denoised_25.2.5.sh).
#
# Atlases warped: Tian S2 striatal parcellation, subcortical auditory
# pathway (CN/SOC/IC/MGN), and carpet_dseg (cortical+subcortical
# HG/PT/STG/SMG/IFG/Ang/Caud/Put/etc.). Tian S3 is intentionally NOT
# warped -- retired in favor of Tian S2 going forward.
#
# GenericLabel interpolation is used throughout since these are discrete
# multi-label atlases -- plain nearest-neighbor can misbehave at label
# boundaries. This matters especially for the auditory-pathway atlas,
# whose targets are smaller brainstem/thalamic structures.

module add ants

sub=$1

fmriprep_dir=/bgfs/bchandrasekaran/krs228/data/FLT/data_denoised/derivatives/denoised_fmriprep-25.2.5/
nilearn_dir=/bgfs/bchandrasekaran/krs228/data/FLT/data_denoised/derivatives/nilearn/

anat_dir=$fmriprep_dir/sub-${sub}/anat
t1w_ref=$anat_dir/sub-${sub}_desc-preproc_T1w.nii.gz
xfm=$anat_dir/sub-${sub}_from-MNI152NLin2009cAsym_to-T1w_mode-image_xfm.h5
# NOTE: confirm the exact transform filename against the real fMRIPrep
# 25.2.5 anat derivatives once that run completes. This naming has been
# stable across recent fMRIPrep versions but has not been verified here.

out_dir=$nilearn_dir/masks/sub-${sub}/space-T1w/atlas-native
mkdir -p $out_dir

# Tian S2 striatal parcellation
tian_s2_atlas=/bgfs/bchandrasekaran/krs228/data/reference/subcortex/Group-Parcellation/7T/Tian_Subcortex_S2_7T.nii
antsApplyTransforms -d 3 \
  -i $tian_s2_atlas \
  -r $t1w_ref \
  -t $xfm \
  -n GenericLabel \
  -o $out_dir/sub-${sub}_space-T1w_atlas-tianS2.nii.gz

# subcortical auditory pathway (cochlear nucleus / superior olivary
# complex / inferior colliculus / medial geniculate nucleus)
subcort_aud_atlas=/bgfs/bchandrasekaran/krs228/data/reference/MNI_space/atlases/sub-bigbrain_MNI_conjunction_rois.nii.gz
antsApplyTransforms -d 3 \
  -i $subcort_aud_atlas \
  -r $t1w_ref \
  -t $xfm \
  -n GenericLabel \
  -o $out_dir/sub-${sub}_space-T1w_atlas-subcortaud.nii.gz

# cortical+subcortical carpet_dseg
carpet_dseg_atlas=/bgfs/bchandrasekaran/krs228/data/reference/tpl-MNI152NLin2009cAsym_res-01_desc-carpet_dseg.nii.gz
antsApplyTransforms -d 3 \
  -i $carpet_dseg_atlas \
  -r $t1w_ref \
  -t $xfm \
  -n GenericLabel \
  -o $out_dir/sub-${sub}_space-T1w_atlas-carpetdseg.nii.gz

echo "warped atlases saved to $out_dir"
