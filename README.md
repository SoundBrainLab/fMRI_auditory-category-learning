Processing and analyzing tone-learning fMRI data collected at the University of Pittsburgh's 7T MRI Center.

# Processing pipeline

## Dicom conversion: `./01_dicom_conversion/`
1. Peek at the dicom .tsv file  using `initialize_dicoms_heudiconv.sh`
2. Create `heuristic.py` based on your MRI sequences
3. Convert dicoms to .nii using `convert_dicoms_heudiconv.sh`

## Image denoising: `./02_denoising/`
1. Run `dwi_denoise` on newly converted BIDS-formatted NIfTI files

## MRI preprocessing: `./03_fmriprep/`
1. Preprocess anatomical and functional MRI with `run_fmriprep.sh` 
> (Note: this runs using a Singularity image, so may need to create that first)

## Behavior Behavioral data conversion: `./04_behavior/`
1. Run `convert_behav_to_bids.py` to get psychopy outputs into BIDS-compatible format
2. Run behavioral analysis notebook

## Masking: `./05_masking/`
1. Create grey matter mask for searchlight using `make_gm_mask.py`
2. Create participant-specific region-of-interest masks

## Univariate analysis: `./06_univariate/`
1. Run `univariate_analysis.py`
2. Run `group_level.ipynb` for group-level GLM and output maps/figures

## Representational similarity analysis: `./07_rsa/`
1. Create event-specific beta estimates
2. Run region-based RSA using atlas masks (see [masking](#Masking))
3. Compute group-level RSA statistics for cortical and striatal networks
