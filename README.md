Processing and analyzing tone-learning fMRI data. WIP - KRS 2022.10

## Processing pipeline

### Dicom conversion: `./dicom_conversion/`
1. Peek at the dicom .tsv file  using `initialize_dicoms_heudiconv.sh`
2. Create `heuristic.py` based on your MRI sequences
3. Convert dicoms to .nii using `convert_dicoms_heudiconv.sh`

### MRI preprocessing: `./fmriprep/`
1. Preprocess anatomical and functional MRI with `run_fmriprep.sh` 
> (Note: this runs using a Singularity image, so may need to create that first)

### Behavior Behavioral data conversion: `./behavior/`
1. Run `convert_behav_to_bids.py` to get psychopy outputs into BIDS-compatible format
2. Run behavioral analysis notebook

### Masking
1. Create grey matter mask for searchlight using `make_gm_mask.py`
2. Create participant-specific region-of-interest masks

### Univariate analysis: `./univariate/`**
1. Run `univariate_analysis.py`
2. Run `group_level.ipynb` for group-level GLM and output maps/figures

### Representational similarity analysis: `./rsa/`
1. Create event-specific beta estimates
2. Run region-based RSA using atlas masks (see [masking](#Masking))
3. Compute group-level RSA statistics for cortical and striatal networks
