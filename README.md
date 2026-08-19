# Auditory category learning in striatum and cortex using 7T functional MRI

Processing and analyzing tone-learning fMRI data collected at the University of Pittsburgh's 7T MRI Center.

## Manuscript details
Currently in revision. Preprint to come shortly!

## Data availability
Data will be uploaded to OpenNeuro.

## Environment
Python dependencies are tracked in `environment.yml` (conda). This pipeline only
runs on the Pitt CRC cluster; external tools (FreeSurfer, FSL, AFNI, ANTs,
MRtrix3, Singularity) are loaded via `module add` on the cluster rather than
managed through conda.

## Processing pipeline

### Dicom conversion: `./01_dicom_conversion/`
1. Peek at the dicom .tsv file  using `initialize_dicoms_heudiconv.sh`
2. Create `heuristic.py` based on your MRI sequences
3. Convert dicoms to .nii using `convert_dicoms_heudiconv.sh`

### Image denoising: `./02_denoising/`
1. Run `dwi_denoise` on newly converted BIDS-formatted NIfTI files

### MRI preprocessing: `./03_fmriprep/`
1. Preprocess anatomical and functional MRI with `run_fmriprep_denoised.sh`
   (fMRIPrep 22.1.1 -- the version used for the originally submitted manuscript)
   or `run_fmriprep_denoised_25.2.5.sh` (fMRIPrep 25.2.5, current LTS; used for
   the revision going forward). Each writes to its own versioned derivatives
   directory (`derivatives/denoised_fmriprep-<version>/`) so both coexist.
> (Note: these run using a Singularity image, so may need to build that first --
> see the commented `singularity build` line at the top of each script)

### Behavior Behavioral data conversion: `./04_behavior/`
1. Run `convert_behav_to_bids.py` to get psychopy outputs into BIDS-compatible format
2. Run behavioral analysis notebook

### Masking: `./05_masking/`
1. Create grey matter mask for searchlight using `make_gm_mask.py`
2. Create participant-specific region-of-interest masks
3. For the 25.2.5-derivatives pipeline, ROI stats are computed in native T1w
   space rather than MNI152NLin2009cAsym (avoids interpolation/partial-volume
   blur on small subcortical ROIs). Run `warp_atlases_to_T1w.sh` first to warp
   the Tian S2 striatal parcellation, subcortical auditory-pathway atlas, and
   `carpet_dseg` atlas into each subject's native space, then
   `make_atlas_region_masks.py --space=T1w`. Whole-brain/gradient figures
   still use the MNI-space pipeline, since those need a shared voxel grid
   across subjects. Tian S3 has been retired in favor of Tian S2.

### Univariate analysis: `./06_univariate/`
1. Run `univariate_glm.py` (one consolidated script for subject-level GLMs,
   replacing the three previous near-duplicate scripts). Confounds are loaded
   via `nilearn.interfaces.fmriprep.load_confounds_strategy` with the
   `scrubbing` strategy (FD 0.9mm / DVARS 1.5) rather than a fixed
   motion-only regressor list; every run is kept regardless of how much data
   survives scrubbing (n=12 means every run counts), but per-run
   volumes-retained stats are logged to
   `derivatives/nilearn/qc/sub-*_confound-scrubbing-qc.csv` for review.
   `--grouping=none` fits all runs together in one GLM (see
   `run_univariate_analysis_denoised_fb.sh`); `--grouping=grouped` fits
   separate GLMs per early/middle/late run-pair to look at learning-stage
   effects (see `run_univariate_groupedruns_denoised.sh`).
2. Run `group_level.ipynb` for group-level GLM and output maps/figures
3. `robustness_checks.py` runs leave-one-subject-out, exact/Monte Carlo
   permutation, and bootstrap-CI checks on the anterior caudate vs. putamen
   feedback learning-stage effect -- the paper's headline claim, checked
   deliberately narrowly rather than across every ROI. Reads the
   tab-separated `univariate-results_network-tian-S2_contrast-fb-correct-vs-wrong.tsv`
   that `group_level_all_ROI.ipynb` already writes to
   `derivatives/nilearn/group_fwhm-0.00/` -- no new export needed, just
   `--sep=tab`. That section of the notebook restricts to non-Mandarin
   participants (`sub_list_nman`), so double check the actual n once real
   data is available; the script doesn't assume n=12.

### Representational similarity analysis: `./07_rsa/`
1. Create event-specific beta estimates
2. Run region-based RSA using atlas masks (see [masking](#Masking))
3. Compute group-level RSA statistics for cortical and striatal networks

### Behavioral modeling: `./08_behavioral-modeling/`
The SPC/decision-bound model-fitting code itself lives outside this repo; this
directory works from its output. `model_comparison.py` runs the formal
model comparison Reviewer #1 asked for -- were the alternative models (UDX,
UDY, CON, SPC, RAN; CJH1/CJH2 excluded, see script docstring) ever
statistically compared, and does SPC's fit specifically improve relative to
the best competitor from early to late learning stage. Reads the subject-run
BIC results file directly (no new fitting needed) and restricts to the
non-Mandarin analyzed sample via `participants.tsv`. All paths are required
CLI arguments -- nothing is hardcoded, since the results file lives on the
cluster.

## Shared utilities
`stats_fmt.py` (statistics-table formatting) and `resampling_stats.py`
(leave-one-out/permutation/bootstrap checks, used by both
`06_univariate/robustness_checks.py` and
`08_behavioral-modeling/model_comparison.py`) live at the repo root and are
imported directly by scripts/notebooks in the numbered subdirectories.
