import argparse
import os
import sys
from glob import glob

import nibabel as nib
import numpy as np
import pandas as pd
from nilearn.image import resample_to_img, binarize_img
from nilearn.masking import apply_mask

''' Extract per-ROI mean first-level effects for every participant, contrast,
network and learning stage, and write one long table per network x contrast.

Replaces the hand-toggled "ONLY RUN ONCE" masking cells in
group_level_all_ROI.ipynb (one of which once left a stale n=11 table on disk,
REVISION_PLAN.md #27). Differences from those cells:
  - participants come from participants.tsv (both language groups) rather
    than a hard-coded list, and every output row carries `participant_group`
  - a missing stat map or mask is reported (and written to
    `roi-values_missing.tsv`) instead of being skipped silently or crashing
    the whole loop
  - each mask is resampled once per participant rather than once per map,
    since all of a participant's first-level maps share one grid (checked)

The per-ROI value is computed exactly as in the notebook (nearest-neighbour
resample of the mask onto the stat map, binarize, mean of the masked
effect-size values), so filtering the output to the non-Mandarin group
should reproduce the notebook's tables.

Output schema matches the notebook's univariate-results_*.tsv files:
participant_id, participant_group, region, beta, region_hemi, hemisphere,
learning_stage -- with hemisphere as 'L'/'R' for cortical networks and
'lh'/'rh' for Tian-S2, as before.

Example: python extract_roi_values.py \
    --bidsroot=/ix1/bchandrasekaran/krs228/data/FLT/data_denoised/ \
    --variant_tag=scrubbed-splitfb
'''

# region lists and mask-folder names copied from group_level_all_ROI.ipynb --
# keep in sync. ParsOp/ParsTri (auditory) and IFGo/IFGt (pfc) are the same
# atlas labels; both are kept here so tables match the notebook, and should
# be de-duplicated at the stats stage
REGION_DICT = {
    'auditory': ['L-HG', 'L-PT', 'L-PP', 'L-STGp', 'L-STGa', 'L-ParsOp', 'L-ParsTri',
                 'R-HG', 'R-PT', 'R-PP', 'R-STGp', 'R-STGa', 'R-ParsOp', 'R-ParsTri'],
    'tian-S2': ['aCAU-lh', 'pCAU-lh', 'aPUT-lh', 'pPUT-lh', 'NAc-shell-lh', 'NAc-core-lh',
                'aCAU-rh', 'pCAU-rh', 'aPUT-rh', 'pPUT-rh', 'NAc-shell-rh', 'NAc-core-rh'],
    'pfc': ['L-FP', 'R-FP', 'L-SFG', 'R-SFG', 'L-MFG', 'R-MFG', 'L-IFGt', 'R-IFGt',
            'L-IFGo', 'R-IFGo', 'L-PreCG', 'R-PreCG', 'L-FMC', 'R-FMC', 'L-PCG', 'R-PCG',
            'L-ACC', 'R-ACC', 'L-OFC', 'R-OFC', 'L-FO', 'R-FO'],
}
MASK_DIR_NAME = {'auditory': 'dseg', 'pfc': 'dseg-pfc', 'tian-S2': 'tian-S2'}
STAGES = ['early', 'middle', 'final']
CONTRASTS = ['sound', 'resp', 'fbcorrectvswrong', 'fbcorrect']
IGNORE_SUBS = ['sub-FLT27',  # did not participate
               'sub-FLT07']  # bad QA 11/14/23


def split_region_hemi(region_hemi, network):
    ''' 'L-HG' -> ('HG', 'L'); 'aCAU-lh' -> ('aCAU', 'lh') '''
    if network == 'tian-S2':
        return region_hemi[:-3], region_hemi[-2:]
    return region_hemi[2:], region_hemi[0]


def stat_map_path(l1_dir, sub_id, space, stage, contrast):
    ''' the one effect-size map for this contrast, or None. Matched on the
    `contrast-{id}_` entity (not the full save_glm_to_bids filename) -- the
    trailing underscore keeps `fbcorrect` from matching `fbcorrectvswrong` '''
    pattern = os.path.join(l1_dir, f'{sub_id}_space-{space}', f'{stage}_event-all', sub_id,
                           f'*_contrast-{contrast}_stat-effect_statmap.nii.gz')
    matches = glob(pattern)
    if len(matches) > 1:
        raise ValueError(f'more than one map matches {pattern}: {matches}')
    return matches[0] if matches else None


def mask_path(masks_dir, sub_id, space, network, region_hemi):
    return os.path.join(masks_dir, sub_id, f'space-{space}', f'masks-{MASK_DIR_NAME[network]}',
                        f'{sub_id}_space-{space}_mask-{region_hemi}.nii.gz')


def extract_subject(sub_id, group, l1_dir, masks_dir, space, networks, contrasts):
    ''' all ROI means for one participant, plus a list of what was missing '''
    rows, missing = [], []

    stat_paths = {}
    for stage in STAGES:
        for contrast in contrasts:
            fpath = stat_map_path(l1_dir, sub_id, space, stage, contrast)
            if fpath:
                stat_paths[(stage, contrast)] = fpath
            else:
                missing.append({'participant_id': sub_id, 'kind': 'stat_map',
                                'stage': stage, 'contrast': contrast,
                                'path': os.path.join(l1_dir, f'{sub_id}_space-{space}', f'{stage}_event-all')})
    if not stat_paths:
        return rows, missing

    # a participant's first-level maps normally all share one functional
    # grid, so each mask is resampled once per distinct grid (usually one)
    # rather than once per map -- same values as the notebook either way
    grid_of = {}
    for key, fpath in stat_paths.items():
        img = nib.load(fpath)
        grid_of[key] = (img.shape[:3], tuple(np.round(img.affine, 4).ravel()))
    grid_ref = {}
    for key, grid in grid_of.items():
        grid_ref.setdefault(grid, stat_paths[key])
    if len(grid_ref) > 1:
        print(f'  note: {sub_id} maps span {len(grid_ref)} grids; resampling masks per grid')

    for network in networks:
        for region_hemi in REGION_DICT[network]:
            mpath = mask_path(masks_dir, sub_id, space, network, region_hemi)
            if not os.path.exists(mpath):
                missing.append({'participant_id': sub_id, 'kind': 'mask', 'network': network,
                                'region_hemi': region_hemi, 'path': mpath})
                continue
            mask_by_grid = {}
            for grid, ref_path in grid_ref.items():
                mask_img = resample_to_img(mpath, ref_path, interpolation='nearest',
                                           force_resample=True, copy_header=True)
                mask_by_grid[grid] = binarize_img(mask_img, two_sided=False, copy_header=True)

            region, hemisphere = split_region_hemi(region_hemi, network)
            for (stage, contrast), fpath in stat_paths.items():
                mask_img = mask_by_grid[grid_of[(stage, contrast)]]
                rows.append({'network': network, 'contrast': contrast,
                             'participant_id': sub_id, 'participant_group': group,
                             'region': region, 'beta': apply_mask(fpath, mask_img).mean(),
                             'region_hemi': region_hemi, 'hemisphere': hemisphere,
                             'learning_stage': stage})
    return rows, missing


def main(argv=None):
    parser = argparse.ArgumentParser(description='Extract per-ROI first-level effects for all participants')
    parser.add_argument('--bidsroot', required=True, type=str)
    parser.add_argument('--fmriprep_tag', default='fmriprep-25.2.5', type=str)
    parser.add_argument('--variant_tag', default='scrubbed-splitfb', type=str,
                        help="first-level model folder (see univariate_glm.py's _variant_tag)")
    parser.add_argument('--space', default='T1w', type=str)
    parser.add_argument('--fwhm', default=0.0, type=float)
    parser.add_argument('--networks', nargs='+', default=list(REGION_DICT), choices=list(REGION_DICT))
    parser.add_argument('--contrasts', nargs='+', default=CONTRASTS)
    parser.add_argument('--out_dir', default=None, type=str,
                        help='default: derivatives/nilearn/group_space-{space}_fwhm-{fwhm}/roi-values_{variant_tag}/')
    args = parser.parse_args(argv)

    nilearn_dir = os.path.join(args.bidsroot, 'derivatives', 'nilearn')
    l1_dir = os.path.join(nilearn_dir, args.fmriprep_tag, args.variant_tag,
                          'bids-deriv_level-1_fwhm-%.02f' % args.fwhm)
    masks_dir = os.path.join(nilearn_dir, 'masks')
    out_dir = args.out_dir or os.path.join(nilearn_dir, 'group_space-%s_fwhm-%.02f' % (args.space, args.fwhm),
                                           f'roi-values_{args.variant_tag}')
    os.makedirs(out_dir, exist_ok=True)

    participants = pd.read_csv(os.path.join(args.bidsroot, 'participants.tsv'), sep='\t')
    participants = (participants[~participants.participant_id.isin(IGNORE_SUBS)]
                    .sort_values('participant_id', ignore_index=True))
    print(participants.groupby('group').participant_id.count().to_string())

    all_rows, all_missing = [], []
    for sub_id, group in zip(participants.participant_id, participants.group):
        print(f'extracting {sub_id} ({group})', flush=True)
        rows, missing = extract_subject(sub_id, group, l1_dir, masks_dir, args.space,
                                        args.networks, args.contrasts)
        all_rows += rows
        all_missing += missing
        for m in missing:
            print(f'  MISSING {m["kind"]}: {m["path"]}')

    missing_fpath = os.path.join(out_dir, 'roi-values_missing.tsv')
    pd.DataFrame(all_missing).to_csv(missing_fpath, sep='\t', index=False)
    print(f'{len(all_missing)} missing inputs -> {missing_fpath}')
    if not all_rows:
        print(f'ERROR: no ROI values extracted -- check {l1_dir} exists')
        return 1

    df = pd.DataFrame(all_rows)
    columns =['participant_id', 'participant_group', 'region', 'beta',
               'region_hemi', 'hemisphere', 'learning_stage']
    for (network, contrast), table in df.groupby(['network', 'contrast']):
        fpath = os.path.join(out_dir, f'roi-values_network-{network}_contrast-{contrast}.tsv')
        table[columns].to_csv(fpath, sep='\t', index=False)
        counts = table.groupby('participant_group').participant_id.nunique().to_dict()
        print(f'wrote {fpath} (participants per group: {counts})')
    return 0


if __name__ == '__main__':
    sys.exit(main())
