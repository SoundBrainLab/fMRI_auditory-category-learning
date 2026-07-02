#!/usr/bin/env python
"""
Build a yabplot custom subcortical atlas from the Tian Subcortex S2 atlas.

Run once, in a Python 3.11 environment with yabplot installed
(`pip install yabplot`), on the cluster where the reference atlas file
lives. The output directory (small .vtk mesh files) can then be synced to
any machine for plotting with yab.plot_subcortical(custom_atlas_path=...)
-- yabplot itself is not needed again after that.

Region id/name mapping is copied from `roi_dict_tian_S2` in
make_atlas_region_masks.py (that script can't be imported directly here
since it parses CLI args and exits at import time if none are given).
"""
import os
import yabplot as yab

TIAN_S2_NII = ('/bgfs/bchandrasekaran/krs228/data/reference/subcortex/'
               'Group-Parcellation/7T/Tian_Subcortex_S2_7T.nii')
OUT_DIR = os.path.join(os.path.dirname(__file__), 'yabplot_atlases', 'tian_S2')

# Same order as tian_sc_S2_roi_list in make_atlas_region_masks.py -> id = index + 1
TIAN_S2_ROI_LIST = [
    'HIP-head-rh', 'HIP-body-rh', 'HIP-tail-rh', 'lAMY-rh', 'mAMY-rh',
    'THA-DP-rh', 'THA-VP-rh', 'THA-VA-rh', 'THA-DA-rh', 'pGP-rh', 'aGP-rh',
    'NAc-shell-rh', 'NAc-core-rh', 'aPUT-rh', 'pPUT-rh', 'aCAU-rh', 'pCAU-rh',
    'HIP-head-lh', 'HIP-body-lh', 'HIP-tail-lh', 'lAMY-lh', 'mAMY-lh',
    'THA-DP-lh', 'THA-VP-lh', 'THA-VA-lh', 'THA-DA-lh', 'pGP-lh', 'aGP-lh',
    'NAc-shell-lh', 'NAc-core-lh', 'aPUT-lh', 'pPUT-lh', 'aCAU-lh', 'pCAU-lh',
]
LABELS_DICT = {i + 1: roi for i, roi in enumerate(TIAN_S2_ROI_LIST)}

# Only build meshes for the structures actually used in group_level_all_ROI.ipynb
# (aCAU/pCAU/aPUT/pPUT/NAc-core/NAc-shell x lh/rh). Drop this to build all 34.
INCLUDE_KEYWORDS = ['CAU', 'PUT', 'NAc']

if __name__ == '__main__':
    os.makedirs(OUT_DIR, exist_ok=True)
    yab.build_subcortical_atlas(
        nii_path=TIAN_S2_NII,
        labels_dict=LABELS_DICT,
        out_dir=OUT_DIR,
        include_list=INCLUDE_KEYWORDS,
        smooth_i=20,
        smooth_f=0.7,
    )

    regions = yab.get_atlas_regions(atlas=None, category='subcortical',
                                     custom_atlas_path=OUT_DIR)
    print(f'Built {len(regions)} region meshes at {OUT_DIR}: {regions}')
