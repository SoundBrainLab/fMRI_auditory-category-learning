import nibabel as nib
import numpy as np
import matplotlib.pyplot as plt

from matplotlib.path import Path
from matplotlib.patches import PathPatch
from matplotlib.cm import ScalarMappable
from skimage import measure
from typing import Dict, List, Optional

"""
Render per-ROI group statistics as colored contours on a representative
sagittal slice per striatal structure/hemisphere, adapted from the
corticostriatal schematic (`draw_striatal_structure`) in
hcp7t_mrtrix3_TianS2's connectivity_statistics.ipynb.

Usage:
    from striatal_roi_plotting import plot_striatal_roi_stat

    # region_hemi name (e.g. 'aCAU-lh') -> t-value
    stat_dict = {'aCAU-lh': 2.1, 'pCAU-lh': -1.4, 'aCAU-rh': 1.8, 'pCAU-rh': -0.9}
    # region_hemi name -> path to that ROI's volumetric mask
    mask_path_dict = {'aCAU-lh': '/path/aCAU-lh.nii.gz', ...}
    # display name -> ordered list of base ROI names (without hemisphere suffix)
    # that make up that anatomical structure
    structure_groups = {'Caudate': ['aCAU', 'pCAU'], 'Putamen': ['aPUT', 'pPUT']}

    fig = plot_striatal_roi_stat(stat_dict, mask_path_dict, structure_groups,
                                  title='sound vs. baseline')
    fig.savefig('network-tian-S2_contrast-sound_striatal-tstat.png')
"""

_HEMI_SUFFIX = {'lh': 'left', 'rh': 'right'}


def _hemisphere_for_striatal_region(region_hemi: str) -> str:
    suffix = region_hemi.rsplit('-', 1)[-1].lower()
    if suffix not in _HEMI_SUFFIX:
        raise ValueError(
            f"striatal region name '{region_hemi}' must end in '-lh' or '-rh' "
            "to indicate hemisphere"
        )
    return _HEMI_SUFFIX[suffix]


def _best_sagittal_slice(mask_data_list):
    """Sagittal (x) slice index maximizing the combined cross-sectional area
    of all masks in mask_data_list, so all subdivisions of a structure show
    up together (mirrors get_best_slice_idx in connectivity_statistics.ipynb)."""
    combined_area = sum(data.sum(axis=(1, 2)) for data in mask_data_list)
    return int(np.argmax(combined_area))


def _largest_contour(slice_2d):
    contours = measure.find_contours(slice_2d, level=0.5)
    if not contours:
        return None
    return max(contours, key=len)


def plot_striatal_roi_stat(
    stat_dict: Dict[str, float],
    mask_path_dict: Dict[str, str],
    structure_groups: Dict[str, List[str]],
    cmap: str = 'coolwarm',
    vlim: Optional[float] = None,
    title: Optional[str] = None,
):
    """Render per-ROI group statistics as colored contours on a representative
    sagittal slice per structure/hemisphere.

    Produces a grid with one row per structure (e.g. Caudate, Putamen, NAc)
    and one column per hemisphere. Each panel overlays that structure's
    subdivisions (e.g. anterior/posterior caudate) as filled, black-outlined
    contours colored by their stat value, with a single shared symmetric
    colorbar.
    """
    if vlim is None:
        vlim = max(abs(v) for v in stat_dict.values())
    vmin, vmax = -vlim, vlim
    norm = plt.Normalize(vmin=vmin, vmax=vmax)
    cmap_obj = plt.get_cmap(cmap)

    structures = list(structure_groups.keys())
    hemi_suffixes = ('lh', 'rh')

    fig, axes = plt.subplots(len(structures), len(hemi_suffixes),
                              figsize=(3 * len(hemi_suffixes), 3 * len(structures)),
                              squeeze=False, dpi=300)

    for row, structure in enumerate(structures):
        base_names = structure_groups[structure]
        for col, hemi_suffix in enumerate(hemi_suffixes):
            ax = axes[row][col]
            region_names = [f'{base}-{hemi_suffix}' for base in base_names]
            mask_data = {r: nib.load(mask_path_dict[r]).get_fdata()
                         for r in region_names if r in mask_path_dict}
            if not mask_data:
                ax.axis('off')
                continue

            slice_idx = _best_sagittal_slice(list(mask_data.values()))

            for region in region_names:
                if region not in mask_data:
                    continue
                contour = _largest_contour(mask_data[region][slice_idx, :, :])
                if contour is None:
                    continue
                verts = np.column_stack([contour[:, 0], contour[:, 1]])
                codes = [Path.MOVETO] + [Path.LINETO] * (len(verts) - 2) + [Path.CLOSEPOLY]
                stat_value = stat_dict.get(region, 0)
                ax.add_patch(PathPatch(Path(verts, codes),
                                        facecolor=cmap_obj(norm(stat_value)),
                                        edgecolor='black', linewidth=1))
                base_name = region.rsplit('-', 1)[0]
                ax.text(np.mean(contour[:, 0]), np.mean(contour[:, 1]), base_name,
                        ha='center', va='center', fontsize=6, color='black')

            ax.set_aspect('equal')
            ax.autoscale()
            margin = 3
            ax.set_xlim(ax.get_xlim()[0] - margin, ax.get_xlim()[1] + margin)
            ax.set_ylim(ax.get_ylim()[0] - margin, ax.get_ylim()[1] + margin)
            ax.invert_xaxis()
            ax.axis('off')

            if row == 0:
                ax.set_title(_HEMI_SUFFIX[hemi_suffix], fontsize=10)

        axes[row][0].annotate(structure, xy=(0, 0.5), xycoords='axes fraction',
                               xytext=(-10, 0), textcoords='offset points',
                               ha='right', va='center', fontsize=10, rotation=90)

    sm = ScalarMappable(cmap=cmap_obj, norm=norm)
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=axes, orientation='vertical',
                         fraction=0.03, pad=0.05, shrink=0.6)
    cbar.set_label('group t-statistic')

    if title:
        fig.suptitle(title)

    return fig
