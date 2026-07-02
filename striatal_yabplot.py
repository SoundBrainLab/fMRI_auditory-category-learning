from typing import Dict, Optional, Sequence

import yabplot as yab

"""
Plot striatal (Tian S2) ROI statistics with yabplot's 3D subcortical
rendering, as an alternative to striatal_roi_plotting.py's 2D contour
schematic.

Requires yabplot (`pip install yabplot`, needs Python 3.11) and a
pre-built custom Tian S2 atlas -- see build_yabplot_tian_s2_atlas.py in
05_masking/, which must be run once (on the cluster, where the reference
Tian atlas file lives) before this module is usable.

Usage:
    from striatal_yabplot import plot_striatal_roi_stat_yabplot

    stat_dict = {'aCAU-lh': 1.8, 'pCAU-lh': -1.2, ...}
    fig = plot_striatal_roi_stat_yabplot(
        stat_dict, atlas_dir='05_masking/yabplot_atlases/tian_S2',
        title='sound vs. baseline')
"""


def plot_striatal_roi_stat_yabplot(
    stat_dict: Dict[str, float],
    atlas_dir: str,
    cmap: str = 'coolwarm',
    vlim: Optional[float] = None,
    views: Sequence[str] = ('left_lateral', 'right_lateral', 'superior', 'anterior'),
    title: Optional[str] = None,
):
    """Render per-ROI group statistics on a custom yabplot subcortical atlas.

    `stat_dict` keys must match the region names the atlas was built with
    (e.g. 'aCAU-lh') -- the same region_hemi naming already used for
    striatal_roi_plotting.plot_striatal_roi_stat, so the same stat_dict
    can be passed to either function.
    """
    if vlim is None:
        vlim = max(abs(v) for v in stat_dict.values())

    ax = yab.plot_subcortical(
        data=stat_dict,
        custom_atlas_path=atlas_dir,
        cmap=cmap,
        vminmax=[-vlim, vlim],
        nan_alpha=0.1,
        views=list(views),
    )

    fig = ax.figure if hasattr(ax, 'figure') else ax
    if title:
        fig.suptitle(title)

    return fig
