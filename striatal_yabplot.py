import math

from typing import Dict, Optional, Sequence, Tuple

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


def _default_layout(n_views: int) -> Tuple[int, int]:
    """Mirror yabplot.plotting's own default (nrows, ncols) so per-view
    subtitles can be positioned at the same grid cells it renders into."""
    if n_views <= 1:
        return (1, 1)
    elif n_views <= 4:
        return (1, n_views)
    elif n_views <= 6:
        return (2, 3)
    else:
        return (math.ceil(n_views / 4), 4)


def plot_striatal_roi_stat_yabplot(
    stat_dict: Dict[str, float],
    atlas_dir: str,
    cmap: str = 'coolwarm',
    vlim: Optional[float] = None,
    views: Sequence[str] = ('left_lateral', 'right_lateral', 'superior', 'anterior'),
    layout: Optional[Tuple[int, int]] = None,
    zoom: float = 1.5,
    cbar_label: str = 'group t-statistic',
    title: Optional[str] = None,
):
    """Render per-ROI group statistics on a custom yabplot subcortical atlas.

    `stat_dict` keys must match the region names the atlas was built with
    (e.g. 'aCAU-lh') -- the same region_hemi naming already used for
    striatal_roi_plotting.plot_striatal_roi_stat, so the same stat_dict
    can be passed to either function.

    yabplot renders all views into a single flattened image (via PyVista)
    with no built-in per-panel titles or matplotlib-style subplot spacing,
    so `zoom` (camera zoom-in, tightens empty space around each ROI) and
    the per-view subtitles below are the available levers for that.
    """
    if vlim is None:
        vlim = max(abs(v) for v in stat_dict.values())

    views = list(views)
    nrows, ncols = layout if layout is not None else _default_layout(len(views))

    ax = yab.plot_subcortical(
        data=stat_dict,
        custom_atlas_path=atlas_dir,
        cmap=cmap,
        vminmax=[-vlim, vlim],
        nan_alpha=0.1,
        views=views,
        layout=(nrows, ncols),
        zoom=zoom,
        cbar_kwargs={'label': cbar_label},
    )

    for i, view in enumerate(views):
        row, col = divmod(i, ncols)
        x = (col + 0.5) / ncols
        y = 1.0 - row / nrows
        ax.text(x, y + 0.01, view.replace('_', ' '), transform=ax.transAxes,
                ha='center', va='bottom', fontsize=9)

    fig = ax.figure if hasattr(ax, 'figure') else ax
    if title:
        fig.suptitle(title)

    return fig
