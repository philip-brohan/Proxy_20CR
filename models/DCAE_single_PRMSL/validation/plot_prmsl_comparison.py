# Functions to plot PRMSL before and after autoencoding
#  Takes data in tensorflow format (no geometry metadata, normalised)

import os
import sys
import iris
import numpy as np
import tensorflow as tf

sys.path.append("%s/../../../lib/" % os.path.dirname(__file__))
from geometry import to_analysis_grid

# It's a spatial map, so want the land mask
def get_land_mask():
    mask = iris.load_cube(
        "%s/fixed_fields/land_mask/opfc_global_2019.nc" % os.getenv("DATADIR")
    )
    return to_analysis_grid(mask)


def plot_PRMSL(ax, t_in, t_out, land=None, label=None):
    if land is None:
        land = get_land_mask()
    lats = land.coord("latitude").points
    lons = land.coord("longitude").points
    land_img = ax.pcolorfast(
        lons, lats, land.data, cmap="Greys", alpha=0.3, vmax=1.2, vmin=-0.5, zorder=10
    )
    # 20CR2c data
    if t_in is not None:
        CS = ax.contour(
            lons,
            lats,
            t_in.numpy(),
            colors="red",
            linewidths=1.0,
            alpha=1.0,
            levels=np.arange(-3, 3, 0.3),
            zorder=20,
        )
    # Encoder output
    if t_out is not None:
        CS = ax.contour(
            lons,
            lats,
            t_out.numpy(),
            colors="blue",
            linewidths=1.0,
            alpha=1.0,
            # levels=numpy.arange(870,1050,10),
            zorder=30,
        )
    ax.text(
        -175,
        -85,
        label,
        horizontalalignment="left",
        verticalalignment="bottom",
        color="black",
        bbox=dict(
            facecolor=(0.8, 0.8, 0.8, 0.8), edgecolor="black", boxstyle="round", pad=0.5
        ),
        size=8,
        clip_on=True,
        zorder=40,
    )
