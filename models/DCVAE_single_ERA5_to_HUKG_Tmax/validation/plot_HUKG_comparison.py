# Functions to plot haduk-grid before and after autoencoding
#  Takes data in tensorflow format (no geometry metadata, normalised)

import os
import sys
import iris
import numpy as np
import tensorflow as tf
import matplotlib
import cmocean

sys.path.append(
    "%s/../../../data/prepare_training_tensors_HUKG_Tmax/" % os.path.dirname(__file__)
)
from HUKG_load_tmax import HUKG_trim

# It's a spatial map, so want the land mask
def get_land_mask():
    mask = iris.load_cube(
        "%s/fixed_fields/land_mask/HadUKG_land_from_Copernicus.nc"
        % os.getenv("DATADIR")
    )
    return HUKG_trim(mask)


def plot_Tmax(
    ax,
    tmx,
    vMin=0,
    vMax=1,
    obs=None,
    o_size=1,
    land=None,
    mask=None,
    label=None,
):
    if land is None:
        land = get_land_mask()
    lats = land.coord("projection_y_coordinate").points
    lons = land.coord("projection_x_coordinate").points
    land_img = ax.pcolorfast(
        lons, lats, land.data, cmap="Greys", alpha=1.0, vmax=1.2, vmin=-0.5, zorder=10
    )

    pdata = tf.squeeze(tmx).numpy()
    if mask is not None:
        pdata[mask] = 0
    pdata = np.ma.masked_where(land.data == 0, pdata)

    T_img = ax.pcolorfast(
        lons,
        lats,
        pdata,
        cmap=cmocean.cm.balance,
        vmin=vMin,
        vmax=vMax,
        alpha=1.0,
        zorder=40,
    )
    if obs is not None:
        obs = tf.squeeze(obs)
        x = (obs[:, 1].numpy() / 896) * (lons[-1] - lons[0]) + lons[0]
        y = (obs[:, 0].numpy() / 1440) * (lats[-1] - lats[0]) + lats[0]
        ax.scatter(
            x,  # ((x/2).astype(int)+1)*2,
            y,  # ((y/2).astype(int)+1)*2,
            s=3.0 * o_size,
            c="black",
            marker="o",
            alpha=1.0,
            zorder=60,
        )
    if label is not None:
        ax.text(
            lons[0] + (lons[-1] - lons[0]) * 0.03,
            lats[0] + (lats[-1] - lats[0]) * 0.02,
            label,
            horizontalalignment="left",
            verticalalignment="bottom",
            color="black",
            bbox=dict(
                facecolor=(0.8, 0.8, 0.8, 0.8),
                edgecolor="black",
                boxstyle="round",
                pad=0.5,
            ),
            size=16,
            clip_on=True,
            zorder=100,
        )
    return T_img


def plot_scatter(
    ax, t_in, t_out, land=None, d_max=5, d_min=-5, xlab="Original", ylab="Encoded"
):
    x = (t_in.numpy().flatten() - 0.5) * 10
    y = (t_out.numpy().flatten() - 0.5) * 10
    #    if land is not None:
    #        ld = land.data.flatten
    y = y[x != 0]
    x = x[x != 0]
    ax.hexbin(
        x=x,
        y=y,
        cmap=cmocean.cm.ice_r,
        bins="log",
        mincnt=1,
        extent=(d_min,d_max,d_min,d_max),
        zorder=50
    )
    ax.add_line(
        matplotlib.lines.Line2D(
            xdata=(d_min, d_max),
            ydata=(d_min, d_max),
            linestyle="solid",
            linewidth=1.5,
            color=(0.5, 0.5, 0.5, 1),
            zorder=100,
        )
    )
    ax.set(xlabel=xlab, ylabel=ylab)
    ax.grid(color="black", alpha=0.2, linestyle="-", linewidth=0.5)


def plot_colourbar(
    fig,
    ax,
    T_img,
):
    ax.set_axis_off()
    cb = fig.colorbar(
        T_img, ax=ax, location="bottom", orientation="horizontal", fraction=1.0
    )
