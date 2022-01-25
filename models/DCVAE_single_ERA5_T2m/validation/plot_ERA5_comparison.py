# Functions to plot ERA5 data before and after autoencoding
#  Takes data in tensorflow format (no geometry metadata, normalised)

import os
import sys
import math

# import iris
import numpy as np
import tensorflow as tf
import matplotlib
import cmocean

sys.path.append(
    "%s/../../../data/prepare_training_tensors_ERA5_T2m/" % os.path.dirname(__file__)
)
from ERA5_load import ERA5_trim
from ERA5_load import ERA5_roll_longitude
from ERA5_load import ERA5_load_LS_mask

# It's a spatial map, so want the land mask
def get_land_mask():
    mask = ERA5_load_LS_mask()
    mask = ERA5_roll_longitude(mask)
    return ERA5_trim(mask)


def plot_T2m(
    ax,
    tmx,
    vMin=0,
    vMax=1,
    fog=None,
    fog_threshold=0.5,
    fog_steepness=10,
    obs=None,
    obs_c=None,
    o_size=1,
    land=None,
    label=None,
):
    if land is None:
        land = get_land_mask()
    lats = land.coord("latitude").points
    lons = land.coord("longitude").points
    land_img = ax.pcolorfast(
        lons, lats, land.data, cmap="Greys", alpha=0.1, vmax=1.1, vmin=-0.5, zorder=100
    )
    # Field data
    T_img = None
    if tmx is not None:
        T_img = ax.pcolormesh(
            lons,
            lats,
            tf.squeeze(tmx).numpy(),
            shading="auto",
            cmap="RdYlBu_r",  # cmocean.cm.balance, #"RdYlBu_r",
            vmin=vMin,
            vmax=vMax,
            alpha=1.0,
            zorder=40,
        )
        # Fog of ignorance
        if fog is not None:
            cs = ax.contourf(
                lons,
                lats,
                np.minimum(1.0, tf.squeeze(fog).numpy()),
                [0, fog_threshold, 1],
                colors="none",
                vmin=0,
                vmax=1,
                hatches=[None, "///"],
                extend="upper",
                zorder=500,
            )
            for i, collection in enumerate(cs.collections):
                collection.set_edgecolor((0.8, 0.8, 0.8))
            for collection in cs.collections:
                collection.set_linewidth(0.0)
            cs = ax.contourf(
                lons,
                lats,
                np.minimum(1.0, tf.squeeze(fog).numpy()),
                [0, fog_threshold, 1],
                colors="none",
                vmin=0,
                vmax=1,
                hatches=[None, "\\\\\\"],
                extend="upper",
                zorder=500,
            )
            for i, collection in enumerate(cs.collections):
                collection.set_edgecolor((0.8, 0.8, 0.8))
            for collection in cs.collections:
                collection.set_linewidth(0.0)

    # Observations
    if obs is not None:
        obs = tf.squeeze(obs)
        x = (obs[:, 1].numpy() / 1440) * 360 - 180
        y = (obs[:, 0].numpy() / 720) * 180 - 90
        y *= -1
        if obs_c is None:
            ax.scatter(
                ((x / 2).astype(int) + 1) * 2,
                ((y / 2).astype(int) + 1) * 2,
                s=3.0 * o_size,
                c="black",
                marker="o",
                alpha=0.8,
                zorder=600,
            )
        else:
            ax.scatter(
                ((x / 2).astype(int) + 1) * 2,
                ((y / 2).astype(int) + 1) * 2,
                s=3.0 * o_size,
                c=obs_c.numpy(),
                cmap="RdYlBu_r",  # cmocean.cm.balance, #"RdYlBu_r",
                vmin=vMin,
                vmax=vMax,
                marker="o",
                alpha=1.0,
                zorder=600,
            )

    if label is not None:
        ax.text(
            lons[0] + (lons[-1] - lons[0]) * 0.02,
            lats[0] + (lats[-1] - lats[0]) * 0.04,
            label,
            horizontalalignment="left",
            verticalalignment="top",
            color="black",
            bbox=dict(
                facecolor=(0.8, 0.8, 0.8, 0.8),
                edgecolor="black",
                boxstyle="round",
                pad=0.5,
            ),
            size=matplotlib.rcParams["font.size"] / 1.5,
            clip_on=True,
            zorder=100,
        )
    return T_img


def plot_scatter(ax, t_in, t_out, land=None, d_max=5, d_min=-5):
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
    )
    ax.add_line(
        matplotlib.lines.Line2D(
            xdata=(d_min, d_max),
            ydata=(d_min, d_max),
            linestyle="solid",
            linewidth=0.5,
            color=(0.5, 0.5, 0.5, 1),
            zorder=100,
        )
    )
    ax.set(xlabel="Original", ylabel="Encoded")
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
