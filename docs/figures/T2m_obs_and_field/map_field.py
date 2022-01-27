#!/usr/bin/env python

import os
import sys
import numpy as np

import matplotlib
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
from matplotlib.figure import Figure
import cmocean

import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--year", type=int, required=False, default=1979)
parser.add_argument("--month", type=int, required=False, default=3)
parser.add_argument("--day", type=int, required=False, default=12)
args = parser.parse_args()

sys.path.append(
    "%s/../../../data/prepare_training_tensors_ERA5_T2m" % os.path.dirname(__file__)
)
from ERA5_load import ERA5_load_T2m
from ERA5_load import ERA5_load_T2m_climatology
from ERA5_load import ERA5_roll_longitude
from ERA5_load import ERA5_load_LS_mask

mask = ERA5_load_LS_mask()
mask = ERA5_roll_longitude(mask)

t = ERA5_load_T2m(args.year, args.month, args.day)
c = ERA5_load_T2m_climatology(args.year, args.month, args.day)
a = t - c
a = ERA5_roll_longitude(a)

sys.path.append(
    "%s/../../../models/DCVAE_single_ERA5_T2m/validation" % os.path.dirname(__file__)
)
from plot_ERA5_comparison import get_land_mask


def plot_T2m(
    ax,
    tmx,
    vMin=0,
    vMax=1,
    fog=None,
    fog_threshold=0.5,
    fog_steepness=10,
    obs=None,
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

    if fog is not None:
        mtmx = np.ma.masked_where(fog.data > fog_threshold, tmx.data)
        T_img = ax.pcolormesh(
            lons,
            lats,
            mtmx,
            shading="auto",
            cmap="RdYlBu_r",  # cmocean.cm.balance, #"RdYlBu_r",
            vmin=vMin,
            vmax=vMax,
            alpha=1.0,
            zorder=40,
        )
    else:
        T_img = ax.pcolormesh(
            lons,
            lats,
            tmx.data,
            shading="auto",
            cmap="RdYlBu_r",  # cmocean.cm.balance, #"RdYlBu_r",
            vmin=vMin,
            vmax=vMax,
            alpha=1.0,
            zorder=40,
        )
    # Fog of ignorance
    #    if fog is not None:
    #        def fog_map(x):
    #            return 1/(1+math.exp((x-fog_threshold)*fog_steepness*-1))
    #        cols=[]
    #        for ci in range(100):
    #            cols.append([0.2,0.2,0.2,fog_map(ci/100)])
    #
    #        fog_img = ax.pcolorfast(lons, lats, tf.squeeze(fog).numpy(),
    #                                   cmap=matplotlib.colors.ListedColormap(cols),
    #                                   alpha=0.95,
    #                                   vmin=0,
    #                                   vmax=1,
    #                                   zorder=50)

    # Observations
    if obs is not None:
        obs = tf.squeeze(obs)
        x = (obs[:, 1].numpy() / 1440) * 360 - 180
        y = (obs[:, 0].numpy() / 720) * 180 - 90
        y *= -1
        ax.scatter(
            ((x / 2).astype(int) + 1) * 2,
            ((y / 2).astype(int) + 1) * 2,
            s=3.0 * o_size,
            c="black",
            marker="o",
            alpha=1.0,
            zorder=60,
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


fig = Figure(
    figsize=(20, 10),
    dpi=100,
    facecolor=(0.88, 0.88, 0.88, 1),
    edgecolor=None,
    linewidth=0.0,
    frameon=False,
    subplotpars=None,
    tight_layout=None,
)
canvas = FigureCanvas(fig)
matplotlib.rcParams["font.size"] = 24

ax_global = fig.add_axes([0, 0, 1, 1], facecolor="white")
ax_global.set_axis_off()

ax_of = fig.add_axes([0.01, 0.01, 0.98, 0.98])
ax_of.set_aspect("auto")
ax_of.set_axis_off()
ofp = plot_T2m(
    ax_of,
    a,
    vMin=-10,
    vMax=10,
    land=mask,
    label="",
)


fig.savefig("field.png")
