#!/usr/bin/env python

import os
import sys
import numpy as np

import matplotlib
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
from matplotlib.figure import Figure

import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--year", type=int, required=False, default=1979)
parser.add_argument("--month", type=int, required=False, default=3)
parser.add_argument("--day", type=int, required=False, default=12)
args = parser.parse_args()

sys.path.append("%s/.." % os.path.dirname(__file__))
from ERA5_load import ERA5_load_LS_mask
from ERA5_load import ERA5_load_T2m
from ERA5_load import ERA5_load_T2m_climatology
from ERA5_load import ERA5_roll_longitude
from ERA5_load import ERA5_quantile_normalise


t = ERA5_load_T2m(args.year, args.month, args.day)
c = ERA5_load_T2m_climatology(args.year, args.month, args.day)
a = t - c
a = ERA5_roll_longitude(a)
n = ERA5_quantile_normalise(a)

lm = ERA5_load_LS_mask()
lm = ERA5_roll_longitude(lm)


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
    figsize=(20, 20),
    dpi=100,
    facecolor=(0.88, 0.88, 0.88, 1),
    edgecolor=None,
    linewidth=0.0,
    frameon=False,
    subplotpars=None,
    tight_layout=None,
)
canvas = FigureCanvas(fig)

ax_global = fig.add_axes([0, 0, 1, 1], facecolor="white")

# Top - original anomalies
ax_of = fig.add_axes([0.01, 0.505, 0.98, 0.485])
ax_of.set_aspect("auto")
ax_of.set_axis_off()
ofp = plot_T2m(
    ax_of,
    a,
    vMin=-10,
    vMax=10,
    land=lm,
    label="Original: %04d-%02d-%02d" % (args.year, args.month, args.day),
)
# Bottom - normalised anomalies
ax_of = fig.add_axes([0.01, 0.01, 0.98, 0.485])
ax_of.set_aspect("auto")
ax_of.set_axis_off()
ofp = plot_T2m(
    ax_of,
    n,
    vMin=-0.2,
    vMax=1.2,
    land=lm,
    label="Quantile normalised",
)


fig.savefig("comparison.png")
