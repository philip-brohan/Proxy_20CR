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
from plot_ERA5_comparison import plot_T2m

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
ax_global.autoscale(enable=False)
ax_global.fill((-0.1, 1.1, 1.1, -0.1), (-0.1, -0.1, 1.1, 1.1), "white")

ax_of = fig.add_axes([0.01, 0.01, 0.98, 0.98])
ax_of.set_aspect("auto")
ax_of.set_axis_off()
ofp = plot_T2m(
    ax_of,
    a.data,
    vMin=-10,
    vMax=10,
    land=mask,
    label="",
)


fig.savefig("field.png")
