#!/usr/bin/env python

# Plot a validation figure for the autoencoder.

# Fir components
#  1) Original field
#  2) Encoded field
#  3) Difference field
#  4) Original:Encoded scatter
#

import tensorflow as tf
import os
import sys
import random
import numpy as np

import matplotlib
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
from matplotlib.figure import Figure

import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--epoch", help="Epoch", type=int, required=True)
parser.add_argument(
    "--case", help="Test case to plot", type=int, required=False, default=0
)
args = parser.parse_args()

sys.path.append("%s/." % os.path.dirname(__file__))
from plot_HUKG_comparison import get_land_mask
from plot_HUKG_comparison import plot_Tmax
from plot_HUKG_comparison import plot_scatter
from plot_HUKG_comparison import plot_colourbar

# Load the data source provider
sys.path.append("%s/.." % os.path.dirname(__file__))
from makeDataset import getDataset
from autoencoderModel import DCVAE

testData = getDataset(purpose="test")

autoencoder = DCVAE()
weights_dir = (
    "%s/Proxy_20CR/models/DCVAE_single_ERA5_to_HUKG_Tmax/" + "Epoch_%04d"
) % (
    os.getenv("SCRATCH"),
    args.epoch,
)
load_status = autoencoder.load_weights("%s/ckpt" % weights_dir).expect_partial()
# Check the load worked
load_status.assert_existing_objects_matched()

# Get the field to use
count = 0
for t_in in testData:
    if count == args.case:
        break
    count += 1

# Make encoded version
encoded = tf.convert_to_tensor(
    autoencoder.predict_on_batch(tf.reshape(t_in[0], [1, 1440, 896, 1]))
)

# Make the figure
lm = get_land_mask()

fig = Figure(
    figsize=(15, 15),
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

# Top left - original field
ax_of = fig.add_axes([0.01, 0.565, 0.323, 0.425])
ax_of.set_aspect("auto")
ax_of.set_axis_off()
ofp = plot_Tmax(
    ax_of,
    (t_in[1] - 0.5) * 10,
    vMin=-5,
    vMax=5,
    land=lm,
    label="Original: %d" % args.case,
)
ax_ocb = fig.add_axes([0.0365, 0.505, 0.27, 0.05])
plot_colourbar(fig, ax_ocb, ofp)

# Top centre - ERA5 field
ax_of = fig.add_axes([0.34, 0.565, 0.323, 0.425])
ax_of.set_aspect("auto")
ax_of.set_axis_off()
ofp = plot_Tmax(
    ax_of,
    (t_in[0] - 0.5) * 10,
    vMin=-5,
    vMax=5,
    land=lm,
    label="ERA5",
)
ax_ocb = fig.add_axes([0.3665, 0.505, 0.27, 0.05])
plot_colourbar(fig, ax_ocb, ofp)

# Bottom centre - ERA difference field
ax_of = fig.add_axes([0.34, 0.065, 0.323, 0.425])
ax_of.set_aspect("auto")
ax_of.set_axis_off()
ofp = plot_Tmax(
    ax_of,
    (t_in[0] - t_in[1]) * 10,
    vMin=-5,
    vMax=5,
    land=lm,
    label="ERA5 Difference",
)
ax_ocb = fig.add_axes([0.3665, 0.005, 0.27, 0.05])
plot_colourbar(fig, ax_ocb, ofp)

# Top right - encoded field
ax_of = fig.add_axes([0.67, 0.565, 0.323, 0.425])
ax_of.set_aspect("auto")
ax_of.set_axis_off()
ofp = plot_Tmax(
    ax_of,
    (encoded - 0.5) * 10,
    vMin=-5,
    vMax=5,
    land=lm,
    label="Generator",
)
ax_ocb = fig.add_axes([0.6965, 0.505, 0.27, 0.05])
plot_colourbar(fig, ax_ocb, ofp)

# Bottom right - generated difference field
ax_of = fig.add_axes([0.67, 0.065, 0.323, 0.425])
ax_of.set_aspect("auto")
ax_of.set_axis_off()
ofp = plot_Tmax(
    ax_of,
    (encoded - t_in[1]) * 10,
    vMin=-5,
    vMax=5,
    land=lm,
    label="Generator Difference",
)
ax_ocb = fig.add_axes([0.78, 0.005, 0.27, 0.05])
plot_colourbar(fig, ax_ocb, ofp)

# Bottom right - scatterplots

xmin = np.min(
    np.concatenate(
        (
            t_in[0].numpy().flatten(),
            t_in[1].numpy().flatten(),
            encoded.numpy().flatten(),
        )
    )
)
xmax = np.max(
    np.concatenate(
        (
            t_in[0].numpy().flatten(),
            t_in[1].numpy().flatten(),
            encoded.numpy().flatten(),
        )
    )
)
ax_scatter = fig.add_axes([0.05, 0.29, 0.22, 0.22])
plot_scatter(
    ax_scatter, t_in[1], t_in[0], xlab="Original", ylab="ERA5", d_min=xmin, d_max=xmax
)

ax_scatter2 = fig.add_axes([0.05, 0.05, 0.22, 0.22])
plot_scatter(
    ax_scatter2,
    t_in[1],
    encoded,
    xlab="Original",
    ylab="Generator",
    d_min=xmin,
    d_max=xmax,
)


fig.savefig("comparison.png")
