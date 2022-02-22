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
import iris

import matplotlib
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
from matplotlib.figure import Figure

import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--epoch", help="Epoch", type=int, required=True)
parser.add_argument("--year", type=int, required=False, default=1979)
parser.add_argument("--month", type=int, required=False, default=3)
parser.add_argument("--day", type=int, required=False, default=12)
args = parser.parse_args()

sys.path.append("%s/." % os.path.dirname(__file__))
from plot_HUKG_comparison import get_land_mask
from plot_HUKG_comparison import plot_Tmax
from plot_HUKG_comparison import plot_scatter
from plot_HUKG_comparison import plot_colourbar

# Make the HUKG tensor for the specified date
sys.path.append(
    "%s/../../../data/prepare_training_tensors_HUKG_Tmax" % os.path.dirname(__file__)
)
from HUKG_load_tmax import HUKG_load_tmax
from HUKG_load_tmax import HUKG_load_tmax_climatology
from HUKG_load_tmax import HUKG_trim
from HUKG_load_tmax import HUKG_load_observations

ht = HUKG_load_tmax(args.year, args.month, args.day)
hc = HUKG_load_tmax_climatology(args.year, args.month, args.day)
ht = ht - hc
ht /= 10
ht += 0.5
ht = HUKG_trim(ht)
ht.data.data[ht.data.mask] = 0.5
msk = ht.data.mask 
ht_in = tf.convert_to_tensor(ht.data.data, np.float32)
ht_in = tf.reshape(ht_in, [1, 1440, 896, 1])

# Make the ERA5 tensor for the specified date
sys.path.append(
    "%s/../../../data/prepare_training_tensors_ERA5_HKUG_Tmax" % os.path.dirname(__file__)
)
from ERA5_load import ERA5_load_Tmax
from ERA5_load import ERA5_load_Tmax_climatology

et = ERA5_load_Tmax(args.year, args.month, args.day)
ec = ERA5_load_Tmax_climatology(args.year, args.month, args.day)
et = et - ec
et /= 10
et += 0.5
# Convert it to HadUKGrid grid
et = et.regrid(ht, iris.analysis.Linear())
# discard bottom left to make sizes multiply divisible by 2
et = HUKG_trim(et)
et_in = tf.convert_to_tensor(et.data, np.float32)
et_in = tf.reshape(et_in, [1, 1440, 896, 1])

# Set up the model and load the weights at the chosen epoch
sys.path.append("%s/.." % os.path.dirname(__file__))
from autoencoderModel import DCVAE

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


# Make encoded version
encoded = autoencoder.predict_on_batch(tf.reshape(et_in, [1, 1440, 896, 1]))
encoded = np.squeeze(encoded)
encoded[msk]=0.5
encoded = tf.convert_to_tensor(encoded, np.float32)
encoded = tf.reshape(encoded, [1, 1440, 896, 1])

# Discard masked components of ERA5
et_in = tf.squeeze(et_in).numpy()
et_in[msk]=0.5
et_in = tf.convert_to_tensor(et_in, np.float32)
et_in = tf.reshape(et_in, [1, 1440, 896, 1])

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
matplotlib.rcParams.update({"font.size": 16})

ax_global = fig.add_axes([0, 0, 1, 1], facecolor="white")
ax_global.set_axis_off()
ax_global.autoscale(enable=False)
ax_global.fill((-0.1, 1.1, 1.1, -0.1), (-0.1, -0.1, 1.1, 1.1), "white")

# Top left - original field
ax_of = fig.add_axes([0.01, 0.565, 0.323, 0.425])
ax_of.set_aspect("auto")
ax_of.set_axis_off()
ofp = plot_Tmax(
    ax_of,
    (ht_in - 0.5) * 10,
    vMin=-5,
    vMax=5,
    land=lm,
    label="Original: %04d-%02d-%02d" % (args.year,args.month,args.day),
)
ax_ocb = fig.add_axes([0.0365, 0.505, 0.27, 0.05])
plot_colourbar(fig, ax_ocb, ofp)

# Top centre - ERA5 field
ax_of = fig.add_axes([0.34, 0.565, 0.323, 0.425])
ax_of.set_aspect("auto")
ax_of.set_axis_off()
ofp = plot_Tmax(
    ax_of,
    (et_in - 0.5) * 10,
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
    (et_in - ht_in) * 10,
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
    (encoded - ht_in) * 10,
    vMin=-5,
    vMax=5,
    land=lm,
    label="Generator Difference",
)
ax_ocb = fig.add_axes([0.6965, 0.005, 0.27, 0.05])
plot_colourbar(fig, ax_ocb, ofp)

# Bottom right - scatterplots

xmin = (np.min(
    np.concatenate(
        (
            ht_in.numpy().flatten(),
            et_in.numpy().flatten(),
            encoded.numpy().flatten(),
        )
    )
)-0.5)*10
xmax = (np.max(
    np.concatenate(
        (
            ht_in.numpy().flatten(),
            et_in.numpy().flatten(),
            encoded.numpy().flatten(),
        )
    )
) -0.5)*10
ax_scatter = fig.add_axes([0.07, 0.29, 0.22, 0.22])
plot_scatter(
    ax_scatter, ht_in, et_in, xlab="Original", ylab="ERA5", d_min=xmin, d_max=xmax
)

ax_scatter2 = fig.add_axes([0.07, 0.05, 0.22, 0.22])
plot_scatter(
    ax_scatter2,
    ht_in,
    encoded,
    xlab="Original",
    ylab="Generator",
    d_min=xmin,
    d_max=xmax,
)


fig.savefig("comparison.png")
