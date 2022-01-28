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
import numpy as np

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


sys.path.append(
    "%s/../../../data/prepare_training_tensors_ERA5_T2m" % os.path.dirname(__file__)
)
from ERA5_load import ERA5_load_T2m
from ERA5_load import ERA5_load_T2m_climatology
from ERA5_load import ERA5_roll_longitude
from ERA5_load import ERA5_trim

# Make the input tensor for the specified date
t = ERA5_load_T2m(args.year, args.month, args.day)
c = ERA5_load_T2m_climatology(args.year, args.month, args.day)
t = t - c
t /= 15
t += 0.5
t = ERA5_roll_longitude(t)
t = ERA5_trim(t)
t_in = tf.convert_to_tensor(t.data, np.float32)
t_in = tf.reshape(t_in, [1, 720, 1440, 1])

sys.path.append("%s/." % os.path.dirname(__file__))
from plot_ERA5_comparison import get_land_mask
from plot_ERA5_comparison import plot_T2m
from plot_ERA5_comparison import plot_scatter
from plot_ERA5_comparison import plot_colourbar

# Define the model
sys.path.append("%s/.." % os.path.dirname(__file__))
from autoencoderModel import DCVAE

autoencoder = DCVAE()
weights_dir = ("%s/Proxy_20CR/models/DCVAE_single_ERA5_T2m/" + "Epoch_%04d") % (
    os.getenv("SCRATCH"),
    args.epoch,
)
load_status = autoencoder.load_weights("%s/ckpt" % weights_dir).expect_partial()
# Check the load worked
load_status.assert_existing_objects_matched()

# Make encoded version
encoded = tf.convert_to_tensor(autoencoder.predict_on_batch(t_in), np.float32)

# Make the figure
lm = get_land_mask()

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

ax_global = fig.add_axes([0, 0, 1, 1], facecolor="white")
ax_global.set_axis_off()
ax_global.autoscale(enable=False)
ax_global.fill((-0.1, 1.1, 1.1, -0.1), (-0.1, -0.1, 1.1, 1.1), "white")

# Top left - original field
ax_of = fig.add_axes([0.01, 0.565, 0.485, 0.425])
ax_of.set_aspect("auto")
ax_of.set_axis_off()
ofp = plot_T2m(
    ax_of,
    tf.squeeze(t_in - 0.5).numpy() * 15,
    vMin=-10,
    vMax=10,
    land=lm,
    label="Original: %04d-%02d-%02d" % (args.year, args.month, args.day),
)
ax_ocb = fig.add_axes([0.05, 0.525, 0.405, 0.02])
plot_colourbar(fig, ax_ocb, ofp)

# Top right - encoded field
ax_of = fig.add_axes([0.502, 0.565, 0.485, 0.425])
ax_of.set_aspect("auto")
ax_of.set_axis_off()
ofp = plot_T2m(
    ax_of,
    tf.squeeze(encoded - 0.5).numpy() * 15,
    vMin=-10,
    vMax=10,
    land=lm,
    label="Encoded",
)
ax_ocb = fig.add_axes([0.57, 0.525, 0.405, 0.02])
plot_colourbar(fig, ax_ocb, ofp)

# Bottom left - difference field
ax_of = fig.add_axes([0.01, 0.065, 0.485, 0.425])
ax_of.set_aspect("auto")
ax_of.set_axis_off()
ofp = plot_T2m(
    ax_of,
    tf.squeeze(encoded - t_in).numpy() * 15,
    vMin=-10,
    vMax=10,
    land=lm,
    label="Difference",
)
ax_ocb = fig.add_axes([0.05, 0.025, 0.405, 0.02])
plot_colourbar(fig, ax_ocb, ofp)

# Bottom right - scatterplot

ax_scatter = fig.add_axes([0.67, 0.05, 0.2, 0.4])
plot_scatter(ax_scatter, t_in.numpy(), encoded.numpy(), d_max=15, d_min=-15)


fig.savefig("comparison.png")
