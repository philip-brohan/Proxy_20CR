#!/usr/bin/env python

# Plot a validation figure for the autoencoder.

# Three groups of tests:
#  1) Samples from the training dataset
#  2) Samples from the test dataset
#  3) Generated samples from random points in the latent space
#
# In each case, half the points are fixed (the same each time this is run),
#  and half are random samples (different each time).

import tensorflow as tf
import os
import sys
import random

import matplotlib
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
from matplotlib.figure import Figure

import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--epoch", help="Epoch", type=int, required=True)
args = parser.parse_args()

sys.path.append("%s/." % os.path.dirname(__file__))
from plot_prmsl_comparison import get_land_mask
from plot_prmsl_comparison import plot_PRMSL
from plot_prmsl_comparison import plot_scatter

# Load the data source provider
sys.path.append("%s/.." % os.path.dirname(__file__))
from makeDataset import getDataset

trainingData = getDataset(purpose="training")
testData = getDataset(purpose="test")

# Set up the model and load the weights at the chosen epoch
sys.path.append("%s/.." % os.path.dirname(__file__))
from autoencoderModel import DCVAE

autoencoder = DCVAE()
weights_dir = ("%s/Proxy_20CR/models/DCVAE_single_PRMSL_v3/" + "Epoch_%04d") % (
    os.getenv("SCRATCH"),
    args.epoch,
)
load_status = autoencoder.load_weights("%s/ckpt" % weights_dir).expect_partial()
# Check the load worked
load_status.assert_existing_objects_matched()


fig = Figure(
    figsize=(35.6, 10.8),
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
lm = get_land_mask()

# Plot two examples from the training data set: First, and one at random
count = 0
rnd = random.randint(1, 2567)
for t_in in trainingData:
    if count == 0:
        encoded = autoencoder.sample_call(tf.reshape(t_in, [1, 256, 512, 1]), size=15)
        ax_plot = fig.add_axes([0.01, 0.505, 0.32, 0.485])
        ax_plot.set_aspect("auto")
        ax_plot.set_axis_off()
        plot_PRMSL(
            ax_plot,
            t_in,
            encoded,
            land=lm,
            label="Training: %d" % count,
            linewidths=[1, 0.2],
        )
        ax_scatter = fig.add_axes([0.34, 0.505, 0.32, 0.485])
        ax_plot.set_aspect("auto")
        plot_scatter(ax_plot, t_in, encoded)
    if count == rnd:
        encoded = autoencoder.sample_call(tf.reshape(t_in, [1, 256, 512, 1]), size=15)
        ax_plot = fig.add_axes([0.01, 0.01, 0.32, 0.485])
        ax_plot.set_aspect("auto")
        ax_plot.set_axis_off()
        plot_PRMSL(
            ax_plot,
            t_in,
            encoded,
            land=lm,
            label="Training: %d" % count,
            linewidths=[1, 0.2],
        )
        break
    count += 1

# Plot two examples from the test data set: First, and one at random
count = 0
rnd = random.randint(1, 284)
for t_in in testData:
    if count == 0:
        encoded = autoencoder.sample_call(tf.reshape(t_in, [1, 256, 512, 1]), size=15)
        ax_plot = fig.add_axes([0.34, 0.505, 0.32, 0.485])
        ax_plot.set_aspect("auto")
        ax_plot.set_axis_off()
        plot_PRMSL(
            ax_plot,
            t_in,
            encoded,
            land=lm,
            label="Test: %d" % count,
            linewidths=[1, 0.2],
        )
    if count == rnd:
        encoded = autoencoder.sample_call(tf.reshape(t_in, [1, 256, 512, 1]), size=15)
        ax_plot = fig.add_axes([0.34, 0.01, 0.32, 0.485])
        ax_plot.set_aspect("auto")
        ax_plot.set_axis_off()
        plot_PRMSL(
            ax_plot,
            t_in,
            encoded,
            land=lm,
            label="Test: %d" % count,
            linewidths=[1, 0.2],
        )
        break
    count += 1

# Plot two examples of generated fields
for y in [0.505, 0.01]:
    eps = tf.random.normal(shape=(1, autoencoder.latent_dim))
    generated = autoencoder.decode(eps)
    ax_plot = fig.add_axes([0.67, y, 0.32, 0.485])
    ax_plot.set_aspect("auto")
    ax_plot.set_axis_off()
    plot_PRMSL(
        ax_plot,
        None,
        tf.reshape(generated, [256, 512]),
        land=lm,
        label="Generator only",
    )

fig.savefig("comparison.png")
