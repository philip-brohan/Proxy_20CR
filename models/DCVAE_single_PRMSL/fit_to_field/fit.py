#!/usr/bin/env python

# Find a point in latent space that maximises the fit to a test case,
#  and plot the fitted state.


import tensorflow as tf
import tensorflow_probability as tfp
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

sys.path.append("%s/../validation" % os.path.dirname(__file__))
from plot_prmsl_comparison import get_land_mask
from plot_prmsl_comparison import plot_PRMSL

# Load the data source provider
sys.path.append("%s/../../PRMSL_dataset" % os.path.dirname(__file__))
from makeDataset import getDataset

testData = getDataset(purpose="test")

# Set up the model and load the weights at the chosen epoch
sys.path.append("%s/.." % os.path.dirname(__file__))
from autoencoderModel import DCVAE

autoencoder = DCVAE()
weights_dir = ("%s/Proxy_20CR/models/DCVAE_single_PRMSL/" + "Epoch_%04d") % (
    os.getenv("SCRATCH"),
    args.epoch,
)
load_status = autoencoder.load_weights("%s/ckpt" % weights_dir)
# Check the load worked
load_status.assert_existing_objects_matched()

count=0
for t_in in testData:
    if count == 0:
        latent = tf.random.normal(shape=(1, autoencoder.latent_dim))
        target = tf.reshape(t_in, [1, 80, 160, 1])

        def decodeFit(latent):
            decoded = autoencoder.decode(latent)
            return tf.reduce_mean(tf.keras.metrics.mean_squared_error(decoded, target))

        loss = tfp.math.minimize(
            lambda: decodeFit(latent),
            num_steps=1000,
            optimizer=tf.optimizers.Adam(learning_rate=0.0001),
        )
        print(loss)
        break
        #sys.exit(0)

fig = Figure(
    figsize=(19.2, 10.8),
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

encoded = autoencoder.decode(latent)
ax_plot = fig.add_axes([0.01, 0.01, 0.98, 0.98])
ax_plot.set_aspect("auto")
ax_plot.set_axis_off()
plot_PRMSL(
    ax_plot,
    tf.reshape(target, [80, 160]),
    tf.reshape(encoded, [80, 160]),
    land=lm,
    label="Test: %d" % count,
)

fig.savefig("fit.png")
