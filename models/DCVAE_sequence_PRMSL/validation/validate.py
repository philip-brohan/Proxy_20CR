#!/usr/bin/env python

# Plot a validation figure for the autoencoder.

# Three tests:
#  1) Sample from the training dataset
#  2) Sample from the test dataset
#  3) Generated samples from random point in the latent space
#

import tensorflow as tf
import os 
import sys
import random

import matplotlib
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
from matplotlib.figure import Figure

import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--epoch", help="Epoch", type=int, required=False, default=25)
args = parser.parse_args()

sys.path.append("%s/." % os.path.dirname(__file__))
from plot_prmsl_comparison import get_land_mask
from plot_prmsl_comparison import plot_PRMSL
from plot_prmsl_comparison import plot_scatter

# Load the data source provider
sys.path.append("%s/../../PRMSL_dataset" % os.path.dirname(__file__))
from makeSequenceDataset import getDataset

trainingData = getDataset(purpose="training")
testData = getDataset(purpose="test")

# Set up the model and load the weights at the chosen epoch
sys.path.append("%s/.." % os.path.dirname(__file__))
from autoencoderModel import DCVAE

autoencoder = DCVAE()
weights_dir = ("%s/Proxy_20CR/models/DCVAE_sequence_PRMSL/" + "Epoch_%04d") % (
    os.getenv("SCRATCH"),
    args.epoch - 1,
)
load_status = autoencoder.load_weights("%s/ckpt" % weights_dir)
# Check the load worked
load_status.assert_existing_objects_matched()

margin = 0.05
f_width = (margin * 5 / 8) * 4 + 3 + 3 + 2
f_height = margin * 6 + 5
s_width = 1 / f_width
s_height = 1 / f_height
m_width = (margin * 5 / 8) / f_width
m_height = margin / f_height
fig = Figure(
    figsize=(f_width, f_height),
    dpi=250,
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

# Plot a random example from the training dataset
count = 0
rnd = random.randint(1, 2567)
for t_in in trainingData:
    if count == rnd:
        encoded = autoencoder.sample_call(tf.reshape(t_in, [1, 80, 160, 5]), size=15)
        for tp in range(5):  # Each time-slice
            ax_plot = fig.add_axes(
                [m_width, 1 - (m_height + s_height) * (tp + 1), 2 * s_width, s_height]
            )
            ax_plot.set_aspect("auto")
            ax_plot.set_axis_off()
            plot_PRMSL(
                ax_plot,
                t_in[:, :, tp],
                encoded[:, :, :, :, tp],
                land=lm,
                label="%d, %s" % (count, ("-12h", "-6h", "0h", "+6h", "+12h")[tp]),
                linewidths=[0.2, 0.05],
                d_min=-0.2,
                d_max=1.2,
                c_space=0.1,
            )
            ax_scatter = fig.add_axes(
                [
                    m_width + 2 * s_width,
                    1 - (m_height + s_height) * (tp + 1),
                    s_width,
                    s_height,
                ]
            )
            ax_scatter.set_aspect("auto")
            ax_scatter.set_axis_off()
            plot_scatter(
                ax_scatter,
                t_in[:, :, tp],
                encoded[:, :, :, :, tp],
                d_min=-0.2,
                d_max=1.2,
            )
        break
    count += 1

# Plot one randomexamples from the test data set:
count = 0
rnd = random.randint(1, 256)
for t_in in testData:
    if count == rnd:
        encoded = autoencoder.sample_call(tf.reshape(t_in, [1, 80, 160, 5]), size=15)
        for tp in range(5):  # Each time-slice
            ax_plot = fig.add_axes(
                [
                    m_width * 2 + 3 * s_width,
                    1 - (m_height + s_height) * (tp + 1),
                    2 * s_width,
                    s_height,
                ]
            )
            ax_plot.set_aspect("auto")
            ax_plot.set_axis_off()
            plot_PRMSL(
                ax_plot,
                t_in[:, :, tp],
                encoded[:, :, :, :, tp],
                land=lm,
                label="%d, %s" % (count, ("-12h", "-6h", "0h", "+6h", "+12h")[tp]),
                linewidths=[0.2, 0.05],
                d_min=-0.2,
                d_max=1.2,
                c_space=0.1,
            )
            ax_scatter = fig.add_axes(
                [
                    m_width * 2 + 5 * s_width,
                    1 - (m_height + s_height) * (tp + 1),
                    s_width,
                    s_height,
                ]
            )
            ax_scatter.set_axis_off()
            plot_scatter(
                ax_scatter,
                t_in[:, :, tp],
                encoded[:, :, :, :, tp],
                d_min=-0.2,
                d_max=1.2,
            )
        break
    count += 1

# Plot one example of a generated field
eps = tf.random.normal(shape=(1, autoencoder.latent_dim))
generated = autoencoder.decode(eps)
for tp in range(5):  # Each time-slice
    ax_plot = fig.add_axes(
        [
            m_width * 3 + 6 * s_width,
            1 - (m_height + s_height) * (tp + 1),
            2 * s_width,
            s_height,
        ]
    )
    ax_plot.set_aspect("auto")
    ax_plot.set_axis_off()
    plot_PRMSL(
        ax_plot,
        None,
        generated[:, :, :, tp],
        land=lm,
        label="Generator %s" % ("-12h", "-6h", "0h", "+6h", "+12h")[tp],
        linewidths=[0.5, 0.5],
    )

fig.savefig("comparison.png")
