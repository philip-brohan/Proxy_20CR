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
parser.add_argument("--case", help="Test case", type=int, required=False,default=0)
parser.add_argument('--mslp', dest='mslp', default=False, action='store_true')
parser.add_argument('--uwnd', dest='uwnd', default=False, action='store_true')
parser.add_argument('--vwnd', dest='vwnd', default=False, action='store_true')
args = parser.parse_args()

sys.path.append("%s/../validation" % os.path.dirname(__file__))
from plot_prmsl_comparison import get_land_mask
from plot_prmsl_comparison import plot_PRMSL

# Load the data source provider
sys.path.append("%s/../../PUV_dataset" % os.path.dirname(__file__))
from makeDataset import getDataset

testData = getDataset(purpose="test")

# Set up the model and load the weights at the chosen epoch
sys.path.append("%s/.." % os.path.dirname(__file__))
from autoencoderModel import DCVAE

autoencoder = DCVAE()
weights_dir = ("%s/Proxy_20CR/models/DCVAE_single_PUV/" + "Epoch_%04d") % (
    os.getenv("SCRATCH"),
    args.epoch,
)
load_status = autoencoder.load_weights("%s/ckpt" % weights_dir)
# Check the load worked
load_status.assert_existing_objects_matched()
# We are using it in inference mode - (does this have any effect?)
autoencoder.decoder.trainable=False
for layer in autoencoder.decoder.layers:
        layer.trainable = False
autoencoder.decoder.compile()


count=0
for t_in in testData:
    if count == args.case:
        latent = tf.Variable(tf.random.normal(shape=(1, autoencoder.latent_dim)))
        target = tf.constant(tf.reshape(t_in, [1, 80, 160,3]))
        #print(latent)
        def decodeFit():
            result = 0.0
            decoded = autoencoder.decode(latent)
            if args.mslp:
                result = result + tf.reduce_mean(tf.keras.metrics.mean_squared_error(decoded[:,:,:,0], target[:,:,:,0]))
            if args.uwnd:
                result = result + tf.reduce_mean(tf.keras.metrics.mean_squared_error(decoded[:,:,:,1], target[:,:,:,1]))/10
            if args.vwnd:
                result = result + tf.reduce_mean(tf.keras.metrics.mean_squared_error(decoded[:,:,:,2], target[:,:,:,2]))/10
            return result

        loss = tfp.math.minimize(
            decodeFit,
            trainable_variables=[latent],
            num_steps=1000,
            optimizer=tf.optimizers.Adam(learning_rate=0.05),
        )
        print(loss)
        break
        #sys.exit(0)

#latent = tf.Variable(tf.random.normal(shape=(1, autoencoder.latent_dim)))
#load_status = autoencoder.load_weights("%s/ckpt" % weights_dir)
# Check the load worked
#load_status.assert_existing_objects_matched()
fig = Figure(
    figsize=(19.2, 10.8*3),
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
ax_plot = fig.add_axes([0.01, 0.01/3, 0.98, 0.98/3])
ax_plot.set_aspect("auto")
ax_plot.set_axis_off()
plot_PRMSL(
    ax_plot,
    tf.reshape(target[:,:,:,2], [80, 160]),
    tf.reshape(encoded[:,:,:,2], [80, 160]),
    c_space=0.2,
    land=lm,
    label="vwnd",
)
ax_plot = fig.add_axes([0.01, 0.02/3+1/3, 0.98, 0.98/3])
ax_plot.set_aspect("auto")
ax_plot.set_axis_off()
plot_PRMSL(
    ax_plot,
    tf.reshape(target[:,:,:,1], [80, 160]),
    tf.reshape(encoded[:,:,:,1], [80, 160]),
    c_space=0.2,
    land=lm,
    label="uwnd",
)
ax_plot = fig.add_axes([0.01, 0.03/3+2/3, 0.98, 0.98/3])
ax_plot.set_aspect("auto")
ax_plot.set_axis_off()
plot_PRMSL(
    ax_plot,
    tf.reshape(target[:,:,:,0], [80, 160]),
    tf.reshape(encoded[:,:,:,0], [80, 160]),
    land=lm,
    label="mslp",
)

fig.savefig("fit.png")
