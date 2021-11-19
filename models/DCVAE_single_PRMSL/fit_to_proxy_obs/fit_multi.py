#!/usr/bin/env python

# Find a point in latent space that maximises the fit to a set of pseudo-obs,
#  and plot the fitted state.
# Make multiple fits and plot the ensemble.

import os
import sys

import tensorflow as tf
import tensorflow_probability as tfp
from tensorflow_addons.image import interpolate_bilinear

import random

import iris
import IRData.twcr as twcr
import datetime

import matplotlib
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
from matplotlib.figure import Figure

import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--epoch", help="Epoch", type=int, required=True)
parser.add_argument("--ensemble", help="No. of ensemble members", type=int, required=False, default=10)
parser.add_argument("--case", help="Epoch", type=int, required=False, default=0)
parser.add_argument("--year", help="Year", type=int, required=True)
parser.add_argument("--month", help="Integer month", type=int, required=True)
parser.add_argument("--day", help="Day of month", type=int, required=True)
parser.add_argument(
    "--hour", help="Time of day (0, 6, 12, or 18)", type=int, required=True
)
parser.add_argument(
    "--noise", help="Ob noise stdev (hPa)", type=float, required=False, default=0.0
)
args = parser.parse_args()

sys.path.append("%s/../validation" % os.path.dirname(__file__))
from plot_prmsl_comparison import get_land_mask
from plot_prmsl_comparison import plot_PRMSL

# Get the ob locations at the given time from 2c
def get_obs_as_tensor(year, month, day, hour):
    dte = datetime.datetime(year, month, day, hour)
    obs = twcr.load_observations_1file(dte, version="2c")
    # Convert the obs locations to a tensor in the right units
    t_lats = (obs["Latitude"].values + 90) / 180
    t_lons = (obs["Longitude"].values) / 360
    t_lons[t_lons > 0.5] -= 1
    t_lons += 0.5
    t_lats = tf.convert_to_tensor(t_lats, tf.float32)
    t_lons = tf.convert_to_tensor(t_lons, tf.float32)
    t_obs = tf.stack((t_lats * 80, t_lons * 160), axis=1)
    t_obs = tf.expand_dims(t_obs, 0)
    return t_obs


ob_locations = get_obs_as_tensor(args.year, args.month, args.day, args.hour)

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
# We are using it in inference mode
# (I'm not at all sure this actually works)
autoencoder.decoder.trainable = False
for layer in autoencoder.decoder.layers:
    layer.trainable = False
autoencoder.decoder.compile()

# Get the target field
count = 0
for t_in in testData:
    if count == args.case:
        target = tf.constant(tf.reshape(t_in, [1, 80, 160, 1]))
        break
    count += 1


# Make the pseudo-obs
pseudo_obs = tf.squeeze(interpolate_bilinear(target, ob_locations, indexing="ij"))
# Filter out the nans (bad lat/lon)
ob_locations = tf.boolean_mask(ob_locations, ~tf.math.is_nan(pseudo_obs), axis=1)
pseudo_obs = tf.boolean_mask(pseudo_obs, ~tf.math.is_nan(pseudo_obs), axis=0)


# Find a latent state which generates a field fitted to the pseudo obs.
def findLatent(
    autoencoder,
    latent,
    ob_locations,
    pseudo_obs,
    num_steps=1000,
    optimizer=tf.optimizers.Adam(learning_rate=0.05),
):
    def decodeFit():
        decoded = autoencoder.decode(latent)
        at_obs = tf.squeeze(interpolate_bilinear(decoded, ob_locations, indexing="ij"))
        return tf.reduce_mean(tf.keras.metrics.mean_squared_error(pseudo_obs, at_obs))

    loss = tfp.math.minimize(
        decodeFit,
        trainable_variables=[latent],
        num_steps=num_steps,
        optimizer=optimizer,
    )
    return (latent, loss)


# Make a set of fitted fields
fitted = []
for i in range(args.ensemble):
    latent = tf.Variable(tf.random.normal(shape=(1, autoencoder.latent_dim)))
    pseudo_obs_sample = pseudo_obs + tf.random.normal(
        shape=pseudo_obs.shape, mean=0.0, stddev=args.noise / 30, dtype=tf.float32
    )
    (latent, loss) = findLatent(autoencoder, latent, ob_locations, pseudo_obs_sample)
    fitted.append(autoencoder.decode(latent))

fitted = tf.stack(fitted, axis=0)

# latent = tf.Variable(tf.random.normal(shape=(1, autoencoder.latent_dim)))
# load_status = autoencoder.load_weights("%s/ckpt" % weights_dir)
# Check the load worked
# load_status.assert_existing_objects_matched()
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
    fitted,
    obs=ob_locations,
    land=lm,
    label="Test: %d" % count,
    linewidths=[0.5,0.2,0.5]
)

fig.savefig("fit_multi.png")
