#!/usr/bin/env python

# Find a point in latent space that maximises the fit to a set of pseudo-obs,
#  and plot the fitted state.

import os
import sys
import numpy as np

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
parser.add_argument("--year", type=int, required=False, default=1979)
parser.add_argument("--month", type=int, required=False, default=3)
parser.add_argument("--day", type=int, required=False, default=12)
parser.add_argument("--oyear", help="Year", type=int, required=False)
parser.add_argument("--omonth", help="Integer month", type=int, required=False)
parser.add_argument("--oday", help="Day of month", type=int, required=False)
args = parser.parse_args()
if args.oyear is None:
    args.oyear = args.year
if args.omonth is None:
    args.omonth = args.month
if args.oday is None:
    args.oday = args.day

# Functions for plotting
sys.path.append("%s/../validation" % os.path.dirname(__file__))
from plot_ERA5_comparison import get_land_mask
from plot_ERA5_comparison import plot_T2m
from plot_ERA5_comparison import plot_colourbar


# Make the input tensor for the specified date
sys.path.append(
    "%s/../../../data/prepare_training_tensors_ERA5_T2m" % os.path.dirname(__file__)
)
from ERA5_load import ERA5_load_T2m
from ERA5_load import ERA5_load_T2m_climatology
from ERA5_load import ERA5_roll_longitude
from ERA5_load import ERA5_trim

t = ERA5_load_T2m(args.year, args.month, args.day)
c = ERA5_load_T2m_climatology(args.year, args.month, args.day)
t = t - c
t /= 15
t += 0.5
t = ERA5_roll_longitude(t)
t = ERA5_trim(t)
t_in = tf.convert_to_tensor(t.data, np.float32)
t_in = tf.reshape(t_in, [1, 720, 1440, 1])

# Get the ob locations at the given time from 20CRv3
dte = datetime.datetime(args.oyear, args.omonth, args.oday, 12)
obs = twcr.load_observations_1file(dte, version="3")
# Convert the obs locations to a tensor in the right units (0-1)
t_lats = (obs["Latitude"].values + 90) / 180
t_lons = (obs["Longitude"].values) / 360
t_lons[t_lons > 0.5] -= 1
t_lons += 0.5
t_lats = tf.convert_to_tensor(t_lats, tf.float32)
t_lons = tf.convert_to_tensor(t_lons, tf.float32)
t_obs = tf.stack((t_lats * 720, t_lons * 1440), axis=1)
t_obs = tf.expand_dims(t_obs, 0)

# Set up the model and load the weights at the chosen epoch
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

# We are using it in inference mode
# (I'm not at all sure this actually works)
autoencoder.decoder.trainable = False
for layer in autoencoder.decoder.layers:
    layer.trainable = False
autoencoder.decoder.compile()


latent = tf.Variable(tf.random.normal(shape=(1, autoencoder.latent_dim)))
target = tf.constant(t_in)

exact = tf.squeeze(interpolate_bilinear(target, t_obs, indexing="ij"))
# Filter out the nans (bad lat/lon)
t_obs = tf.boolean_mask(t_obs, ~tf.math.is_nan(exact), axis=1)
exact = tf.boolean_mask(exact, ~tf.math.is_nan(exact), axis=0)
approx = exact + tf.random.normal(
    shape=exact.shape, mean=0.0, stddev=1.0 / 15, dtype=tf.float32
)


def decodeFit():
    decoded = autoencoder.decode(latent)
    at_obs = tf.squeeze(interpolate_bilinear(decoded, t_obs, indexing="ij"))
    return tf.reduce_mean(tf.keras.metrics.mean_squared_error(approx, at_obs))


loss = tfp.math.minimize(
    decodeFit,
    trainable_variables=[latent],
    num_steps=1000,
    optimizer=tf.optimizers.Adam(learning_rate=0.05),
)
print(loss)

fig = Figure(
    figsize=(19.2 / 2, 10.8),
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

# Bottom - original field
ax_of = fig.add_axes([0.01, 0.065, 0.98, 0.425])
ax_of.set_aspect("auto")
ax_of.set_axis_off()
ofp = plot_T2m(
    ax_of,
    (t_in - 0.5) * 15,
    vMin=-10,
    vMax=10,
    land=lm,
    label="Original: %04d-%02d-%02d" % (args.year, args.month, args.day),
)
ax_ocb = fig.add_axes([0.05, 0.025, 0.81, 0.02])
plot_colourbar(fig, ax_ocb, ofp)


# Top, encoded field
encoded = autoencoder.decode(latent)
ax_ef = fig.add_axes([0.01, 0.565, 0.98, 0.425])
ax_ef.set_aspect("auto")
ax_ef.set_axis_off()
efp = plot_T2m(
    ax_ef,
    (encoded - 0.5) * 15,
    vMin=-10,
    vMax=10,
    land=lm,
    label="Encoded",
)
ax_ecb = fig.add_axes([0.05, 0.525, 0.81, 0.02])
plot_colourbar(fig, ax_ecb, efp)


fig.savefig("fit.png")
