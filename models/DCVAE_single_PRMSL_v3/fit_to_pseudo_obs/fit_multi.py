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
import numpy as np

import iris
import IRData.twcr as twcr
import datetime

import matplotlib
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
from matplotlib.figure import Figure

import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--epoch", help="Epoch", type=int, required=True)
parser.add_argument(
    "--ensemble", help="No. of ensemble members", type=int, required=False, default=25
)
parser.add_argument("--year", type=int, required=False, default=1979)
parser.add_argument("--month", type=int, required=False, default=3)
parser.add_argument("--day", type=int, required=False, default=12)
parser.add_argument(
    "--hour", help="Time of day (0, 6, 12, or 18)", type=int, required=False, default=6
)
parser.add_argument("--oyear", help="Year", type=int, required=False)
parser.add_argument("--omonth", help="Integer month", type=int, required=False)
parser.add_argument("--oday", help="Day of month", type=int, required=False)
parser.add_argument(
    "--ohour",
    help="Time of day (0, 6, 12, or 18)",
    type=int,
    required=False,
)
parser.add_argument(
    "--noise", help="Ob noise stdev (hPa)", type=float, required=False, default=0.0
)
args = parser.parse_args()
if args.oyear is None:
    args.oyear = args.year
if args.omonth is None:
    args.omonth = args.month
if args.oday is None:
    args.oday = args.day
if args.ohour is None:
    args.ohour = args.hour


sys.path.append("%s/../validation" % os.path.dirname(__file__))
from plot_prmsl_comparison import get_land_mask
from plot_prmsl_comparison import plot_PRMSL

# Make the input tensor for the specified date
sys.path.append(
    "%s/../../../data/prepare_training_20CRv3_PRMSL" % os.path.dirname(__file__)
)
from v3_PRMSL_load import v3_load_PRMSL
from v3_PRMSL_load import v3_load_PRMSL_climatology
from v3_PRMSL_load import v3_roll_longitude

t = v3_load_PRMSL(args.year, args.month, args.day, args.hour)
c = v3_load_PRMSL_climatology(args.year, args.month, args.day, args.hour)
t = t - c
t /= 3000
t += 0.5
t = v3_roll_longitude(t)
t_in = tf.convert_to_tensor(t.data, np.float32)
t_in = tf.reshape(t_in, [1, 256, 512, 1])

# Get the ob locations at the given time
dte = datetime.datetime(args.oyear, args.omonth, args.oday, args.ohour)
obs = twcr.load_observations_1file(dte, version="3")
# Convert the obs locations to a tensor in the right units (0-1)
t_lats = (obs["Latitude"].values + 90) / 180
t_lons = (obs["Longitude"].values) / 360
t_lons[t_lons > 0.5] -= 1
t_lons += 0.5
t_lats = tf.convert_to_tensor(1.0 - t_lats, tf.float32)
t_lons = tf.convert_to_tensor(t_lons, tf.float32)
t_obs = tf.stack((t_lats * 256, t_lons * 512), axis=1)
t_obs = tf.expand_dims(t_obs, 0)

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
# We are using it in inference mode
# (I'm not at all sure this actually works)
autoencoder.decoder.trainable = False
for layer in autoencoder.decoder.layers:
    layer.trainable = False
autoencoder.decoder.compile()

exact = tf.squeeze(interpolate_bilinear(t_in, t_obs, indexing="ij"))
# Filter out the nans (bad lat/lon)
t_obs = tf.boolean_mask(t_obs, ~tf.math.is_nan(exact), axis=1)
exact = tf.boolean_mask(exact, ~tf.math.is_nan(exact), axis=0)

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
        at_obs = tf.squeeze(interpolate_bilinear(decoded, t_obs, indexing="ij"))
        return tf.keras.metrics.mean_squared_error(pseudo_obs, at_obs)

    loss = tfp.math.minimize(
        decodeFit,
        trainable_variables=[latent],
        num_steps=num_steps,
        optimizer=optimizer,
        convergence_criterion=tfp.optimizer.convergence_criteria.LossNotDecreasing(
            atol=0.00001, min_num_steps=100
        ),
    )
    return (latent, loss)


# Make a set of fitted fields
latent = tf.Variable(tf.random.normal(shape=(args.ensemble, autoencoder.latent_dim)))
pseudo_obs_sample = exact + tf.random.normal(
    shape=exact.shape, mean=0.0, stddev=args.noise / 30, dtype=tf.float32
)
(latent, loss) = findLatent(autoencoder, latent, t_obs, pseudo_obs_sample)
fitted = autoencoder.decode(latent)

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
matplotlib.rcParams.update({"font.size": 16})

ax_global = fig.add_axes([0, 0, 1, 1], facecolor="white")
lm = get_land_mask()

encoded = autoencoder.decode(latent)
ax_plot = fig.add_axes([0.01, 0.01, 0.98, 0.98])
ax_plot.set_aspect("auto")
ax_plot.set_axis_off()
plot_PRMSL(
    ax_plot,
    tf.reshape(t_in, [256, 512]),
    tf.reshape(encoded, [args.ensemble, 256, 512]),
    obs=t_obs,
    linewidths=[0.5, 0.2, 0.5],
    land=lm,
    label="Field: %04d-%02d-%02d:%02d\nObservations: %04d-%02d-%02d:%02d"
    % (
        args.year,
        args.month,
        args.day,
        args.hour,
        args.oyear,
        args.omonth,
        args.oday,
        args.ohour,
    ),
)

fig.savefig("fit_multi.png")
