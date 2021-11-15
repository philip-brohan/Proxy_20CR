#!/usr/bin/env python

# Find a point in latent space that maximises the fit to the pseudo obs,
#  and plot the fitted state.

import os
import sys

import iris
import IRData.twcr as twcr
import datetime

import tensorflow as tf
import tensorflow_probability as tfp
from tensorflow_graphics.math.interpolation import trilinear
import random

import matplotlib
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
from matplotlib.figure import Figure

import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--epoch", help="Epoch", type=int, required=False, default=25)
parser.add_argument("--year", help="Year", type=int, required=True)
parser.add_argument("--month", help="Integer month", type=int, required=True)
parser.add_argument("--day", help="Day of month", type=int, required=True)
parser.add_argument(
    "--hour", help="Time of day (0, 6, 12, or 18)", type=int, required=True
)
parser.add_argument(
    "--variable", help="variable name", default="prmsl", type=str, required=False
)
parser.add_argument(
    "--ipdir",
    help="Directory for pseudo obs tensors",
    default="%s/Proxy_20CR/pseudo_obs" % os.getenv("SCRATCH"),
    type=str,
    required=False,
)
args = parser.parse_args()

# Script directory (for finding libraries) - '.' if run interactively
dirName = None
try:
    dirName = os.path.dirname(__file__)
except NameError:
    dirName = "."



# Load the pseudo obs tensor
ipfile = "%s/%s/%04d-%02d-%02d:%02d.tfd" % (
    args.ipdir,
    args.variable,
    args.year,
    args.month,
    args.day,
    args.hour,
)
sict = tf.io.read_file(ipfile)
pot = tf.io.parse_tensor(sict, tf.float32)
# Extract lat, lon and time in shape for interpolation
pot_l = tf.stack((pot[0,:],pot[1,:], pot[2,:]), axis=1)
pot_l = tf.expand_dims(pot_l, 0)
# Extract perturbed pseudo-obs as target
pot_t = pot[4, :]

# Load the target data from 20CR - convert to normalised tensor
sys.path.append("%s/../../../lib/" % dirName)

from geometry import to_analysis_grid
from normalise import load_normal
from normalise import normalise_prmsl_anomaly


def load_hour(year, month, day, hour, variable=args.variable, member=1):
    ic = twcr.load(
        variable,
        datetime.datetime(year, month, day, hour),
        version="2c",
    )
    ic = ic.extract(iris.Constraint(member=member))
    ic = to_analysis_grid(ic)
    n = load_normal(variable, year, month, day, hour)
    n = n.regrid(ic, iris.analysis.Linear())
    ic.data = normalise_prmsl_anomaly(ic.data, n.data)
    ict = tf.convert_to_tensor(ic.data, tf.float32)
    return ict


target = load_hour(args.year, args.month, args.day, args.hour)

# Load the trained generative model
sys.path.append("%s/.." % dirName)
from autoencoderModel import DCVAE

autoencoder = DCVAE()
weights_dir = ("%s/Proxy_20CR/models/DCVAE_sequence_PRMSL/" + "Epoch_%04d") % (
    os.getenv("SCRATCH"),
    args.epoch - 1,
)
load_status = autoencoder.load_weights("%s/ckpt" % weights_dir)
# Check the load worked
load_status.assert_existing_objects_matched()
# We are using it in inference mode
autoencoder.trainable=False

# Random start point
latent = tf.Variable(tf.random.normal(shape=(1, autoencoder.latent_dim)))
#print(latent)

def log_normal_pdf(sample, mean, logvar, raxis=1):
    log2pi = tf.math.log(2.0 * 3.141592)
    return tf.reduce_sum(
        -0.5 * ((sample - mean) ** 2.0 * tf.exp(-logvar) + logvar + log2pi), axis=raxis
    )
# Function to calculate the fit of the generated field to the pseudo-obs
def decodeFit():
    decoded = autoencoder.decode(latent)
    #print(decoded.shape)
    # convert field from diffs back to (normalised) pressure
    #df = tf.unstack(decoded, axis=2)
    #for idx in [0, 4]:
    #    df[idx] = df[2] + df[idx] / 2
    #for idx in [1, 3]:
    #    df[idx] = df[2] + df[idx] / 3
    #decoded = tf.stack(df, axis=2)
    #decoded = tf.expand_dims(decoded,4)
    #print(decoded.shape)
    #print(pot_l.shape)
    #sys.exit(0)
   # Reshape for interpolation
    #at_proxies = tf.squeeze(trilinear.interpolate(decoded, pot_l))
    #print(at_proxies)
    #print(pot_t)
    #return tf.reduce_mean(tf.keras.metrics.mean_squared_error(at_proxies, pot_t))
    rmse = tf.reduce_mean(tf.keras.metrics.mean_squared_error(decoded[:,:,:,2],target))
    logpz = log_normal_pdf(latent, 0.0, 0.0)
    return (rmse-logpz)


#print(decodeFit())
#print(latent)
#sys.exit(0)

loss = tfp.math.minimize(
    decodeFit,
    trainable_variables=[latent],
    num_steps=10000,
    optimizer=tf.optimizers.Adam(learning_rate=0.1),
)
print(loss)

# Plot the reconstructed and target fields
sys.path.append("%s/../validation" % dirName)
from plot_prmsl_comparison import get_land_mask
from plot_prmsl_comparison import plot_PRMSL

fig = Figure(
    figsize=(19.2, 10.8),
    dpi=100,
    facecolor=(0.88, 0.88, 0.88, 1),
    edgecolor=None,
    linewidth=0.0,
    frameon=False,
)
canvas = FigureCanvas(fig)

ax_global = fig.add_axes([0, 0, 1, 1], facecolor="white")
lm = get_land_mask()

encoded = autoencoder.decode(latent)
ax_plot = fig.add_axes([0.01, 0.01, 0.98, 0.98])
ax_plot.set_aspect("auto")
ax_plot.set_axis_off()
#print(target)
#print(encoded[:,:,:,2])
#sys.exit(0)
plot_PRMSL(
    ax_plot,
    tf.reshape(target, [80, 160]),
    tf.reshape(encoded[:,:,:,2], [80, 160]),
    land=lm,
    label="%04d-%02d-%02d:%02d" % (args.year, args.month, args.day, args.hour),
)

fig.savefig("fit.png")
