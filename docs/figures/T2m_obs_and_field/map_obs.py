#!/usr/bin/env python

# Plot a set of pseudo-obs.

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
parser.add_argument("--year", type=int, required=False, default=1979)
parser.add_argument("--month", type=int, required=False, default=3)
parser.add_argument("--day", type=int, required=False, default=12)
parser.add_argument("--oyear", help="Year", type=int, required=False)
parser.add_argument("--omonth", help="Integer month", type=int, required=False)
parser.add_argument("--oday", help="Day of month", type=int, required=False)
parser.add_argument(
    "--osize", help="Obs. point size", type=float, required=False, default=4.0
)
args = parser.parse_args()
if args.oyear is None:
    args.oyear = args.year
if args.omonth is None:
    args.omonth = args.month
if args.oday is None:
    args.oday = args.day

# Functions for plotting
sys.path.append(
    "%s/../../../models/DCVAE_single_ERA5_T2m/validation" % os.path.dirname(__file__)
)
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
t_lats = tf.convert_to_tensor(1.0 - t_lats, tf.float32)
t_lons = tf.convert_to_tensor(t_lons, tf.float32)
t_obs = tf.stack((t_lats * 720, t_lons * 1440), axis=1)
t_obs = tf.expand_dims(t_obs, 0)

exact = tf.squeeze(interpolate_bilinear(t_in, t_obs, indexing="ij"))
# Filter out the nans (bad lat/lon)
t_obs = tf.boolean_mask(t_obs, ~tf.math.is_nan(exact), axis=1)
exact = tf.boolean_mask(exact, ~tf.math.is_nan(exact), axis=0)
approx = exact + tf.random.normal(
    shape=exact.shape, mean=0.0, stddev=2.0 / 15, dtype=tf.float32
)


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
matplotlib.rcParams.update({"font.size": 16})

ax_global = fig.add_axes([0, 0, 1, 1], facecolor="white")
ax_global.set_axis_off()
ax_global.autoscale(enable=False)
ax_global.fill((-0.1, 1.1, 1.1, -0.1), (-0.1, -0.1, 1.1, 1.1), "white")

lm = get_land_mask()

# Obs only
ax_of = fig.add_axes([0.01, 0.01, 0.98, 0.98])
ax_of.set_aspect("equal")
ax_of.set_axis_off()
ax_of.set_xlim(-180, 180)
ax_of.set_ylim(-90, 90)
efp = plot_T2m(
    ax_of,
    None,
    vMin=-10,
    vMax=10,
    obs=tf.squeeze(t_obs).numpy(),
    obs_c=(tf.squeeze(approx).numpy() - 0.5) * 15,
    o_size=args.osize,
    land=lm,
    label="",
)

fig.savefig("obs.png")
