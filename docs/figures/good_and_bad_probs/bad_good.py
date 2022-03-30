#!/usr/bin/env python

import os
import sys
import numpy as np

import matplotlib
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
from matplotlib.figure import Figure
import cmocean
import iris
import IRData.twcr as twcr
import datetime

import tensorflow as tf
import tensorflow_probability as tfp
from tensorflow_addons.image import interpolate_bilinear

obs_year = 1979
obs_month = 3
obs_day = 12

sys.path.append(
    "%s/../../../data/prepare_training_tensors_ERA5_T2m" % os.path.dirname(__file__)
)
from ERA5_load import ERA5_load_T2m
from ERA5_load import ERA5_load_T2m_climatology
from ERA5_load import ERA5_roll_longitude
from ERA5_load import ERA5_trim
from ERA5_load import ERA5_load_LS_mask

sys.path.append("%s/../../../lib" % os.path.dirname(__file__))
from geometry import to_analysis_grid

mask = ERA5_load_LS_mask()
mask = ERA5_roll_longitude(mask)
mask = ERA5_trim(mask)

# Make a bad background field
t = ERA5_load_T2m(obs_year, obs_month, obs_day)
c = ERA5_load_T2m_climatology(obs_year, obs_month, obs_day)
a = t - c
a = ERA5_roll_longitude(a)
a = ERA5_trim(a)
cs = iris.coord_systems.RotatedGeogCS(90.0, 180.0, 0.0)
a.coord("latitude").coord_system = cs
a.coord("longitude").coord_system = cs
bg = to_analysis_grid(a)
bg.data = np.random.random(bg.data.shape) * 10 - 5
a_in = tf.convert_to_tensor(a.data, np.float32)
a_in = tf.reshape(a_in, [1, 720, 1440, 1])

# Get the ob locations at the given time from 20CRv3
dte = datetime.datetime(obs_year, obs_month, obs_day, 12)
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
exact = tf.squeeze(interpolate_bilinear(a_in, t_obs, indexing="ij"))
# Filter out the nans (bad lat/lon)
t_obs = tf.boolean_mask(t_obs, ~tf.math.is_nan(exact), axis=1)
exact = tf.boolean_mask(exact, ~tf.math.is_nan(exact), axis=0)
approx = exact + tf.random.normal(
    shape=exact.shape, mean=0.0, stddev=20.0 / 15, dtype=tf.float32
)
# Modify bg to match the obs, where present
lat_i = t_obs[0, :, 0].numpy() * 80 / 720
lon_i = t_obs[0, :, 1].numpy() * 160 / 1440
for i in range(len(exact)):
    latidx = 80 - int(lat_i[i])
    lonidx = int(lon_i[i]) - 1
    bg.data[latidx, lonidx] = exact[i]
bg = bg.regrid(a, iris.analysis.Linear())
bg.data[mask.data == 1] *= 1.1
bg_in = tf.convert_to_tensor(bg.data, np.float32)
bg_in = tf.reshape(bg_in, [1, 720, 1440, 1])

target = tf.squeeze(interpolate_bilinear(bg_in, t_obs, indexing="ij"))
target = tf.boolean_mask(target, ~tf.math.is_nan(exact), axis=0)
target = exact

sys.path.append(
    "%s/../../../models/DCVAE_single_ERA5_T2m/validation" % os.path.dirname(__file__)
)
from plot_ERA5_comparison import get_land_mask
from plot_ERA5_comparison import plot_T2m
from plot_ERA5_comparison import plot_scatter

fig = Figure(
    figsize=(30, 10),
    dpi=100,
    facecolor=(0.88, 0.88, 0.88, 1),
    edgecolor=None,
    linewidth=0.0,
    frameon=False,
    subplotpars=None,
    tight_layout=None,
)
canvas = FigureCanvas(fig)
matplotlib.rcParams["font.size"] = 24

ax_global = fig.add_axes([0, 0, 1, 1], facecolor="white")
ax_global.set_axis_off()
ax_global.autoscale(enable=False)
ax_global.fill((-0.1, 1.1, 1.1, -0.1), (-0.1, -0.1, 1.1, 1.1), "white")

ax_of = fig.add_axes([0.01, 0.01, 0.65, 0.98])
ax_of.set_aspect("auto")
ax_of.set_axis_off()
ofp = plot_T2m(
    ax_of,
    bg.data,
    vMin=-10,
    vMax=10,
    land=mask,
    label="",
)

ax_scatter = fig.add_axes([0.72, 0.14, 0.24, 0.72])
plot_scatter(
    ax_scatter,
    target.numpy() / 10 + 0.5,
    approx.numpy() / 10 + 0.5,
    d_max=15,
    d_min=-15,
    lw=2,
    xlab="Field",
    ylab="Observations",
)


fig.savefig("bad_good.png")
