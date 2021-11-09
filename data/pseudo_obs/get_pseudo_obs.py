#!/usr/bin/env python

# Make pseudo obs for a 24 hour period.
# Use the obs coverage from 20CRv3, but make the obs from 20CRv3 ensemble member 1
#  with noise added (independent random gaussian, sd=2hPa).

import os
import sys

import iris
import IRData.twcr as twcr
import datetime

import tensorflow as tf
from tensorflow_graphics.math.interpolation import trilinear
import numpy as np
import pandas as pd

try:
    sys.path.append("%s/../../lib/" % os.path.dirname(__file__))
except NameError:
    sys.path.append("%s/../../lib/" % ".")

from geometry import to_analysis_grid
from normalise import load_normal
from normalise import normalise_prmsl_anomaly

import argparse

parser = argparse.ArgumentParser()
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
    "--opdir",
    help="Directory for output files",
    default="%s/Proxy_20CR/pseudo_obs" % os.getenv("SCRATCH"),
    type=str,
    required=False,
)
args = parser.parse_args()
if not os.path.isdir("%s/%s" % (args.opdir, args.variable)):
    os.makedirs("%s/%s" % (args.opdir, args.variable))
opfile = "%s/%s/%04d-%02d-%02d:%02d.tfd" % (
    args.opdir,
    args.variable,
    args.year,
    args.month,
    args.day,
    args.hour,
)


dte = datetime.datetime(args.year, args.month, args.day, args.hour)

# Get the surrounding 24 hours worth of observations from 2c
obs = twcr.load_observations(
    dte - datetime.timedelta(hours=12), dte + datetime.timedelta(hours=12), version="2c"
)

# Get the surrounding 24 hours worth of pressure fields from 2c
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
    ict = tf.convert_to_tensor(ic.data, np.float32)
    return ict


fields = []
for offset in [-12, -6, 0, 6, 12]:
    dtc = datetime.datetime(
        args.year, args.month, args.day, args.hour
    ) + datetime.timedelta(hours=offset)
    fields.append(load_hour(dtc.year, dtc.month, dtc.day, dtc.hour))

fields = tf.stack(fields, axis=2)
fields = tf.expand_dims(tf.expand_dims(fields, 0), 4)


# Convert the obs locations to a tensor in the right units (0-1)
t_lats = (obs["Latitude"].values + 90) / 180
t_lons = (obs["Longitude"].values) / 360
o_dtm = pd.to_datetime(obs["UID"].str.slice(0, 10), format="%Y%m%d%H")
dts = dte - datetime.timedelta(hours=12)
t_dte = ((o_dtm - dts) / pd.Timedelta(hours=1)).values / 24

t_obs = tf.stack((t_lats, t_lons, t_dte), axis=1)
t_obs = tf.expand_dims(t_obs, 0)

exact = tf.squeeze(trilinear.interpolate(fields, tf.cast(t_obs, "float32")))
approx = exact + tf.random.normal(
    shape=exact.shape, mean=0.0, stddev=2.0 / 30, dtype=tf.float32
)
ict = tf.stack((t_lats, t_lons, t_dte, exact, approx))

# Save the pseudo-obs
sict = tf.io.serialize_tensor(ict)
tf.io.write_file(opfile, sict)
