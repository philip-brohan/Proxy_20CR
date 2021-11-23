#!/usr/bin/env python

# Read in a field from 20CR as an Iris cube.
# Rescale it and move UK to the centre of the field.
# Convert it into a TensorFlow tensor.
# Serialise it and store it on $SCRATCH.

# This version stores prmsl, u and v wind

import tensorflow as tf
import numpy

# Going to do external parallelism - run this on one core
tf.config.threading.set_inter_op_parallelism_threads(1)
import dask
dask.config.set(scheduler='single-threaded')

import IRData.twcr as twcr
import iris
import datetime
import argparse
import os
import sys

sys.path.append("%s/../../lib/" % os.path.dirname(__file__))
from normalise import load_normal
from normalise import normalise_wind_anomaly
from normalise import normalise_prmsl_anomaly
from geometry import to_analysis_grid

import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--year", help="Year", type=int, required=True)
parser.add_argument("--month", help="Integer month", type=int, required=True)
parser.add_argument("--day", help="Day of month", type=int, required=True)
parser.add_argument("--hour", help="Hour of day (0 to 23)", type=int, required=True)
parser.add_argument(
    "--member", help="Ensemble member", default=1, type=int, required=False
)
parser.add_argument(
    "--source", help="Data source", default="20CR2c", type=str, required=False
)

parser.add_argument("--test", help="test data, not training", action="store_true")
parser.add_argument(
    "--opfile", help="tf data file name", default=None, type=str, required=False
)
args = parser.parse_args()
if args.opfile is None:
    purpose = "training"
    if args.test:
        purpose = "test"
    args.opfile = ("%s/Proxy_20CR/datasets/" + "%s/puv/%s/%04d-%02d-%02d:%02d.tfd") % (
        os.getenv("SCRATCH"),
        args.source,
        purpose,
        args.year,
        args.month,
        args.day,
        args.hour,
    )

if not os.path.isdir(os.path.dirname(args.opfile)):
    os.makedirs(os.path.dirname(args.opfile))

# Load the normals

# Load and standardise data
def load_hour(year, month, day, hour, var):
    ic = twcr.load(
        var,
        datetime.datetime(year, month, day, hour),
        version="2c",
    )
    ic = ic.extract(iris.Constraint(member=args.member))
    ic = to_analysis_grid(ic)
    if var == "uwnd.10m" or var == "vwnd.10m":
        n = load_normal(var, year, month, day, hour)
        n = n.regrid(ic, iris.analysis.Linear())
        ic.data = normalise_wind_anomaly(ic.data, n.data)
    elif var == "prmsl":
        n = load_normal(var, year, month, day, hour)
        n = n.regrid(ic, iris.analysis.Linear())
        ic.data = normalise_prmsl_anomaly(ic.data, n.data)
    else:
        raise ValueError("Variable %s is not supported" % args.variable)
    ict = tf.convert_to_tensor(ic.data, numpy.float32)
    return ict


dte = datetime.datetime(args.year, args.month, args.day, args.hour)
puv = []
for var in ["prmsl", "uwnd.10m", "vwnd.10m"]:
    puv.append(load_hour(dte.year, dte.month, dte.day, dte.hour, var))

ict = tf.stack(puv, axis=2)

# Rescale everything *approximately* onto the 0-1 range
ict /= 2.5
ict += 0.5


# Write to file
sict = tf.io.serialize_tensor(ict)
tf.io.write_file(args.opfile, sict)
