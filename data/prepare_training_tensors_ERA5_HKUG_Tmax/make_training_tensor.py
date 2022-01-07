#!/usr/bin/env python

# Read in a field of ERA5 Tmax as an Iris cube.
# Convert it into an anomaly
# Regrid it to match the HadUKGrid tensord
# Convert it into a TensorFlow tensor.
# Serialise it and store it on $SCRATCH.

import tensorflow as tf
import numpy as np

# Going to do external parallelism - run this on one core
tf.config.threading.set_inter_op_parallelism_threads(1)
import dask

dask.config.set(scheduler="single-threaded")


import IRData.twcr as twcr
import iris
import datetime
import argparse
import os
import sys

sys.path.append("%s" % os.path.dirname(__file__))
from ERA5_load import ERA5_load_Tmax
from ERA5_load import ERA5_load_Tmax_climatology

sys.path.append("%s/../prepare_training_tensors_HUKG_Tmax" % os.path.dirname(__file__))
from HUKG_load_tmax import HUKG_load_tmax
from HUKG_load_tmax import HUKG_trim

import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--year", help="Year", type=int, required=True)
parser.add_argument("--month", help="Integer month", type=int, required=True)
parser.add_argument("--day", help="Day of month", type=int, required=True)
parser.add_argument("--test", help="test data, not training", action="store_true")
parser.add_argument(
    "--opfile", help="tf data file name", default=None, type=str, required=False
)
args = parser.parse_args()
if args.opfile is None:
    purpose = "training"
    if args.test:
        purpose = "test"
    args.opfile = ("%s/Proxy_20CR/datasets/" + "%s/%s/%s/%04d-%02d-%02d.tfd") % (
        os.getenv("SCRATCH"),
        "ERA5",
        "daily_Tmax",
        purpose,
        args.year,
        args.month,
        args.day,
    )

if not os.path.isdir(os.path.dirname(args.opfile)):
    os.makedirs(os.path.dirname(args.opfile))

# Load and anomalise data
t = ERA5_load_Tmax(args.year, args.month, args.day)
c = ERA5_load_Tmax_climatology(args.year, args.month, args.day)
t = t - c
# Rescale to range 0-1 (approx)
t /= 10
t += 0.5

# Convert it to HadUKGrid grid
g = HUKG_load_tmax(args.year, args.month, args.day)
t = t.regrid(g, iris.analysis.Linear())
# discard bottom left to make sizes multiply divisible by 2
t = HUKG_trim(t)

# Convert to Tensor
ict = tf.convert_to_tensor(t.data, np.float32)

# Write to file
sict = tf.io.serialize_tensor(ict)
tf.io.write_file(args.opfile, sict)
