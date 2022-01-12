#!/usr/bin/env python

# Read in a field of 20CRv3 PRMSL as an Iris cube.
# Convert it into an anomaly
# Rotate it to have the UK in the centre
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
from v3_PRMSL_load import v3_load_PRMSL
from v3_PRMSL_load import v3_load_PRMSL_climatology
from v3_PRMSL_load import v3_roll_longitude

import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--year", help="Year", type=int, required=True)
parser.add_argument("--month", help="Integer month", type=int, required=True)
parser.add_argument("--day", help="Day of month", type=int, required=True)
parser.add_argument("--hour", help="Hour of day", type=int, required=True)
parser.add_argument("--test", help="test data, not training", action="store_true")
parser.add_argument(
    "--opfile", help="tf data file name", default=None, type=str, required=False
)
args = parser.parse_args()
if args.opfile is None:
    purpose = "training"
    if args.test:
        purpose = "test"
    args.opfile = ("%s/Proxy_20CR/datasets/" + "%s/%s/%s/%04d-%02d-%02d:%02d.tfd") % (
        os.getenv("SCRATCH"),
        "20CRv3",
        "PRMSL",
        purpose,
        args.year,
        args.month,
        args.day,
        args.hour,
    )

if not os.path.isdir(os.path.dirname(args.opfile)):
    os.makedirs(os.path.dirname(args.opfile))

# Load and anomalise data
t = v3_load_PRMSL(args.year, args.month, args.day, args.hour)
c = v3_load_PRMSL_climatology(args.year, args.month, args.day, args.hour)
t = t - c
# Rescale to range 0-1 (approx)
t /= 3000
t += 0.5
# Roll the longitude to put the UK in the centre
t = v3_roll_longitude(t)

# Convert to Tensor
ict = tf.convert_to_tensor(t.data, np.float32)

# Write to file
sict = tf.io.serialize_tensor(ict)
tf.io.write_file(args.opfile, sict)
