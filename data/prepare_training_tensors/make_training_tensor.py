#!/usr/bin/env python

# Read in a field from 20CR as an Iris cube.
# Rescale it and move UK to the centre of the field.
# Convert it into a TensorFlow tensor.
# Serialise it and store it on $SCRATCH.

import tensorflow as tf
import numpy

import IRData.twcr as twcr
import iris
import datetime
import argparse
import os
import sys

sys.path.append("%s/../../lib/" % os.path.dirname(__file__))
from normalise import normalise_t2m
from normalise import normalise_wind
from normalise import normalise_prmsl
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
parser.add_argument(
    "--variable", help="variable name", default="prmsl", type=str, required=False
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
    args.opfile = ("%s/Proxy_20CR/datasets/" + "%s/%s/%s/%04d-%02d-%02d:%02d.tfd") % (
        os.getenv("SCRATCH"),
        args.source,
        args.variable,
        purpose,
        args.year,
        args.month,
        args.day,
        args.hour,
    )

if not os.path.isdir(os.path.dirname(args.opfile)):
    os.makedirs(os.path.dirname(args.opfile))

# Load and standardise data
if args.source == "20CR2c":
    ic = twcr.load(
        args.variable,
        datetime.datetime(args.year, args.month, args.day, args.hour),
        version="2c",
    )
    ic = ic.extract(iris.Constraint(member=args.member))
    ic = to_analysis_grid(ic)
    if args.variable == "uwnd.10m" or args.variable == "vwnd.10m":
        ic.data = normalise_wind(ic.data)
    elif args.variable == "air.2m":
        ic.data = normalise_t2m(ic.data)
    elif args.variable == "prmsl":
        ic.data = normalise_prmsl(ic.data)
    else:
        raise ValueError("Variable %s is not supported" % args.variable)

else:
    raise ValueError("Source %s is not supported" % args.source)

# Convert to Tensor
ict = tf.convert_to_tensor(ic.data, numpy.float32)

# Write to file
sict = tf.io.serialize_tensor(ict)
tf.io.write_file(args.opfile, sict)
