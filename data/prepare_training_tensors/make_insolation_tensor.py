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
from normalise import normalise_insolation
from geometry import to_analysis_grid

import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--year", help="Year", type=int, required=True)
parser.add_argument("--month", help="Integer month", type=int, required=True)
parser.add_argument("--day", help="Day of month", type=int, required=True)
parser.add_argument("--hour", help="Hour of day (0 to 23)", type=int, required=True)
parser.add_argument(
    "--opfile", help="tf data file name", default=None, type=str, required=False
)
args = parser.parse_args()
if args.opfile is None:
    args.opfile = ("%s/ML_GCM/datasets/" + "%s/%s/%s/%04d-%02d-%02d:%02d.tfd") % (
        os.getenv("SCRATCH"),
        "20CR2c",
        "insolation",
        "training",
        args.year,
        args.month,
        args.day,
        args.hour,
    )

if not os.path.isdir(os.path.dirname(args.opfile)):
    os.makedirs(os.path.dirname(args.opfile))

# Don't distinguish between training and test for insolation.
#  Make a 'test' directory that's a copy of the 'training' directory'
tstdir = os.path.dirname(args.opfile).replace("training", "test")
if not os.path.exists(tstdir):
    os.symlink(os.path.dirname(args.opfile), tstdir)

# Load the 20CR2c data as an iris cube
time_constraint = iris.Constraint(
    time=iris.time.PartialDateTime(
        year=args.year, month=args.month, day=args.day, hour=args.hour
    )
)
ic = iris.load_cube(
    "%s/20CR/version_2c/ensmean/cduvb.1969.nc" % os.getenv("DATADIR"),
    iris.Constraint(name="3-hourly Clear Sky UV-B Downward Solar Flux")
    & time_constraint,
)
coord_s = iris.coord_systems.GeogCS(iris.fileformats.pp.EARTH_RADIUS)
ic.coord("latitude").coord_system = coord_s
ic.coord("longitude").coord_system = coord_s

# Standardise
ic = to_analysis_grid(ic)
ic.data = normalise_insolation(ic.data)

# Convert to Tensor
ict = tf.convert_to_tensor(ic.data, numpy.float32)

# Write to file
sict = tf.io.serialize_tensor(ict)
tf.io.write_file(args.opfile, sict)
