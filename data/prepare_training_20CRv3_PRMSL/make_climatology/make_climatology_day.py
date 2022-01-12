#!/usr/bin/env python

import os
import sys
import iris

sys.path.append("%s/.." % os.path.dirname(__file__))
from v3_PRMSL_load import v3_load_PRMSL

# Going to do external parallelism - run this on one core
import dask

dask.config.set(scheduler="single-threaded")

import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--month", help="Integer month", type=int, required=True)
parser.add_argument("--day", help="Day of month", type=int, required=True)
parser.add_argument("--hour", help="Hour of day", type=int, required=True)
args = parser.parse_args()

opfile = "%s/Proxy_20CR/datasets/20CRv3/PRMSL/climatology/%02d/%02d/%02d.nc" % (
    os.getenv("SCRATCH"),
    args.month,
    args.day,
    args.hour,
)

if not os.path.isdir(os.path.dirname(opfile)):
    os.makedirs(os.path.dirname(opfile))

c = None
for year in range(1980, 2010):
    yd = v3_load_PRMSL(year, args.month, args.day, args.hour)
    if c is None:
        c = yd.copy()
    else:
        c += yd
c /= 30

iris.save(c, opfile)
