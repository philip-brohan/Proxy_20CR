#!/usr/bin/env python

import os
import sys
import iris

sys.path.append("%s/.." % os.path.dirname(__file__))
from ERA5_load import ERA5_load_T2m

# Going to do external parallelism - run this on one core
import dask

dask.config.set(scheduler="single-threaded")

import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--month", help="Integer month", type=int, required=True)
parser.add_argument("--day", help="Day of month", type=int, required=True)
args = parser.parse_args()

opfile = "%s/Proxy_20CR/datasets/ERA5/daily_T2m/climatology/%02d/%02d.nc" % (
    os.getenv("SCRATCH"),
    args.month,
    args.day,
)

if not os.path.isdir(os.path.dirname(opfile)):
    os.makedirs(os.path.dirname(opfile))

c = None
for year in range(1981, 2010):
    yd = ERA5_load_T2m(year, args.month, args.day)
    if c is None:
        c = yd.copy()
    else:
        c += yd
c /= 30

iris.save(c, opfile)
