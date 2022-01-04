#!/usr/bin/env python

import os
import sys
import numpy as np
import iris
import iris.analysis.maths

sys.path.append("%s/.." % os.path.dirname(__file__))
from ERA5_load import ERA5_load_T2m
from ERA5_load import ERA5_load_T2m_climatology

# Going to do external parallelism - run this on one core
import dask

dask.config.set(scheduler="single-threaded")

import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--month", help="Integer month", type=int, required=True)
parser.add_argument("--day", help="Day of month", type=int, required=True)
args = parser.parse_args()

opfile = (
    "%s/Proxy_20CR/datasets/ERA5/daily_T2m/variability_climatology/%02d/%02d.nc"
    % (
        os.getenv("SCRATCH"),
        args.month,
        args.day,
    )
)

if not os.path.isdir(os.path.dirname(opfile)):
    os.makedirs(os.path.dirname(opfile))

yc = ERA5_load_T2m_climatology(1981, args.month, args.day)
c = None
for year in range(1981, 2011):
    yd = ERA5_load_T2m(year, args.month, args.day)
    yd = yd - yc
    yd = yd * yd
    if c is None:
        c = yd.copy()
    else:
        c += yd
c /= 30
c = iris.analysis.maths.apply_ufunc(np.sqrt, c)

iris.save(c, opfile)
