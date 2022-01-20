#!/usr/bin/env python

import os
import sys
import iris
import pickle
from tdigest import TDigest
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--month", help="Integer month", type=int, required=True)
parser.add_argument("--day", help="Day of month", type=int, required=True)
args = parser.parse_args()

sys.path.append("%s/.." % os.path.dirname(__file__))
from ERA5_load import ERA5_load_T2m
from ERA5_load import ERA5_load_T2m_climatology


opdir = "%s/Proxy_20CR/datasets/ERA5/daily_T2m/Quantiles" % (os.getenv("SCRATCH"),)

if not os.path.isdir(opdir):
    os.makedirs(opdir)

qd = TDigest()
yc = ERA5_load_T2m_climatology(1981, args.month, args.day)
for year in range(1981, 2011):
    yd = ERA5_load_T2m(year, args.month, args.day)
    yd -= yc
    qd.batch_update(yd.data.flatten())
    qd.compress()

pickle.dump(qd, open("%s/total_qd_%02d%02d.pkl" % (opdir, args.month, args.day), "wb"))
