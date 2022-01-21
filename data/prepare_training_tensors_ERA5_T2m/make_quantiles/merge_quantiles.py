#!/usr/bin/env python

# Merge the daily quantiles into an annual one.

import os
import sys
import datetime
import pickle
from tdigest import TDigest

opdir = "%s/Proxy_20CR/datasets/ERA5/daily_T2m/Quantiles" % (os.getenv("SCRATCH"),)

if not os.path.isdir(opdir):
    raise Exception("Daily quantiles not available")


def load_daily(month, day):
    td_file_name = (("%s/total_qd_%02d%02d.pkl")) % (
        opdir,
        month,
        day,
    )
    tdd = pickle.load(open(td_file_name, "rb"))
    return tdd


sd = load_daily(1, 1)
start_day = datetime.date(1981, 1, 2)
end_day = datetime.date(1981, 12, 31)

cday = datetime.date(1981, 1, 2)
while cday.year <= 1982:
    sd += load_daily(cday.month, cday.day)
    cday += datetime.timedelta(days=1)

sd.compress()

pickle.dump(sd, open("%s/total_qd_annual.pkl" % opdir, "wb"))

# Also save a list of the percentile values (1-99)
pcl = []
for p in range(1, 100):
    pcl.append(sd.percentile(p))

pickle.dump(pcl, open("%s/total_pctl.pkl" % opdir, "wb"))
