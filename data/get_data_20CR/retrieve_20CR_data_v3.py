#!/usr/bin/env python

# We're going to work with 20CRv3 fields.
# Retrieve the netCDF files from NERSC and store on $SCRATCH

import datetime
import IRData.twcr as twcr
import time


def mfetch(var, dte, version="3"):
    try:
        twcr.fetch(var, dte, version=version)
    except:
        time.sleep(300)
        twcr.fetch(var, dte, version=version)


for year in [1919, 1929, 1939, 1949, 1959]:
    dte = datetime.datetime(year, 1, 1)
    mfetch("observations", dte, version="3")

for year in [1903, 1916]:
    dte = datetime.datetime(year, 1, 1)
    mfetch("PRMSL", dte, version="3")
    mfetch("TMP2m", dte, version="3")
    mfetch("TMPS", dte, version="3")
    mfetch("PRATE", dte, version="3")
    mfetch("UGRD10m", dte, version="3")
    mfetch("VGRD10m", dte, version="3")
    mfetch("observations", dte, version="3")

for year in range(1969, 2010):
    dte = datetime.datetime(year, 1, 1)
    mfetch("PRMSL", dte, version="3")
    mfetch("TMP2m", dte, version="3")
    mfetch("TMPS", dte, version="3")
    mfetch("PRATE", dte, version="3")
    mfetch("UGRD10m", dte, version="3")
    mfetch("VGRD10m", dte, version="3")
    mfetch("observations", dte, version="3")

    mfetch("TMP2m", dte, version="3")
    mfetch("TMPS", dte, version="3")
    mfetch("UGRD10m", dte, version="3")

    mfetch("VGRD10m", dte, version="3")
    mfetch("PRATE", dte, version="3")
    mfetch("observations", dte, version="3")

for year in range(1969, 2010):

    dte = datetime.datetime(year, 1, 1)
    mfetch("PRMSL", dte, version="3")
    mfetch("TMP2m", dte, version="3")
    mfetch("TMPS", dte, version="3")

    mfetch("UGRD10m", dte, version="3")
    mfetch("VGRD10m", dte, version="3")
    mfetch("PRATE", dte, version="3")

    mfetch("observations", dte, version="3")
