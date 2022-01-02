#!/usr/bin/env python

# We're going to work with 20CR2c fields.
# Retrieve the netCDF files from NERSC and store on $SCRATCH

import os
import datetime
import IRData.twcr as twcr
import urllib.request

for year in [1903, 1916]:
    # 2c is in 1 year batches so month and day don't matter
    dte = datetime.datetime(year, 1, 1)
    twcr.fetch("prmsl", dte, version="2c")
    twcr.fetch("air.2m", dte, version="2c")
    twcr.fetch("tsfc", dte, version="2c")
    twcr.fetch("uwnd.10m", dte, version="2c")
    twcr.fetch("vwnd.10m", dte, version="2c")
    twcr.fetch("prate", dte, version="2c")
    twcr.fetch("observations", dte, version="2c")

for year in range(1969, 2010):
    # 2c is in 1 year batches so month and day don't matter
    dte = datetime.datetime(year, 1, 1)
    twcr.fetch("prmsl", dte, version="2c")
    twcr.fetch("air.2m", dte, version="2c")
    twcr.fetch("tsfc", dte, version="2c")
    twcr.fetch("uwnd.10m", dte, version="2c")
    twcr.fetch("vwnd.10m", dte, version="2c")
    twcr.fetch("prate", dte, version="2c")
    twcr.fetch("observations", dte, version="2c")

# Also need one year of insolation data
# I'd like TOA incoming shortwave, but clear sky at surface
#  is close enough.
# This is only available as ensemble means
srce = (
    "ftp://ftp.cdc.noaa.gov/Datasets/20thC_ReanV2c/"
    + "gaussian/monolevel/cduvb.1969.nc"
)
dst = "%s/20CR/version_2c/ensmean/cduvb.1969.nc" % os.getenv("SCRATCH")
if not os.path.exists(dst):
    if not os.path.isdir(os.path.dirname(dst)):
        os.makedirs(os.path.dirname(dst))
    urllib.request.urlretrieve(srce, dst)
