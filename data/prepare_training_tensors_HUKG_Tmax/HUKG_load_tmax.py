# Load daily tmax data from HadUKGrid

import os
import iris
import calendar
import datetime


def HUKG_trim(f, x=896, y=1440):
    xmin = f.coord("projection_x_coordinate").points[-x]
    cx = iris.Constraint(projection_x_coordinate=lambda cell: cell >= xmin)
    ymin = f.coord("projection_y_coordinate").points[-y]
    cy = iris.Constraint(projection_y_coordinate=lambda cell: cell >= ymin)
    return f.extract(cx & cy)


def HUKG_load_tmax(year, month, day):
    if year > 2018:
        dirname = (
            "%s/Proxy_20CR/datasets/haduk-grid/"
            + "series_archive_provisional/grid/daily_maxtemp/%04d/%02d"
        ) % (os.getenv("SCRATCH"), year, month)
    else:
        dirname = (
            "%s/Proxy_20CR/datasets/haduk-grid/"
            + "v1.0.3.0/grid/daily_maxtemp/%04d/%02d"
        ) % (os.getenv("SCRATCH"), year, month)
    filename = "%02d.nc" % day
    hdata = iris.load_cube("%s/%s" % (dirname, filename))
    return hdata


# Get an approximate daily climatology from the HadUKGrid monthlies
def HUKG_load_tmax_climatology(year, month, day):
    dirname = (
        "%s/Proxy_20CR/datasets/haduk-grid/v1.0.3.0/"
        + "monthly_maxtemp_climatology/"
        + "1981-2010/"
    ) % os.getenv("SCRATCH")
    if day == 15:
        filename = "%s.nc" % calendar.month_abbr[month].lower()
        hdata = iris.load_cube("%s/%s" % (dirname, filename))
    elif day < 15:
        dte = datetime.date(year, month, day)
        dt2 = datetime.date(year, month, 15)
        dt1 = dt2 - datetime.timedelta(days=30)
        dt1 = datetime.date(dt1.year, dt1.month, 15)
        fn1 = "%s.nc" % calendar.month_abbr[dt1.month].lower()
        hdata = iris.load_cube("%s/%s" % (dirname, fn1))
        fn2 = "%s.nc" % calendar.month_abbr[dt2.month].lower()
        hd2 = iris.load_cube("%s/%s" % (dirname, fn2))
        weight = (dt2 - dte).total_seconds() / (dt2 - dt1).total_seconds()
        hdata.data = hdata.data * weight + hd2.data * (1 - weight)
    else:
        dte = datetime.date(year, month, day)
        dt1 = datetime.date(year, month, 15)
        dt2 = dt1 + datetime.timedelta(days=30)
        dt2 = datetime.date(dt2.year, dt2.month, 15)
        fn1 = "%s.nc" % calendar.month_abbr[month].lower()
        hdata = iris.load_cube("%s/%s" % (dirname, fn1))
        fn2 = "%s.nc" % calendar.month_abbr[dt2.month].lower()
        hd2 = iris.load_cube("%s/%s" % (dirname, fn2))
        weight = (dt2 - dte).total_seconds() / (dt2 - dt1).total_seconds()
        hdata.data = hdata.data * weight + hd2.data * (1 - weight)
    return hdata
