# Load daily T2m data from ERA5

import os
import iris


def ERA5_trim(f, x=1440, y=720):
    xmin = f.coord("longitude").points[-x]
    cx = iris.Constraint(longitude=lambda cell: cell >= xmin)
    ymin = f.coord("latitude").points[y - 1]
    cy = iris.Constraint(latitude=lambda cell: cell >= ymin)
    return f.extract(cx & cy)


def ERA5_load_T2m(year, month, day):
    fname = (
        "%s/Proxy_20CR/datasets/ERA5/daily_T2m/"
        + "era5_daily_2m_temperature_19790101to20210831.nc"
    ) % os.getenv("SCRATCH")
    ddata = iris.load_cube(
        fname,
        iris.Constraint(
            time=lambda cell: cell.point.year == year
            and cell.point.month == month
            and cell.point.day == day
        ),
    )
    return ddata


def ERA5_load_T2m_climatology(year, month, day):
    if month == 2 and day == 29:
        day = 28
    fname = ("%s/Proxy_20CR/datasets/ERA5/daily_T2m/" + "climatology/%02d/%02d.nc") % (
        os.getenv("SCRATCH"),
        month,
        day,
    )
    ddata = iris.load_cube(fname)
    return ddata
