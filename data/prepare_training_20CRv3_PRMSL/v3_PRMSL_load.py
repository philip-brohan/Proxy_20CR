# Load 20CRv3 PRMSL (ensemble member 1)

import os
import iris
import datetime
import IRData.twcr as twcr


def v3_roll_longitude(f):
    f1 = f.extract(iris.Constraint(longitude=lambda cell: cell > 180))
    longs = f1._dim_coords_and_dims[1][0].points - 360
    f1._dim_coords_and_dims[1][0].points = longs
    f2 = f.extract(iris.Constraint(longitude=lambda cell: cell <= 180))
    f = iris.cube.CubeList([f1, f2]).concatenate_cube()
    return f


def ERA5_load_LS_mask():
    fname = (
        "%s/Proxy_20CR/datasets/ERA5/daily_SST/"
        + "era5_daily_0m_sea_surface_temperature_19790101to20210831.nc"
    ) % os.getenv("SCRATCH")
    ddata = iris.load_cube(
        fname,
        iris.Constraint(
            time=lambda cell: cell.point.year == 1979
            and cell.point.month == 1
            and cell.point.day == 1
        ),
    )
    ddata.data = ddata.data.data  # Remove mask
    ddata.data[ddata.data < 10000] = 0
    ddata.data[ddata.data > 0] = 1
    return ddata


def v3_load_PRMSL(year, month, day, hour):
    ddata = twcr.load(
        "PRMSL", datetime.datetime(year, month, day, hour), version="3", member=1
    )
    return ddata


def v3_load_PRMSL_climatology(year, month, day, hour):
    if month == 2 and day == 29:
        day = 28
    fname = (
        "%s/Proxy_20CR/datasets/20CRv3/PRMSL/" + "climatology/%02d/%02d/%02d.nc"
    ) % (
        os.getenv("SCRATCH"),
        month,
        day,
        hour,
    )
    ddata = iris.load_cube(fname)
    return ddata


def v3_load_PRMSL_variability_climatology(year, month, day, hour):
    if month == 2 and day == 29:
        day = 28
    fname = (
        "%s/Proxy_20CR/datasets/20CRv3/PRMSL/"
        + "variability_climatology/%02d/%02d/%02d.nc"
    ) % (
        os.getenv("SCRATCH"),
        month,
        day,
        hour,
    )
    ddata = iris.load_cube(fname)
    return ddata
