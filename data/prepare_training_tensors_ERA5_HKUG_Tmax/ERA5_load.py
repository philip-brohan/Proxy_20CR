# Load daily Tmax data from ERA5

import os
import iris

coord_s = iris.coord_systems.GeogCS(iris.fileformats.pp.EARTH_RADIUS)


def ERA5_roll_longitude(f):
    f1 = f.extract(iris.Constraint(longitude=lambda cell: cell > 180))
    longs = f1._dim_coords_and_dims[1][0].points - 360
    f1._dim_coords_and_dims[1][0].points = longs
    f2 = f.extract(iris.Constraint(longitude=lambda cell: cell <= 180))
    f = iris.cube.CubeList([f1, f2]).concatenate_cube()
    return f


def ERA5_trim(f, x=1440, y=720):
    xmin = f.coord("longitude").points[-x]
    cx = iris.Constraint(longitude=lambda cell: cell >= xmin)
    ymin = f.coord("latitude").points[y - 1]
    cy = iris.Constraint(latitude=lambda cell: cell >= ymin)
    return f.extract(cx & cy)


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


def ERA5_load_Tmax(year, month, day):
    fname = (
        "%s/Proxy_20CR/datasets/ERA5/daily_Tmax/"
        + "era5_daily_2m_maximum_temperature_19790101to20200831.nc"
    ) % os.getenv("SCRATCH")
    ddata = iris.load_cube(
        fname,
        iris.Constraint(
            time=lambda cell: cell.point.year == year
            and cell.point.month == month
            and cell.point.day == day
        ),
    )
    ddata.coord("latitude").coord_system = coord_s
    ddata.coord("longitude").coord_system = coord_s
    return ddata


def ERA5_load_Tmax_climatology(year, month, day):
    if month == 2 and day == 29:
        day = 28
    fname = ("%s/Proxy_20CR/datasets/ERA5/daily_Tmax/" + "climatology/%02d/%02d.nc") % (
        os.getenv("SCRATCH"),
        month,
        day,
    )
    ddata = iris.load_cube(fname)
    ddata.coord("latitude").coord_system = coord_s
    ddata.coord("longitude").coord_system = coord_s
    return ddata


def ERA5_load_Tmax_variability_climatology(year, month, day):
    if month == 2 and day == 29:
        day = 28
    fname = (
        "%s/Proxy_20CR/datasets/ERA5/daily_Tmax/"
        + "variability_climatology/%02d/%02d.nc"
    ) % (
        os.getenv("SCRATCH"),
        month,
        day,
    )
    ddata = iris.load_cube(fname)
    return ddata
