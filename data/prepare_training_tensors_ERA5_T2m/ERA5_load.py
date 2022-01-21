# Load daily T2m data from ERA5

import os
import iris
import pickle


# Convert anomalies to the range 0-1 (ish) by replacing each value
#  with its corresponding CDF value
def ERA5_quantile_normalise(f, quantiles=None):
    if quantiles is None:
        quantiles = pickle.load(
            open(
                "%s/Proxy_20CR/datasets/ERA5/daily_T2m/Quantiles/total_pctl.pkl"
                % os.getenv("SCRATCH"),
                "rb",
            )
        )
    res = f.data.copy()
    wkg = f.data[f.data < quantiles[0]]
    if len(wkg) > 0:
        wkg = 1 - (quantiles[0] - wkg) / (quantiles[1] - quantiles[0])
        res[f.data < quantiles[0]] = wkg
    for pct in range(0, 98):
        wkg = f.data[(f.data >= quantiles[pct]) & (f.data < quantiles[pct + 1])]
        if len(wkg) > 0:
            wkg = pct + 1 + (quantiles[pct + 1] - wkg) / (
                quantiles[pct + 1] - quantiles[pct]
            )
            res[(f.data >= quantiles[pct]) & (f.data < quantiles[pct + 1])] = wkg
    wkg = f.data[f.data > quantiles[98]]
    if len(wkg) > 0:
        wkg = 99 + (wkg - quantiles[98]) / (quantiles[98] - quantiles[97])
        res[f.data > quantiles[98]] = wkg
    r = f.copy()
    r.data = res / 100
    return r


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


def ERA5_load_T2m_variability_climatology(year, month, day):
    if month == 2 and day == 29:
        day = 28
    fname = (
        "%s/Proxy_20CR/datasets/ERA5/daily_T2m/"
        + "variability_climatology/%02d/%02d.nc"
    ) % (
        os.getenv("SCRATCH"),
        month,
        day,
    )
    ddata = iris.load_cube(fname)
    return ddata
