# Functions for normalising (and unnormalising) weather data
#  to a range around 0-1

# Each function takes a numpy array as argument and returns a
#  scaled copy of the array. So pass the .data component
#  of an iris Cube.

import numpy as np
import iris
import datetime
import os

# Load the normals
def load_normal(var, year, month, day, hour):
    if month == 2 and day == 29:
        day = 28
    st = datetime.datetime(year, month, day, hour)
    sd = iris.load_cube(
        "%s/20CR/version_3.4.1/normal/%s.nc" % (os.getenv("DATADIR"), var),
        iris.Constraint(
            time=iris.time.PartialDateTime(
                year=1981, month=st.month, day=st.day, hour=st.hour
            )
        ),
    )
    coord_s = iris.coord_systems.GeogCS(iris.fileformats.pp.EARTH_RADIUS)
    sd.coord("latitude").coord_system = coord_s
    sd.coord("longitude").coord_system = coord_s
    return sd

def normalise_insolation(p):
    res = np.copy(p)
    res /= 25
    return res


def unnormalise_insolation(p):
    res = np.copy(p)
    res *= 25
    return res


def normalise_t2m(p):
    res = np.copy(p)
    res -= 280
    res /= 50
    return res


def unnormalise_t2m(p):
    res = np.copy(p)
    res *= 50
    res += 280
    return res


def normalise_wind(p):
    res = np.copy(p)
    res /= 12
    return res


def unnormalise_wind(p):
    res = np.copy(p)
    res *= 12
    return res


def normalise_wind_anomaly(p, c):
    res = np.copy(p)
    res -= c
    res /= 12
    return res


def unnormalise_wind_anomaly(p, c):
    res = np.copy(p)
    res *= 12
    res += c
    return res


def normalise_prmsl(p):
    res = np.copy(p)
    res -= 101325
    res /= 3000
    return res


def unnormalise_prmsl(p):
    res = np.copy(p)
    res *= 3000
    res += 101325
    return res


def normalise_prmsl_anomaly(p, c):
    res = np.copy(p)
    res -= c
    res /= 3000
    return res


def unnormalise_prmsl_anomaly(p, c):
    res = np.copy(p)
    res *= 3000
    res += c
    return res
