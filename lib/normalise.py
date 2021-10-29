# Functions for normalising (and unnormalising) weather data
#  to a range around 0-1

# Each function takes a numpy array as argument and returns a
#  scaled copy of the array. So pass the .data component
#  of an iris Cube.

import numpy
import iris
import datetime
import os

# Load the normals
def load_normal(var,year,month,day,hour):
    if month==2 and day==29:
        day=28
    prevt = datetime.datetime(
        year, month, day, hour
    )
    prevcsd = iris.load_cube(
        "%s/20CR/version_3.4.1/normal/%s.nc" % (var,os.getenv('DATADIR')),
        iris.Constraint(
            time=iris.time.PartialDateTime(
                year=1981, month=prevt.month, day=prevt.day, hour=prevt.hour
            )
        ),
    )
    coord_s = iris.coord_systems.GeogCS(iris.fileformats.pp.EARTH_RADIUS)
    prevcsd.coord("latitude").coord_system = coord_s
    prevcsd.coord("longitude").coord_system = coord_s
    return(prevcsd)

def normalise_insolation(p):
    res = numpy.copy(p)
    res /= 25
    return res


def unnormalise_insolation(p):
    res = numpy.copy(p)
    res *= 25
    return res


def normalise_t2m(p):
    res = numpy.copy(p)
    res -= 280
    res /= 50
    return res


def unnormalise_t2m(p):
    res = numpy.copy(p)
    res *= 50
    res += 280
    return res


def normalise_wind(p):
    res = numpy.copy(p)
    res /= 12
    return res


def unnormalise_wind(p):
    res = numpy.copy(p)
    res *= 12
    return res


def normalise_prmsl(p):
    res = numpy.copy(p)
    res -= 101325
    res /= 3000
    return res


def unnormalise_prmsl(p):
    res = numpy.copy(p)
    res *= 3000
    res += 101325
    return res

def normalise_prmsl_anomaly(p,c):
    res = numpy.copy(p)
    res -= c
    res /= 3000
    return res


def unnormalise_prmsl_anomaly(p,c):
    res = numpy.copy(p)
    res *= 3000
    res += c
    return res
