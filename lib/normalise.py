# Functions for normalising (and unnormalising) weather data
#  to a range around 0-1

# Each function takes a numpy array as argument and returns a
#  scaled copy of the array. So pass the .data component
#  of an iris Cube.

import numpy


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
