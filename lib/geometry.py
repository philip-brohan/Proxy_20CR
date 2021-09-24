# Functions for regridding data

# The 20CRv2c grid has the UK on the grid boundary - I want it in the middle.
# Also we need grid dimensions that match strided conviolutions nicely.

# So adopt a standard analysis grid - equirectangular, with pole at
#  lat 90, lon, 180, and grid size of 80x160.

import iris
import numpy

# Regrid an iris cube onto the standard analysis grid
def to_analysis_grid(cbe):
    # Standard pole with UK at centre
    cs = iris.coord_systems.RotatedGeogCS(90.0, 180.0, 0.0)
    # Latitudes cover -90 to 90 with 80 values
    lat_values = numpy.arange(-90, 91, 180 / 79)
    latitude = iris.coords.DimCoord(
        lat_values, standard_name="latitude", units="degrees_north", coord_system=cs
    )
    # Longitudes cover -180 to 180 with 160 values
    lon_values = numpy.arange(-180, 181, 360 / 159)
    longitude = iris.coords.DimCoord(
        lon_values, standard_name="longitude", units="degrees_east", coord_system=cs
    )
    dummy_data = numpy.zeros((len(lat_values), len(lon_values)))
    dummy_cube = iris.cube.Cube(
        dummy_data, dim_coords_and_dims=[(latitude, 0), (longitude, 1)]
    )
    n_cube = cbe.regrid(dummy_cube, iris.analysis.Linear())
    return n_cube
