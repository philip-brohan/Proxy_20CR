#!/usr/bin/env python

# Plot an example pressure (& obs) plot from v2c

import sys
import os
import IRData.twcr as twcr
import datetime

import iris
import numpy as np
import scipy

# Fix dask SPICE bug
import dask

dask.config.set(scheduler="single-threaded")

import matplotlib
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
from matplotlib.figure import Figure
from matplotlib.patches import Rectangle
from matplotlib.lines import Line2D
import matplotlib.colors
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--year", help="Year", type=int, required=True)
parser.add_argument("--month", help="Integer month", type=int, required=True)

parser.add_argument("--day", help="Day of month", type=int, required=True)
parser.add_argument(
    "--hour", help="Time of day (0 to 23.99)", type=float, required=True
)
parser.add_argument(
    "--pole_latitude",
    help="Latitude of projection pole",
    default=90,
    type=float,
    required=False,
)
parser.add_argument(
    "--pole_longitude",
    help="Longitude of projection pole",
    default=180,
    type=float,
    required=False,
)
parser.add_argument(
    "--npg_longitude",
    help="Longitude of view centre",
    default=0,
    type=float,
    required=False,
)
parser.add_argument(
    "--zoom",
    help="Scale factor for viewport (1=global)",
    default=1,
    type=float,
    required=False,
)
parser.add_argument(
    "--opdir",
    help="Directory for output files",
    default="%s/Proxy_20CR/images/v2c/spread" % os.getenv("SCRATCH"),
    type=str,
    required=False,
)

args = parser.parse_args()
if not os.path.isdir(args.opdir):
    os.makedirs(args.opdir)

dte = datetime.datetime(
    args.year, args.month, args.day, int(args.hour), int(args.hour % 1 * 60)
)

# Land-sea mask
mask = iris.load_cube(
    "%s/fixed_fields/land_mask/opfc_global_2019.nc" % os.getenv("DATADIR")
)

# Load the model data
prmsl = twcr.load("prmsl", dte, version="2c")
obs = twcr.load_observations_fortime(dte, version="2c")

# Load the normals
def load_normal(var, year, month, day, hour):
    if month == 2 and day == 29:
        day = 28
    prevt = datetime.datetime(year, month, day, hour)
    prevcsd = iris.load_cube(
        "%s/20CR/version_3.4.1/normal/%s.nc" % (os.getenv("DATADIR"), var),
        iris.Constraint(
            time=iris.time.PartialDateTime(
                year=1981, month=prevt.month, day=prevt.day, hour=prevt.hour
            )
        ),
    )
    coord_s = iris.coord_systems.GeogCS(iris.fileformats.pp.EARTH_RADIUS)
    prevcsd.coord("latitude").coord_system = coord_s
    prevcsd.coord("longitude").coord_system = coord_s
    return prevcsd


dtp = datetime.datetime(args.year, args.month, args.day, int(args.hour / 6) * 6)
npf = load_normal("prmsl", dtp.year, dtp.month, dtp.day, dtp.hour)
dtn = dtp + datetime.timedelta(hours=6)
nnf = load_normal("prmsl", dtn.year, dtn.month, dtn.day, dtn.hour)
nnf.attributes = npf.attributes
ncl = iris.cube.CubeList((npf, nnf)).merge_cube()
if dte.month == 2 and dte.day == 29:
    dti = datetime.datetime(1981, 2, 28, dte.hour, dte.minute)
else:
    dti = datetime.datetime(1981, dte.month, dte.day, dte.hour, dte.minute)
normals = ncl.interpolate([("time", dti)], iris.analysis.Linear())

# Dummy cube for plotting
def plot_cube(
    resolution,
    xmin=-180,
    xmax=180,
    ymin=-90,
    ymax=90,
    pole_latitude=90,
    pole_longitude=180,
    npg_longitude=0,
):

    cs = iris.coord_systems.RotatedGeogCS(pole_latitude, pole_longitude, npg_longitude)
    lat_values = np.arange(ymin, ymax + resolution, resolution)
    latitude = iris.coords.DimCoord(
        lat_values, standard_name="latitude", units="degrees_north", coord_system=cs
    )
    lon_values = np.arange(xmin, xmax + resolution, resolution)
    longitude = iris.coords.DimCoord(
        lon_values, standard_name="longitude", units="degrees_east", coord_system=cs
    )
    dummy_data = np.zeros((len(lat_values), len(lon_values)))
    plot_cube = iris.cube.Cube(
        dummy_data, dim_coords_and_dims=[(latitude, 0), (longitude, 1)]
    )
    return plot_cube


pc = plot_cube(
    0.25,
    pole_latitude=args.pole_latitude,
    pole_longitude=args.pole_longitude,
    npg_longitude=args.npg_longitude,
)
cs = iris.coord_systems.RotatedGeogCS(
    args.pole_latitude, args.pole_longitude, args.npg_longitude
)

# Define the figure (page size, background color, resolution, ...
fig = Figure(
    figsize=(19.2, 10.8),  # Width, Height (inches)
    dpi=100,
    facecolor=(0.5, 0.5, 0.5, 1),
    edgecolor=None,
    linewidth=0.0,
    frameon=False,  # Don't draw a frame
    subplotpars=None,
    tight_layout=None,
)
fig.set_frameon(False)
# Attach a canvas
canvas = FigureCanvas(fig)
ax = fig.add_axes([0, 0, 1, 1])
ax.set_axis_off()
# Background (sea) colour
ax.add_patch(Rectangle((0, 0), 1, 1, facecolor=(0.6, 0.6, 0.6, 1), fill=True, zorder=1))
ax.set_xlim(-180 / args.zoom, 180 / args.zoom)
ax.set_ylim(-90 / args.zoom, 90 / args.zoom)
ax.set_aspect("auto")

# Background (sea) colour
ax.add_patch(
    Rectangle(
        (-180 / args.zoom, -90 / args.zoom),
        360 / args.zoom,
        180 / args.zoom,
        facecolor=(0.8, 0.8, 0.8, 1),
        fill=True,
        zorder=1,
    )
)

# Land
mask = mask.regrid(pc, iris.analysis.Linear())
lats = mask.coord("latitude").points
lons = mask.coord("longitude").points
mask_img = ax.pcolorfast(
    lons,
    lats,
    mask.data,
    cmap=matplotlib.colors.ListedColormap(((0.6, 0.6, 0.6, 0), (0.6, 0.6, 0.6, 1))),
    vmin=0,
    vmax=1,
    alpha=1.0,
    zorder=20,
)


# Plot ensemble mean contours, using transparency as an uncertainty indicator
def plot_mean_spread(
    ax,
    pe,
    levels=np.arange(870, 1050, 10),
    mccmap=None,
    threshold=0.05,
    vmax=0.4,
    zorder=100,
    line_threshold=None,
    label=True,
):

    if mccmap is None:
        mccmap = matplotlib.colors.LinearSegmentedColormap(
            "mc_cmap",
            {
                "red": ((0.0, 0.0, 0.0), (1.0, 0.0, 0.0)),
                "blue": ((0.0, 0.3, 0.3), (1.0, 0.3, 0.3)),
                "green": ((0.0, 0.0, 0.0), (1.0, 0.0, 0.0)),
                "alpha": ((0.0, 0.0, 0.0), (1.0, 0.75, 0.75)),
            },
        )

    pe_m = pe.collapsed("member", iris.analysis.MEAN)
    pe_s = pe.collapsed("member", iris.analysis.STD_DEV)

    # Estimate, at each point, the probability that a contour goes through it.
    pe_u = pe_m.copy()
    pe_u.data = pe_m.data * 0.0
    pe_t = pe_u.copy()
    for level in levels:
        pe_t.data = 1 - scipy.stats.norm.cdf(
            np.absolute(pe_m.data - level) / pe_s.data
        )
        pe_u.data = np.maximum(pe_u.data, pe_t.data)
    # Plot this probability as a colormap
    lats = pe_u.coord("latitude").points
    lons = pe_u.coord("longitude").points
    u_img = ax.pcolorfast(
        lons,
        lats,
        pe_u.data,
        cmap=mccmap,
        vmin=threshold / 2.0 - 0.01,
        vmax=vmax,
        zorder=zorder,
    )

    # Generate the mean contour lines, but don't draw them (linewidth=0)
    CS = ax.contour(
        lons,
        lats,
        pe_m.data,
        colors="black",
        linewidths=0,
        alpha=1,
        levels=levels,
        zorder=zorder,
    )

    # Label the mean contours - transparency dependent on spread
    interpolator = iris.analysis.Linear().interpolator(pe_s, ["latitude", "longitude"])
    if label:
        cl = ax.clabel(CS, inline=1, fontsize=12, fmt="%d", zorder=zorder + 10)
        if line_threshold is not None:
            for ilabel in cl:
                pos = ilabel.get_position()
                local_spread = interpolator([pos[1], pos[0]]).data
                alpha_s = np.sqrt(max(0.04, 1 - local_spread / line_threshold))
                ilabel.set_alpha(alpha_s)

    # Draw the mean contours, with transparency dependent on spread
    base_col = matplotlib.colors.colorConverter.to_rgb("blue")
    for collection in CS.collections:
        segments = collection.get_segments()
        for segment in segments:
            for idx in range(segment.shape[0] - 1):
                alpha_s = 1
                if line_threshold is not None:
                    local_spread = interpolator(
                        [
                            (segment[idx, 1] + segment[idx + 1, 1]) / 2.0,
                            (segment[idx, 0] + segment[idx + 1, 0]) / 2.0,
                        ]
                    ).data
                    alpha_s = np.sqrt(max(0.04, 1 - local_spread / line_threshold))
                clr = (base_col[0], base_col[1], base_col[2], alpha_s)
                ax.add_line(
                    matplotlib.lines.Line2D(
                        xdata=segment[idx : (idx + 2), 0],
                        ydata=segment[idx : (idx + 2), 1],
                        linestyle="solid",
                        linewidth=1.0,
                        color=clr,
                        zorder=zorder + 10,
                    )
                )

    return CS


# PRMSL contours
prmsl = prmsl.regrid(pc, iris.analysis.Linear())
normals = normals.regrid(pc, iris.analysis.Linear())
prmsl -= normals
CS = plot_mean_spread(ax, prmsl*0.01,levels=np.arange(-50,50,7.5),label=False,line_threshold=None,threshold=0.0,vmax=0.25)

# Plot the observations
for i in range(0, len(obs["Longitude"].values)):
    weight = 0.85
    if "weight" in obs.columns:
        weight = obs["weight"].values[i]
    rp = iris.analysis.cartography.rotate_pole(
        np.array(obs["Longitude"].values[i]),
        np.array(obs["Latitude"].values[i]),
        args.pole_longitude,
        args.pole_latitude,
    )
    nlon = rp[0][0]
    nlat = rp[1][0]
    ax.add_patch(
        matplotlib.patches.Circle(
            (nlon, nlat),
            radius=0.3,
            facecolor="black",
            edgecolor="black",
            linewidth=0.1,
            alpha=weight,
            zorder=180,
        )
    )

# Label with the date
ax.text(
    180 / args.zoom - (360 / args.zoom) * 0.009,
    90 / args.zoom - (180 / args.zoom) * 0.016,
    "%04d-%02d-%02d" % (args.year, args.month, args.day),
    horizontalalignment="right",
    verticalalignment="top",
    color="black",
    bbox=dict(
        facecolor=(0.8, 0.8, 0.8, 0.8), edgecolor="black", boxstyle="round", pad=0.5
    ),
    size=14,
    clip_on=True,
    zorder=500,
)

# Make the figure
fig.savefig(
    "%s/%04d%02d%02d%02d%02d.png"
    % (
        args.opdir,
        args.year,
        args.month,
        args.day,
        int(args.hour),
        int(args.hour % 1 * 60),
    )
)
