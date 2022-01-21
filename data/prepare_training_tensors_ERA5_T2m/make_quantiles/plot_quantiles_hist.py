import os
import sys
import iris

import numpy as np
import matplotlib.pyplot as plt

sys.path.append("%s/.." % os.path.dirname(__file__))
from ERA5_load import ERA5_load_T2m
from ERA5_load import ERA5_load_T2m_climatology
from ERA5_load import ERA5_quantile_normalise

yc = ERA5_load_T2m_climatology(1981, 7, 12)
yd = ERA5_load_T2m(2001, 7, 12)
ya = yd - yc

fig = plt.figure()
ax1 = fig.add_axes([0.15, 0.55, 0.7, 0.3])
n, bins, patches = ax1.hist(yd.data.flatten(), 500)

yn = ERA5_quantile_normalise(ya)

ax2 = fig.add_axes([0.15, 0.1, 0.7, 0.3])
n, bins, patches = ax2.hist(yn.data.flatten(), 500)

plt.show()
