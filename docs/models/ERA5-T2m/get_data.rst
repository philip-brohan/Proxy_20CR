ERA5 T2m - get the training data
================================

Ideally I'd put in here a script to download the training data to use. This time, however, I cheated and used pre-prepared data provided by a colleague at the Met Office (thanks Robin). So you are on your own for this one.

You can get the data I used from the `Copernicus climate Data Store <https://cds.climate.copernicus.eu/cdsapp#!/dataset/reanalysis-era5-single-levels?tab=overview>`_ but I don't have a pre-prepared script for it. (Download the hourly data at 0.25 degree resolution, and average into daily means). 

Once you've got the data as a local netCDF file you are going to need to make a climatology (our VAE works on anomalies) and a variability climatology (comparing model uncertainty to expected climatological variability tells us where we have skill).

Utility script for data access (imported in most other scripts using this data):

.. literalinclude:: ../../../data/prepare_training_tensors_ERA5_T2m/ERA5_load.py

Scripts to make the climatology:

.. literalinclude:: ../../../data/prepare_training_tensors_ERA5_T2m/make_climatology/make_climatology_day.py

.. literalinclude:: ../../../data/prepare_training_tensors_ERA5_T2m/make_climatology/makeall.py

Scripts to make the variability climatology:

.. literalinclude:: ../../../data/prepare_training_tensors_ERA5_T2m/make_variability_climatology/make_climatology_day.py

.. literalinclude:: ../../../data/prepare_training_tensors_ERA5_T2m/make_variability_climatology/makeall.py

