#!/bin/bash

# We're going to work with ERA5 fields.
# Copy the files Robin Clarke has curated onto $SCRATCH

mkdir -p $SCRATCH/Proxy_20CR/datasets/haduk-grid

mkdir -p $SCRATCH/Proxy_20CR/datasets/haduk-grid/series_archive_provisional/grid/daily_maxtemp/

for year in 2019 2020
do
for month in 01 02 03 04 05 06 07 08 09 10 11 12
do
mkdir -p $SCRATCH/Proxy_20CR/datasets/haduk-grid/series_archive_provisional/grid/daily_maxtemp/$year/$month
cp -rn /data/users/haduk/uk_climate_data/supported/haduk-grid/series_archive_provisional/grid/daily_maxtemp/$year/$month $SCRATCH/Proxy_20CR/datasets/haduk-grid/series_archive_provisional/grid/daily_maxtemp/$year
done
done

for year in {1969..2018}
do
for month in 01 02 03 04 05 06 07 08 09 10 11 12
do
mkdir -p $SCRATCH/Proxy_20CR/datasets/haduk-grid/v1.0.3.0/grid/daily_maxtemp/$year/$month
cp -rn /data/users/haduk/uk_climate_data/supported/haduk-grid/v1.0.3.0/data/grid_archives/series_archive/grid/daily_maxtemp/$year/$month $SCRATCH/Proxy_20CR/datasets/haduk-grid/v1.0.3.0/grid/daily_maxtemp/$year
done
done

