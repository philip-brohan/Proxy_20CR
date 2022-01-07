#!/bin/bash

# We're going to work with haduk-grid fields.
# Copy the files onto $SCRATCH

mkdir -p $SCRATCH/Proxy_20CR/datasets/haduk-grid

mkdir -p $SCRATCH/Proxy_20CR/datasets/haduk-grid/series_archive_provisional/grid/daily_maxtemp/

# Recent years
for year in 2019 2020
do
for month in 01 02 03 04 05 06 07 08 09 10 11 12
do
mkdir -p $SCRATCH/Proxy_20CR/datasets/haduk-grid/series_archive_provisional/grid/daily_maxtemp/$year/$month
cp -rn /data/users/haduk/uk_climate_data/supported/haduk-grid/series_archive_provisional/grid/daily_maxtemp/$year/$month $SCRATCH/Proxy_20CR/datasets/haduk-grid/series_archive_provisional/grid/daily_maxtemp/$year
mkdir -p $SCRATCH/Proxy_20CR/datasets/haduk-grid/series_archive_provisional/station/daily_maxtemp/$year/$month
cp -rn /data/users/haduk/uk_climate_data/supported/haduk-grid/series_archive_provisional/station/daily_maxtemp/$year/$month $SCRATCH/Proxy_20CR/datasets/haduk-grid/series_archive_provisional/station/daily_maxtemp/$year
done
done

# Archive years
for year in {1969..2018}
do
for month in 01 02 03 04 05 06 07 08 09 10 11 12
do
mkdir -p $SCRATCH/Proxy_20CR/datasets/haduk-grid/v1.0.3.0/grid/daily_maxtemp/$year/$month
cp -rn /data/users/haduk/uk_climate_data/supported/haduk-grid/v1.0.3.0/data/grid_archives/series_archive/grid/daily_maxtemp/$year/$month $SCRATCH/Proxy_20CR/datasets/haduk-grid/v1.0.3.0/grid/daily_maxtemp/$year
mkdir -p $SCRATCH/Proxy_20CR/datasets/haduk-grid/v1.0.3.0/station/daily_maxtemp/$year/$month
cp -rn /data/users/haduk/uk_climate_data/supported/haduk-grid/v1.0.3.0/data/grid_archives/series_archive/station/daily_maxtemp/$year/$month $SCRATCH/Proxy_20CR/datasets/haduk-grid/v1.0.3.0/station/daily_maxtemp/$year
done
done

# Climatology
mkdir -p $SCRATCH/Proxy_20CR/datasets/haduk-grid/v1.0.3.0/monthly_maxtemp_climatology
cp -rn /data/users/haduk/uk_climate_data/supported/haduk-grid/v1.0.3.0/data/grid_archives/lta_archive_v1/grid/monthly_maxtemp_climatology/1981-2010 $SCRATCH/Proxy_20CR/datasets/haduk-grid/v1.0.3.0/monthly_maxtemp_climatology/
