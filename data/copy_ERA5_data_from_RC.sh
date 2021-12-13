#!/bin/bash

# We're going to work with ERA5 fields.
# Copy the files Robin Clarke has curated onto $SCRATCH

mkdir -p $SCRATCH/Proxy_20CR/datasets/ERA5

mkdir -p $SCRATCH/Proxy_20CR/datasets/ERA5/daily_Tmax
cp -n /data/users/hadrc/obs/daily/era5_daily_2m_maximum_temperature_19790101to20200831.nc $SCRATCH/Proxy_20CR/datasets/ERA5/daily_Tmax/

mkdir -p $SCRATCH/Proxy_20CR/datasets/ERA5/daily_T2m
cp -n /data/users/hadrc/obs/daily/era5_daily_2m_temperature_19790101to20210831.nc $SCRATCH/Proxy_20CR/datasets/ERA5/daily_T2m/

