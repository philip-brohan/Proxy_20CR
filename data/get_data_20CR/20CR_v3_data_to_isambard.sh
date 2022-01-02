#!/usr/bin/bash

# Copy stored 20CRv3 data (ensemble member 1 only) to isambard

for year in {1969..2009}
do
ssh login.isambard "mkdir -p SCRATCH/20CR/version_3/$year"
for var in PRMSL PRATE TMP2m TMPS UGRD10m VGRD10m
do 
rsync -av --ignore-existing $SCRATCH/20CR/version_3/$year/$var.$year\_mem001.nc login.isambard:SCRATCH/20CR/version_3/$year
done
done
 
for year in 1903 1916
do
ssh login.isambard "mkdir -p SCRATCH/20CR/version_3/$year"
for var in PRMSL PRATE TMP2m TMPS UGRD10m VGRD10m
do 
rsync -av --ignore-existing $SCRATCH/20CR/version_3/$year/$var.$year\_mem001.nc login.isambard:SCRATCH/20CR/version_3/$year
done
done

rsync -av --ignore-existing $SCRATCH/20CR/version_3/observations login.isambard:SCRATCH/20CR/version_3/
