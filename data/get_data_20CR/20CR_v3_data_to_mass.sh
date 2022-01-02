#!/usr/bin/bash

# Store downloaded 20CRv3 data on MASS

for year in {1969..2009}
do
for var in PRMSL PRATE TMP2m TMPS UGRD10m VGRD10m observations
do 
/home/h03/hadpb/Projects/20CRv3-diagnostics/tools/extract_data/store_on_mass/v3_release_to_mass.py --year=$year --variable=$var
done
done
 
for year in 1903 1916
do
for var in PRMSL PRATE TMP2m TMPS UGRD10m VGRD10m observations
do 
/home/h03/hadpb/Projects/20CRv3-diagnostics/tools/extract_data/store_on_mass/v3_release_to_mass.py --year=$year --variable=$var
done
done 
