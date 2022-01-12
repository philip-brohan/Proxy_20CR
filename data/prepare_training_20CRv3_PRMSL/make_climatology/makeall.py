#!/usr/bin/env python

# Make an 20CRv3 PRMSL climatology file for each 3-hourly period

# This script does not run the commands - it makes a list of commands
#  (in the file 'run.txt') which can be run in parallel.

import os
import datetime

# Function to check if the job is already done for this timepoint
def is_done(month, day, hour):
    op_file_name = (
        ("%s/Proxy_20CR/datasets/20CRv3/PRMSL/climatology/%02d/%02d/%02d.nc")
    ) % (
        os.getenv("SCRATCH"),
        month,
        day,
        hour,
    )
    if os.path.isfile(op_file_name):
        return True
    return False


f = open("run.txt", "w+")

start_day = datetime.datetime(1981, 1, 1, 0)
end_day = datetime.datetime(1981, 12, 31, 21)

current_day = start_day
while current_day <= end_day:
    if not is_done(
        current_day.month,
        current_day.day,
        current_day.hour,
    ):
        cmd = ("./make_climatology_day.py --month=%d --day=%d --hour=%d\n") % (
            current_day.month,
            current_day.day,
            current_day.hour,
        )
        f.write(cmd)
    current_day = current_day + datetime.timedelta(hours=3)

f.close()
