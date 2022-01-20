#!/usr/bin/env python

# Make an ER5 T2m quantile tdigest file for each day

# This script does not run the commands - it makes a list of commands
#  (in the file 'run.txt') which can be run in parallel.

import os
import datetime

# Function to check if the job is already done for this timepoint
def is_done(month, day):
    op_file_name = (
        ("%s/Proxy_20CR/datasets/ERA5/daily_T2m/Quantiles/total_qd_%02d%02d.pkl")
    ) % (
        os.getenv("SCRATCH"),
        month,
        day,
    )
    if os.path.isfile(op_file_name):
        return True
    return False


f = open("run.txt", "w+")

start_day = datetime.date(1981, 1, 1)
end_day = datetime.date(1981, 12, 31)

current_day = start_day
while current_day <= end_day:
    if not is_done(
        current_day.month,
        current_day.day,
    ):
        cmd = ("./make_quantile_day.py --month=%d --day=%d \n") % (
            current_day.month,
            current_day.day,
        )
        f.write(cmd)
    current_day = current_day + datetime.timedelta(days=1)

f.close()
