#!/usr/bin/env python

# Make a few thousand tf data files
#  for training the VAE models.

# Get one data file every 5 days

# Partition off 1/10 of them to be test data

# This script does not run the commands - it makes a list of commands
#  (in the file 'run.txt') which can be run in parallel.

import os
import datetime

# Function to check if the job is already done for this timepoint
def is_done(year, month, day, group):
    op_file_name = (
        ("%s/Proxy_20CR/datasets/ERA5/daily_Tmax/" + "%s/%04d-%02d-%02d.tfd")
    ) % (
        os.getenv("SCRATCH"),
        group,
        year,
        month,
        day,
    )
    if os.path.isfile(op_file_name):
        return True
    return False


f = open("run.txt", "w+")

start_day = datetime.date(1979, 1, 1)
end_day = datetime.date(2020, 8, 31)

current_day = start_day
count = 1
while current_day <= end_day:
    if count % 10 == 8:  # To match with hadUKgrid test cases 
        if not is_done(
            current_day.year,
            current_day.month,
            current_day.day,
            "test",
        ):
            cmd = (
                "./make_training_tensor.py --year=%d --month=%d --day=%d --test \n"
            ) % (
                current_day.year,
                current_day.month,
                current_day.day,
            )
            f.write(cmd)
    else:
        if not is_done(
            current_day.year,
            current_day.month,
            current_day.day,
            "training",
        ):
            cmd = ("./make_training_tensor.py --year=%d --month=%d --day=%d \n") % (
                current_day.year,
                current_day.month,
                current_day.day,
            )
            f.write(cmd)
    current_day = current_day + datetime.timedelta(days=1)
    count += 1

f.close()
