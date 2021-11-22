#!/usr/bin/env python

# Make a few hundred tf data files
#  for training the GCM models.

# Get one data file every 2 days+6 hours over the selected years
#  They should be far enough apart to be mostly independent.

# Partition off 1/10 of them to be test data

# This script does not run the commands - it makes a list of commands
#  (in the file 'run.txt') which can be run in parallel.

import os
import datetime

# Function to check if the job is already done for this timepoint
def is_done(year, month, day, hour, group):
    op_file_name = (
        ("%s/Proxy_20CR/datasets/20CR2c/puv/" + "%s/%04d-%02d-%02d:%02d.tfd")
    ) % (
        os.getenv("SCRATCH"),
        group,
        year,
        month,
        day,
        hour,
    )
    if os.path.isfile(op_file_name):
        return True
    return False


f = open("run.txt", "w+")

start_day = datetime.datetime(1969, 1, 3, 0)
end_day = datetime.datetime(2009, 12, 29, 23)

current_day = start_day
count = 1
while current_day <= end_day:
    if count % 10 == 0:
        if not is_done(
            current_day.year,
            current_day.month,
            current_day.day,
            current_day.hour,
            "test",
        ):
            cmd = (
                "./make_training_tensor.py --year=%d --month=%d"
                + " --day=%d --hour=%d --test \n"
            ) % (
                current_day.year,
                current_day.month,
                current_day.day,
                current_day.hour,
            )
            f.write(cmd)
    else:
        if not is_done(
            current_day.year,
            current_day.month,
            current_day.day,
            current_day.hour,
            "training",
        ):
            cmd = (
                "./make_training_tensor.py --year=%d --month=%d"
                + " --day=%d --hour=%d \n"
            ) % (
                current_day.year,
                current_day.month,
                current_day.day,
                current_day.hour,
            )
            f.write(cmd)
    current_day = current_day + datetime.timedelta(hours=54)
    count += 1

f.close()
