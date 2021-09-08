#!/usr/bin/env python

# Make a list of the commands needed to make a few hundred tf data files
#  for training the autoencoder.

# Special case for insolation - all time points, one year.
import os
import datetime

# Function to check if the job is already done for this timepoint
def is_done(year, month, day, hour, group):
    op_file_name = (
        "%s/ML_GCM/datasets/" + "20CR2c/insolation/" + "%s/%04d-%02d-%02d:%02d.tfd"
    ) % (os.getenv("SCRATCH"), group, year, month, day, hour)
    if os.path.isfile(op_file_name):
        return True
    return False


f = open("run.txt", "w+")

count = 1

start_day = datetime.datetime(1969, 1, 1, 0)
end_day = datetime.datetime(1969, 12, 31, 23)

current_day = start_day
while current_day <= end_day:
    if not is_done(
        current_day.year,
        current_day.month,
        current_day.day,
        current_day.hour,
        "training",
    ):
        cmd = (
            "conda activate mlgcm; ./make_insolation_tensor.py --year=%d --month=%d"
            + " --day=%d --hour=%d \n"
        ) % (current_day.year, current_day.month, current_day.day, current_day.hour)
        f.write(cmd)
    current_day = current_day + datetime.timedelta(hours=6)
    count += 1

f.close()
