# Make tf.data.Datasets from 20CRv2c P, U & V fields

import os
import tensorflow as tf
import numpy as np

# Load a pre-standardised MSLP tensor from a file
def load_MSLP_tensor(file_name):
    sict = tf.io.read_file(file_name)
    imt = tf.io.parse_tensor(sict, np.float32)
    imt = tf.reshape(imt, [80, 160, 3])
    return imt


# Get a dataset
def getDataset(purpose, nImages=None):

    # Get a list of filenames containing tensors
    inFiles = os.listdir(
        "%s/Proxy_20CR/datasets/20CR2c/puv/%s" % (os.getenv("SCRATCH"), purpose)
    )

    if nImages is not None:
        if len(inFiles) >= nImages:
            inFiles = inFiles[0:nImages]
        else:
            raise ValueError(
                "Only %d images available, can't provide %d" % (len(inFiles), nImages)
            )

    # Create TensorFlow Dataset object from the file namelist
    inFiles = [
        "%s/Proxy_20CR/datasets/20CR2c/puv/%s/%s"
        % (os.getenv("SCRATCH"), purpose, x)
        for x in inFiles
    ]
    tr_data = tf.data.Dataset.from_tensor_slices(tf.constant(inFiles))

    # Convert the Dataset from file names to file contents
    tr_data = tr_data.map(
        load_MSLP_tensor, num_parallel_calls=tf.data.experimental.AUTOTUNE
    )
    # Optimisation
    tr_data = tr_data.cache()
    tr_data = tr_data.prefetch(tf.data.experimental.AUTOTUNE)

    return tr_data
