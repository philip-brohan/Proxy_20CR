# Make tf.data.Datasets from 20CRv3 mslp fields

import os
import tensorflow as tf
import numpy as np

# Load a pre-standardised MSLP tensor from a file
def load_MSLP_tensor(file_name):
    sict = tf.io.read_file(file_name)
    imt = tf.io.parse_tensor(sict, np.float32)
    imt = tf.reshape(imt, [80, 160, 5])
    # Rescale so that all components have about the same variance
    imt = tf.unstack(imt,axis=2)
    imt[1] *= 3
    imt[3] *= 3
    imt[0] *= 2
    imt[4] *= 2
    imt = tf.stack(imt,axis=2)
    return imt


# Get a dataset
def getDataset(purpose, nImages=None):

    # Get a list of filenames containing tensors
    inFiles = os.listdir(
        "%s/Proxy_20CR/datasets/20CR2c/prmsl_24h/%s" % (os.getenv("SCRATCH"), purpose)
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
        "%s/Proxy_20CR/datasets/20CR2c/prmsl_24h/%s/%s"
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
