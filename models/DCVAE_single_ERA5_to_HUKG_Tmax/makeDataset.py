# Make tf.data.Datasets from ERA5 Tmax fields (source) and
#   haduk-grid Tmax fields (target)

import os
import tensorflow as tf
import numpy as np

# Load a pre-standardised tensor from a file
def load_tensor(file_name):
    sict = tf.io.read_file(file_name)
    imt = tf.io.parse_tensor(sict, np.float32)
    imt = tf.reshape(imt, [1440, 896, 1])
    return imt


# Get a dataset
def getDataset(purpose, nImages=None):

    # Get a list of filenames containing tensors
    hinFiles = os.listdir(
        "%s/Proxy_20CR/datasets/haduk-grid/daily_maxtemp/%s"
        % (os.getenv("SCRATCH"), purpose)
    )
    einFiles = os.listdir(
        "%s/Proxy_20CR/datasets/ERA5/daily_Tmax/%s"
        % (os.getenv("SCRATCH"), purpose)
    )
    inFiles = [value for value in einFiles if value in hinFiles]

    if nImages is not None:
        if len(inFiles) >= nImages:
            inFiles = inFiles[0:nImages]
        else:
            raise ValueError(
                "Only %d images available, can't provide %d" % (len(inFiles), nImages)
            )

    # Create TensorFlow Dataset object from the file namelist
    hinFiles = [
        "%s/Proxy_20CR/datasets/haduk-grid/daily_maxtemp/%s/%s"
        % (os.getenv("SCRATCH"), purpose, x)
        for x in inFiles
    ]
    htr_data = tf.data.Dataset.from_tensor_slices(tf.constant(hinFiles))
    htr_data = htr_data.map(
        load_tensor, num_parallel_calls=tf.data.experimental.AUTOTUNE
    )
    einFiles = [
        "%s/Proxy_20CR/datasets/ERA5/daily_Tmax/%s/%s"
        % (os.getenv("SCRATCH"), purpose, x)
        for x in inFiles
    ]
    etr_data = tf.data.Dataset.from_tensor_slices(tf.constant(einFiles))
    etr_data = etr_data.map(
        load_tensor, num_parallel_calls=tf.data.experimental.AUTOTUNE
    )
    tr_data = tf.data.Dataset.zip((etr_data,htr_data))

    # Optimisation
    tr_data = tr_data.cache()
    tr_data = tr_data.prefetch(tf.data.experimental.AUTOTUNE)

    return tr_data
