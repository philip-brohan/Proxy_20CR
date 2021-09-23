#!/usr/bin/env python

# Convolutional Autoencoder for 20CR2c MSLP

import os
import sys
import time
import tensorflow as tf
import pickle
import argparse

parser = argparse.ArgumentParser()
parser.add_argument(
    "--epoch", help="Restart from epoch", type=int, required=False, default=0
)
args = parser.parse_args()

# Distribute across all GPUs
# strategy = tf.distribute.MirroredStrategy()
# Doesn't yet work - need to change the training loop and datasets to be
#  strategy aware.
strategy = tf.distribute.get_strategy()

# Load the model specification
sys.path.append("%s/." % os.path.dirname(__file__))
from autoencoderModel import DCAE
from autoencoderModel import train_step
from autoencoderModel import compute_loss

# Load the data source provider
sys.path.append("%s/../PRMSL_dataset" % os.path.dirname(__file__))
from makeDataset import getDataset

# How big is the latent space?
latent_dim = 20

# How many images to use?
nTrainingImages = 2568  # Max is 2568
nTestImages = 285  # Max is 285

# How many epochs to train for
nEpochs = 100
# Length of an epoch - if None, same as nTrainingImages
nImagesInEpoch = None

if nImagesInEpoch is None:
    nImagesInEpoch = nTrainingImages

# Dataset parameters
bufferSize = 1000  # Untested
batchSize = 32  # Arbitrary

# Set up the training data
trainingData = getDataset(purpose="training", nImages=nTrainingImages)
trainingData = trainingData.shuffle(bufferSize).batch(batchSize)

# Set up the test data
testData = getDataset(purpose="test", nImages=nTestImages)
testData = testData.batch(batchSize)

# Instantiate the model
with strategy.scope():
    autoencoder = DCAE(latent_dim)
    optimizer = tf.keras.optimizers.Adam(1e-4)
    # If we are doing a restart, load the weights
    if args.epoch > 0:
        weights_dir = ("%s/Proxy_20CR/models/DCAE_single_PRMSL/" + "Epoch_%04d") % (
            os.getenv("SCRATCH"),
            args.epoch - 1,
        )
        load_status = autoencoder.load_weights("%s/ckpt" % weights_dir)
        # Check the load worked
        load_status.assert_existing_objects_matched()


# Save the model weights and the history state after every epoch
history = {}
history["loss"] = []
history["val_loss"] = []


def save_state(model, epoch, loss):
    save_dir = ("%s/Proxy_20CR/models/DCAE_single_PRMSL/" + "Epoch_%04d") % (
        os.getenv("SCRATCH"),
        epoch,
    )
    if not os.path.isdir(save_dir):
        os.makedirs(save_dir)
    model.save_weights("%s/ckpt" % save_dir)
    history["loss"].append(loss)
    # history["val_loss"].append(logs["val_loss"])
    history_file = "%s/history.pkl" % save_dir
    pickle.dump(history, open(history_file, "wb"))


for epoch in range(nEpochs):
    start_time = time.time()
    for train_x in trainingData:
        train_step(autoencoder, train_x, optimizer)
    end_time = time.time()

    train_loss = tf.keras.metrics.Mean()
    for test_x in trainingData:
        train_loss(compute_loss(autoencoder, test_x))
    train_elbo = train_loss.result()
    test_loss = tf.keras.metrics.Mean()
    for test_x in testData:
        test_loss(compute_loss(autoencoder, test_x))
    test_elbo = test_loss.result()
    print(
        "Epoch: {}, Train Loss: {}, Test Loss: {}, time: {}".format(
            epoch, train_elbo, test_elbo, end_time - start_time
        )
    )
    save_state(autoencoder, epoch, test_elbo)
