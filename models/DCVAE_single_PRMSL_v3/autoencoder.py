#!/usr/bin/env python

# Convolutional Variational Autoencoder for 20CRv3 MSLP

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
from autoencoderModel import DCVAE
from autoencoderModel import train_step
from autoencoderModel import compute_loss

# Load the data source provider
sys.path.append("%s/." % os.path.dirname(__file__))
from makeDataset import getDataset

# How many images to use?
nTrainingImages = 11981  # Max is 11981
nTestImages = 1331  # Max is 1331

# How many epochs to train for
nEpochs = 201
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

# Subset of the training data for metrics
validationData = getDataset(purpose="training", nImages=nTestImages).batch(batchSize)

# Set up the test data
testData = getDataset(purpose="test", nImages=nTestImages)
testData = testData.batch(batchSize)

# Instantiate the model
with strategy.scope():
    autoencoder = DCVAE()
    optimizer = tf.keras.optimizers.Adam(1e-4)
    # If we are doing a restart, load the weights
    if args.epoch > 0:
        weights_dir = ("%s/Proxy_20CR/models/DCVAE_single_PRMSL_v3/" + "Epoch_%04d") % (
            os.getenv("SCRATCH"),
            args.epoch,
        )
        load_status = autoencoder.load_weights("%s/ckpt" % weights_dir)
        # Check the load worked
        load_status.assert_existing_objects_matched()


# Save the model weights and the history state after every epoch
history = {}
history["loss"] = []
history["val_loss"] = []


def save_state(model, epoch, loss):
    save_dir = ("%s/Proxy_20CR/models/DCVAE_single_PRMSL_v3/" + "Epoch_%04d") % (
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


for epoch in range(args.epoch, args.epoch + nEpochs):
    start_time = time.time()
    for train_x in trainingData:
        train_step(autoencoder, train_x, optimizer)
    end_time = time.time()

    train_rmse = tf.keras.metrics.Mean()
    train_logpz = tf.keras.metrics.Mean()
    train_logqz_x = tf.keras.metrics.Mean()
    for test_x in validationData:
        (rmse, logpz, logqz_x) = compute_loss(autoencoder, test_x)
        train_rmse(rmse)
        train_logpz(logpz)
        train_logqz_x(logqz_x)
    test_rmse = tf.keras.metrics.Mean()
    test_logpz = tf.keras.metrics.Mean()
    test_logqz_x = tf.keras.metrics.Mean()
    for test_x in testData:
        (rmse, logpz, logqz_x) = compute_loss(autoencoder, test_x)
        test_rmse(rmse)
        test_logpz(logpz)
        test_logqz_x(logqz_x)
    print("Epoch: {}".format(epoch))
    print("RMSE: {}, {}".format(train_rmse.result(), test_rmse.result()))
    print("logpz: {}, {}".format(train_logpz.result(), test_logpz.result()))
    print("logqz_x: {}, {}".format(train_logqz_x.result(), test_logqz_x.result()))
    print("time: {}".format(end_time - start_time))
    if epoch % 10 == 0:
        save_state(autoencoder, epoch, test_rmse.result())
