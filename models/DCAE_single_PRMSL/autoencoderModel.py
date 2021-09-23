# Specify a Deep Convolutional AutoEncoder
#  for the 20CR2c PRMSL fields

# Based on the TF tutorial at:
#   https://www.tensorflow.org/tutorials/generative/cvae

# This version is not variational

import tensorflow as tf

# import numpy as np
# import sys


class DCAE(tf.keras.Model):
    def __init__(self, latent_dim):
        super(DCAE, self).__init__()
        self.latent_dim = latent_dim
        self.encoder = tf.keras.Sequential(
            [
                tf.keras.layers.InputLayer(input_shape=(80, 160, 1)),
                tf.keras.layers.Conv2D(
                    filters=32, kernel_size=3, strides=(2, 2), activation="relu"
                ),
                tf.keras.layers.Conv2D(
                    filters=64, kernel_size=3, strides=(2, 2), activation="relu"
                ),
                tf.keras.layers.Flatten(),
                # No activation
                tf.keras.layers.Dense(latent_dim),
            ]
        )

        self.decoder = tf.keras.Sequential(
            [
                tf.keras.layers.InputLayer(input_shape=(latent_dim,)),
                tf.keras.layers.Dense(units=20 * 40 * 32, activation=tf.nn.relu),
                tf.keras.layers.Reshape(target_shape=(20, 40, 32)),
                tf.keras.layers.Conv2DTranspose(
                    filters=64,
                    kernel_size=3,
                    strides=2,
                    padding="same",
                    activation="relu",
                ),
                tf.keras.layers.Conv2DTranspose(
                    filters=32,
                    kernel_size=3,
                    strides=2,
                    padding="same",
                    activation="relu",
                ),
                # No activation
                tf.keras.layers.Conv2DTranspose(
                    filters=1, kernel_size=3, strides=1, padding="same"
                ),
            ]
        )

    def encode(self, x):
        encoded = self.encoder(x)
        return encoded

    def decode(self, z):
        decoded = self.decoder(z)
        return decoded

    def call(self, x):
        return self.decode(self.encode(x))


def compute_loss(model, x):
    latent = model.encode(x)
    encoded = model.decode(latent)
    rmse = tf.keras.metrics.mean_squared_error(encoded, x)
    return rmse


@tf.function  # Optimiser ~25% speedup on VDI (CPU-only)
def train_step(model, x, optimizer):
    """Executes one training step and returns the loss.

    This function computes the loss and gradients, and uses the latter to
    update the model's parameters.
    """
    with tf.GradientTape() as tape:
        loss = compute_loss(model, x)
    gradients = tape.gradient(loss, model.trainable_variables)
    optimizer.apply_gradients(zip(gradients, model.trainable_variables))
