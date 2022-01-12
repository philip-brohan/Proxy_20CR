# Specify a Deep Convolutional Variational AutoEncoder
#  for the 20CRv3 PRMSL fields

# Based on the TF tutorial at:
#   https://www.tensorflow.org/tutorials/generative/cvae

import tensorflow as tf
import numpy as np
import sys


class DCVAE(tf.keras.Model):
    def __init__(self):
        super(DCVAE, self).__init__()
        self.latent_dim = 100
        self.encoder = tf.keras.Sequential(
            [
                tf.keras.layers.InputLayer(input_shape=(256, 512, 1)),
                tf.keras.layers.Dropout(0.6),
                tf.keras.layers.Conv2D(
                    filters=5,
                    kernel_size=3,
                    strides=(2, 2),
                    padding="same",
                    activation="elu",
                ),
                tf.keras.layers.Dropout(0.4),
                tf.keras.layers.Conv2D(
                    filters=10,
                    kernel_size=3,
                    strides=(2, 2),
                    padding="same",
                    activation="elu",
                ),
                tf.keras.layers.Dropout(0.2),
                tf.keras.layers.Conv2D(
                    filters=20,
                    kernel_size=3,
                    strides=(2, 2),
                    padding="same",
                    activation="elu",
                ),
                tf.keras.layers.Dropout(0.1),
                tf.keras.layers.Conv2D(
                    filters=40,
                    kernel_size=3,
                    strides=(2, 2),
                    padding="same",
                    activation="elu",
                ),
                tf.keras.layers.Flatten(),
                tf.keras.layers.Dropout(0.1),
                # No activation
                tf.keras.layers.Dense(self.latent_dim + self.latent_dim),
            ]
        )

        self.decoder = tf.keras.Sequential(
            [
                tf.keras.layers.InputLayer(input_shape=(self.latent_dim,)),
                tf.keras.layers.Dense(units=16 * 32 * 40, activation=tf.nn.elu),
                tf.keras.layers.Reshape(target_shape=(16, 32, 40)),
                tf.keras.layers.Conv2DTranspose(
                    filters=20,
                    kernel_size=3,
                    strides=2,
                    padding="same",
                    activation="elu",
                ),
                tf.keras.layers.Conv2DTranspose(
                    filters=10,
                    kernel_size=3,
                    strides=2,
                    padding="same",
                    activation="elu",
                ),
                tf.keras.layers.Conv2DTranspose(
                    filters=5,
                    kernel_size=3,
                    strides=2,
                    padding="same",
                    activation="elu",
                ),
                # No activation
                tf.keras.layers.Conv2DTranspose(
                    filters=1, kernel_size=3, strides=2, padding="same"
                ),
            ]
        )

    def encode(self, x):
        mean, logvar = tf.split(self.encoder(x), num_or_size_splits=2, axis=1)
        return mean, logvar

    def decode(self, z):
        decoded = self.decoder(z)
        return decoded

    def call(self, x):
        mean, logvar = self.encode(x)
        latent = self.reparameterize(mean, logvar)
        encoded = self.decode(latent)
        return encoded

    def reparameterize(self, mean, logvar):
        eps = tf.random.normal(shape=mean.shape)
        return eps * tf.exp(logvar * 0.5) + mean

    # Decode each member of the batch several times, to make a sample
    #  returns a 4d tensor (size, batch, y, x)
    def sample_decode(self, mean, logvar, size=100):
        encoded = []
        eps = tf.random.normal(shape=(size, self.latent_dim))
        mean = tf.unstack(mean, axis=0)
        logvar = tf.unstack(logvar, axis=0)
        for batchI in range(len(mean)):
            latent = eps * tf.exp(logvar[batchI] * 0.5) + mean[batchI]
            encoded.append(self.decode(latent))
        return tf.stack(encoded, axis=0)

    # Autoencode each member of the batch several times, to make a sample
    def sample_call(self, x, size=100):
        mean, logvar = self.encode(x)
        encoded = self.sample_decode(mean, logvar, size=size)
        return encoded


def log_normal_pdf(sample, mean, logvar, raxis=1):
    log2pi = tf.math.log(2.0 * np.pi)
    return tf.reduce_sum(
        -0.5 * ((sample - mean) ** 2.0 * tf.exp(-logvar) + logvar + log2pi), axis=raxis
    )


def compute_loss(model, x):
    mean, logvar = model.encode(x)
    latent = model.reparameterize(mean, logvar)
    encoded = model.decode(latent)
    rmse = tf.reduce_mean(tf.keras.metrics.mean_squared_error(encoded, x), axis=[1, 2])
    logpz = log_normal_pdf(latent, 0.0, 0.0)
    logqz_x = log_normal_pdf(latent, mean, logvar)
    return (rmse * 100000, logpz, logqz_x)


@tf.function  # Optimiser ~25% speedup on VDI (CPU-only)
def train_step(model, x, optimizer):
    """Executes one training step and returns the loss.

    This function computes the loss and gradients, and uses the latter to
    update the model's parameters.
    """
    with tf.GradientTape() as tape:
        (rmse, logpz, logqz_x) = compute_loss(model, x)
        metric = tf.reduce_mean(rmse - logpz + logqz_x)
    gradients = tape.gradient(metric, model.trainable_variables)
    optimizer.apply_gradients(zip(gradients, model.trainable_variables))
