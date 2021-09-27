# Specify a Deep Convolutional Variational AutoEncoder
#  for the 20CR2c PRMSL fields

# Based on the TF tutorial at:
#   https://www.tensorflow.org/tutorials/generative/cvae

import tensorflow as tf
import numpy as np


class DCVAE(tf.keras.Model):
    def __init__(self, latent_dim):
        super(DCVAE, self).__init__()
        self.latent_dim = latent_dim
        self.encoder = tf.keras.Sequential(
            [
                tf.keras.layers.InputLayer(input_shape=(80, 160, 1)),
                # tf.keras.layers.Dropout(0.2),
                tf.keras.layers.Conv2D(
                    filters=5,
                    kernel_size=3,
                    strides=(2, 2),
                    padding="same",
                    activation="relu",
                ),
                # tf.keras.layers.SpatialDropout2D(0.2),
                tf.keras.layers.Conv2D(
                    filters=5,
                    kernel_size=3,
                    strides=(2, 2),
                    padding="same",
                    activation="relu",
                ),
                # tf.keras.layers.SpatialDropout2D(0.2),
                tf.keras.layers.Conv2D(
                    filters=10,
                    kernel_size=3,
                    strides=(2, 2),
                    padding="same",
                    activation="relu",
                ),
                # tf.keras.layers.SpatialDropout2D(0.2),
                tf.keras.layers.Conv2D(
                    filters=20,
                    kernel_size=3,
                    strides=(2, 2),
                    padding="same",
                    activation="relu",
                ),
                # tf.keras.layers.SpatialDropout2D(0.2),
                tf.keras.layers.Flatten(),
                # No activation
                tf.keras.layers.Dense(latent_dim + latent_dim),
                # tf.keras.layers.BatchNormalization(),
                # tf.keras.layers.GaussianNoise(2.0),
            ]
        )

        self.decoder = tf.keras.Sequential(
            [
                tf.keras.layers.InputLayer(input_shape=(latent_dim,)),
                # tf.keras.layers.InputLayer(input_shape=(5*10*20)),
                tf.keras.layers.Dense(units=5 * 10 * 20, activation=tf.nn.relu),
                tf.keras.layers.Reshape(target_shape=(5, 10, 20)),
                tf.keras.layers.Conv2DTranspose(
                    filters=10,
                    kernel_size=3,
                    strides=2,
                    padding="same",
                    activation="relu",
                ),
                tf.keras.layers.Conv2DTranspose(
                    filters=5,
                    kernel_size=3,
                    strides=2,
                    padding="same",
                    activation="relu",
                ),
                tf.keras.layers.Conv2DTranspose(
                    filters=5,
                    kernel_size=3,
                    strides=2,
                    padding="same",
                    activation="relu",
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
        return self.decode(self.encode(x))

    def reparameterize(self, mean, logvar):
        eps = tf.random.normal(shape=mean.shape)
        return eps * tf.exp(logvar * 0.5) + mean


def log_normal_pdf(sample, mean, logvar, raxis=1):
    log2pi = tf.math.log(2.0 * np.pi)
    return tf.reduce_sum(
        -0.5 * ((sample - mean) ** 2.0 * tf.exp(-logvar) + logvar + log2pi), axis=raxis
    )


def compute_loss(model, x):
    mean, logvar = model.encode(x)
    latent = model.reparameterize(mean, logvar)
    encoded = model.decode(latent)
    rmse = tf.keras.metrics.mean_squared_error(encoded, x)
    logpz = log_normal_pdf(latent, 0.0, 0.0)
    logqz_x = log_normal_pdf(latent, mean, logvar)
    return (rmse, logpz, logqz_x)


@tf.function  # Optimiser ~25% speedup on VDI (CPU-only)
def train_step(model, x, optimizer):
    """Executes one training step and returns the loss.

    This function computes the loss and gradients, and uses the latter to
    update the model's parameters.
    """
    with tf.GradientTape() as tape:
        (rmse, logpz, logqz_x) = compute_loss(model, x)
    gradients = tape.gradient(rmse, model.trainable_variables)
    optimizer.apply_gradients(zip(gradients, model.trainable_variables))
