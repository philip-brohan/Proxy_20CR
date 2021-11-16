# Specify a Deep Convolutional AutoEncoder
#  for the 20CR2c PRMSL fields

# Based on the TF tutorial at:
#   https://www.tensorflow.org/tutorials/generative/cvae

# This version is not variational

import tensorflow as tf

class DCAE(tf.keras.Model):
    def __init__(self):
        super(DCAE, self).__init__()
        self.latent_dim = 100
        self.encoder = tf.keras.Sequential(
            [
                tf.keras.layers.InputLayer(input_shape=(80, 160, 1)),
                #tf.keras.layers.Dropout(0.2),
                tf.keras.layers.Conv2D(
                    filters=5, kernel_size=3, strides=(2, 2), padding='same', activation="relu"
                ),
                #tf.keras.layers.SpatialDropout2D(0.2),
                tf.keras.layers.Conv2D(
                    filters=5, kernel_size=3, strides=(2, 2), padding='same', activation="relu"
                ),
                #tf.keras.layers.SpatialDropout2D(0.2),
                tf.keras.layers.Conv2D(
                    filters=10, kernel_size=3, strides=(2, 2), padding='same', activation="relu"
                ),
                #tf.keras.layers.SpatialDropout2D(0.2),
                tf.keras.layers.Conv2D(
                    filters=20, kernel_size=3, strides=(2, 2), padding='same', activation="relu"
                ),
                #tf.keras.layers.SpatialDropout2D(0.2),
                tf.keras.layers.Flatten(),
                # No activation
                tf.keras.layers.Dense(self.latent_dim),
                #tf.keras.layers.BatchNormalization(),
                #tf.keras.layers.GaussianNoise(2.0),
            ]
        )

        self.decoder = tf.keras.Sequential(
            [
                tf.keras.layers.InputLayer(input_shape=(self.latent_dim,)),
                #tf.keras.layers.InputLayer(input_shape=(5*10*20)),
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
