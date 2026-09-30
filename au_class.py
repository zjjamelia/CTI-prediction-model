import numpy as np
import tensorflow as tf


def xavier_init(fan_in, fan_out, constant=1):
    low = -constant * np.sqrt(6.0 / (fan_in + fan_out))
    high = constant * np.sqrt(6.0 / (fan_in + fan_out))
    return tf.random.uniform((fan_in, fan_out), minval=low, maxval=high, dtype=tf.float32)


class Autoencoder(tf.keras.Model):
    def __init__(self, n_input, n_hidden, transfer_function=tf.nn.softplus, optimizer=None, scale=0.2):
        super(Autoencoder, self).__init__()
        self.n_input = n_input
        self.n_hidden = n_hidden
        self.transfer = transfer_function
        self.scale = scale
        self.training_scale = scale

        # Define the layers using Keras
        self.hidden_layer = tf.keras.layers.Dense(n_hidden, activation=transfer_function,
                                                  kernel_initializer=tf.keras.initializers.GlorotUniform())
        self.output_layer = tf.keras.layers.Dense(n_input, kernel_initializer=tf.keras.initializers.GlorotUniform())

        # Compile the model
        self.compile(optimizer=optimizer or tf.keras.optimizers.Adam(), loss=self.custom_loss)

    def custom_loss(self, y_true, y_pred):
        return 0.5 * tf.reduce_sum(tf.pow(tf.subtract(y_pred, y_true), 2.0))

    def call(self, inputs):
        noisy_input = inputs + self.scale * tf.random.normal(tf.shape(inputs))
        hidden = self.hidden_layer(noisy_input)
        reconstruction = self.output_layer(hidden)
        return reconstruction

    def partial_fit(self, X):
        cost = self.train_on_batch(X, X)
        return cost

    def before_loss(self, X):
        cost = self.evaluate(X, X)
        return cost

    def transform(self, X):
        hidden = self.hidden_layer(X)
        return hidden

    def generate(self, hidden=None):
        if hidden is None:
            hidden = np.random.normal(size=self.hidden_layer.input_shape[1])
        return self.output_layer(hidden)

    def reconstruct(self, X):
        return self.call(X)

    def getWeights(self):
        return self.hidden_layer.kernel.numpy()

    def getBias(self):
        return self.hidden_layer.bias.numpy()
