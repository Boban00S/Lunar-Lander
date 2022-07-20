import tensorflow as tf
from tensorflow.keras import layers


class DQNNetwork(tf.keras.Model):
    """Deep Q Learning three-layer network."""

    def __init__(self, num_actions, fc1_units=256, fc2_units=256):
        super().__init__()

        self.fc1_layer = layers.Dense(fc1_units, activation='relu')
        self.fc2_layer = layers.Dense(fc2_units, activation='relu')
        self.output_layer = layers.Dense(num_actions)

    def call(self, input_tensor, training=False):
        x = self.fc1_layer(input_tensor, training=training)
        x = self.fc2_layer(x, training=training)
        return self.output_layer(x)
