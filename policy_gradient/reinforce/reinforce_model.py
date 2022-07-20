import tensorflow as tf
import tensorflow_probability as tfp

from tensorflow.keras import layers


class ReinforceNetwork(tf.keras.Model):
    """REINFORCE network"""

    def __init__(self, num_actions, num_hidden_units=128):
        super().__init__()

        self.hidden_layer = layers.Dense(num_hidden_units, activation='relu')
        self.pi = layers.Dense(num_actions)

    def call(self, input_tensor, training=False):
        x = self.hidden_layer(input_tensor, training=training)
        return tfp.distributions.Categorical(logits=self.pi(x))
