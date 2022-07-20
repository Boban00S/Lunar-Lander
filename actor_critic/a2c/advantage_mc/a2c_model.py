import tensorflow as tf
import tensorflow_probability as tfp

from tensorflow.python.keras import layers


class ActorCriticNetwork(tf.keras.Model):
    """Combined actor-critic network."""

    def __init__(self, num_actions, num_hidden_units):
        super().__init__()

        self.hidden_layer = layers.Dense(num_hidden_units, activation='relu')
        self.actor = layers.Dense(num_actions)
        self.critic = layers.Dense(1)

    def call(self, input_tensor, training=False):
        x = self.hidden_layer(input_tensor, training=training)
        return tfp.distributions.Categorical(logits=self.actor(x)), self.critic(x)
