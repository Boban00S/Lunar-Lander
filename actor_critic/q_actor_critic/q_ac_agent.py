import collections
import statistics
import tensorflow as tf
import tqdm
import numpy as np

from q_ac_model import ActorCriticNetwork


class ActorCriticAgent:
    """
        The actor-critic agent optimizes the policy(actor) directly and uses a critic
        to estimate the return or future rewards.
    """

    def __init__(self, env, gamma=0.99, learning_rate=0.01, num_hidden_units=128):
        """
        @param env: the environment
        @param gamma: discount factor
        @param learning_rate: model's learning rate
        @param num_hidden_units: number of neurons in hidden layer
        """

        self.env = env
        num_actions = env.action_space.n
        self.model = ActorCriticNetwork(num_actions, num_hidden_units)
        self.gamma = gamma
        self.optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)

    def env_step(self, action: np.ndarray):
        """Returns state, reward and done flag given an action."""

        state, reward, done, _ = self.env.step(action)
        return state.astype(np.float32), np.array(reward, np.float32), np.array(done, np.int32)

    def tf_env_step(self, action: tf.Tensor):
        """This would allow env_step to be included in a callable TensorFlow graph"""

        return tf.numpy_function(self.env_step, [action], [tf.float32, tf.float32, tf.int32])

    def train_step(self, state, initial_state_shape, action):
        """Runs a model training step.

        @param state: state at time t
        @param initial_state_shape: environment's state shape
        @param action: action to take at time t
        @return: state at time t+1, action to take at time t+1, done flag and reward of step(s, a) at time t
        """
        with tf.GradientTape() as tape:
            action_distributions, q_values = self.model(state)
            if action is None:
                action = action_distributions.sample()[0]

            action_log_prob = action_distributions.log_prob(action)
            q_value = q_values[0, action]

            next_state, reward, done = self.tf_env_step(action)
            next_state.set_shape(initial_state_shape)
            next_state = tf.expand_dims(next_state, 0)

            next_action_distributions, next_q_values = self.model(next_state)
            next_action = next_action_distributions.sample()[0]
            next_q_value = next_q_values[0, next_action]

            next_q_value = tf.where(done == 0, next_q_value, tf.constant(0.0))

            loss = self.compute_loss(action_log_prob, q_value, reward, next_q_value)

        grads = tape.gradient(loss, self.model.trainable_variables)

        self.optimizer.apply_gradients(zip(grads, self.model.trainable_variables))

        return next_state, next_action, done, reward

    @tf.function
    def run_episode(self, initial_state, max_steps=1000):
        """Runs a single episode of environment and trains agent at each step.

        @param initial_state: environment's initial state
        @param max_steps: max possible step's per episode
        @return: returns episode reward
        """
        initial_state_shape = initial_state.shape
        state = initial_state
        action = None

        state = tf.expand_dims(state, 0)
        episode_reward = tf.constant(0.0)
        episode_reward_shape = episode_reward.shape

        for _ in tf.range(max_steps):
            state, action, done, reward = self.train_step(state, initial_state_shape, action)
            episode_reward += reward
            episode_reward.set_shape(episode_reward_shape)

            if tf.cast(done, tf.bool):
                break

        return episode_reward

    def compute_loss(self, action_log_prob, q_value, reward, next_q_value):
        """Computes the combined actor-critic loss.

        @param action_log_prob: log probability of action taken at time t
        @param q_value: the value of the action performed in state s
        @param reward: reward of taken action
        @param next_q_value: the value of the action performed in state s+1
        @return: returns sum of actor and critic loss
        """

        actor_loss = -action_log_prob * q_value

        critic_loss = reward + self.gamma * next_q_value - q_value

        return actor_loss + critic_loss

    def train_agent(self, max_episodes=10000, reward_threshold=200, min_episodes_criterion=100):
        """Runs agent training

        @param reward_threshold: threshold when the agent is trained
        @param max_episodes: maximum number of agent training episodes
        @param min_episodes_criterion: minimum number of episodes to meet the criteria
        """

        episodes_reward = collections.deque(maxlen=100)

        with tqdm.trange(max_episodes) as t:
            for i in t:
                initial_state = tf.constant(self.env.reset())
                episode_reward = int(self.run_episode(initial_state))

                episodes_reward.append(episode_reward)
                running_reward = statistics.mean(episodes_reward)

                t.set_description(f"Episode {i}")
                t.set_postfix(episode_reward=episode_reward, running_reward=running_reward)

                if running_reward > reward_threshold and i >= min_episodes_criterion:
                    break

        print(f'Solved at episode {i}: average reward: {running_reward:.2f}!')
