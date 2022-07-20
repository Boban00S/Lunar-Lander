import collections
import numpy as np
import statistics
import tensorflow as tf
import tqdm
import os

from a2c_model import ActorCriticNetwork

os.environ["TP_CPP_MIN_LOG_LEVEL"] = "3"


class ActorCriticAgent:
    """
        The actor-critic agent optimizes the policy(actor) directly and uses a critic
        to estimate the return or future rewards.
    """

    def __init__(self, env, gamma=0.99, learning_rate=0.001, num_hidden_units=128):
        """
        @param env: the environment
        @param gamma: discount factor
        @param learning_rate: model's learning rate
        @param num_hidden_units: number of neurons in hidden layer
        """

        self.env = env
        num_actions = self.env.action_space.n
        self.model = ActorCriticNetwork(num_actions, num_hidden_units)
        self.gamma = gamma
        self.eps = np.finfo(np.float32).eps.item()
        self.huber_loss = tf.keras.losses.Huber(reduction=tf.keras.losses.Reduction.SUM)
        self.optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)

    def tf_env_step(self, action: tf.Tensor):
        """This would allow env_step to be included in a callable TensorFlow graph."""

        return tf.numpy_function(self.env_step, [action], [tf.float32, tf.float32, tf.int32])

    def env_step(self, action: np.ndarray):
        """Returns state, reward and done flag given an action."""

        state, reward, done, _ = self.env.step(action)
        return state.astype(np.float32), np.array(reward, np.float32), np.array(done, np.int32)

    @tf.function
    def run_episode(self, initial_state, max_steps=1000):
        """Runs a single episode of environment and trains agent at each step(TD).

        @param initial_state: environment's initial state
        @param max_steps: max possible step's per episode
        @return: returns episode reward
        """

        initial_state_shape = initial_state.shape
        state = initial_state
        state = tf.expand_dims(state, 0)
        episode_reward = tf.constant(0.0)
        episode_reward_shape = episode_reward.shape

        for _ in tf.range(max_steps):
            with tf.GradientTape() as tape:
                action_logits, value = self.model(state)
                action = action_logits.sample()[0]

                action_log_prob = action_logits.log_prob(action)

                state, reward, done = self.tf_env_step(action)
                state.set_shape(initial_state_shape)
                state = tf.expand_dims(state, 0)
                episode_reward = tf.add(episode_reward, reward)
                episode_reward.set_shape(episode_reward_shape)

                _, next_value = self.model(state)

                next_value = tf.where(done == 0, next_value, 0)

                loss = self.compute_loss(value, action_log_prob, reward, next_value)

            grads = tape.gradient(loss, self.model.trainable_variables)

            self.optimizer.apply_gradients(zip(grads, self.model.trainable_variables))

            if tf.cast(done, tf.bool):
                break

        return episode_reward

    def compute_loss(self, value, action_log_prob, reward, next_value):
        """ Computes the combined actor-critic loss.

        @param value: state value at time t
        @param action_log_prob: log probability of action taken at time t
        @param reward: reward of taken action
        @param next_value: state value at time t+1
        @return: returns sum of actor and critic loss
        """

        td_target = reward + self.gamma * next_value
        advantage = td_target - value

        critic_loss = self.huber_loss(td_target, value)

        actor_loss = - action_log_prob * advantage

        return critic_loss + actor_loss

    def train_agent(self, reward_threshold=200, max_episodes=10000, min_episodes_criterion=200):
        """Runs agent training

        @param reward_threshold: threshold when the agent is trained
        @param max_episodes: maximum number of agent training episodes
        @param min_episodes_criterion: minimum number of episodes to meet the criteria
        """

        episodes_reward = collections.deque(maxlen=100)

        with tqdm.trange(max_episodes) as t:
            for i in t:
                initial_state = tf.constant(self.env.reset(), dtype=tf.float32)
                episode_reward = int(self.run_episode(initial_state))

                episodes_reward.append(episode_reward)
                running_reward = statistics.mean(episodes_reward)

                t.set_description(f'Episode {i}')
                t.set_postfix(episode_reward=episode_reward, running_reward=running_reward)

                if running_reward > reward_threshold and i >= min_episodes_criterion:
                    break

            self.model.save_weights("a2c_td_model_weights/")
            print(f'Solved at episode {i}: average reward: {running_reward:.2f}!')

import gym
env = gym.make("LunarLander-v2")
agent = ActorCriticAgent(env)
agent.train_agent()
