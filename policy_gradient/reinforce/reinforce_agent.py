import collections
import statistics
import tensorflow as tf
import tqdm
import numpy as np

from reinforce_model import ReinforceNetwork


class ReinforceAgent:
    """
        A PG agent is a policy-based reinforcement learning agent that uses the REINFORCE algorithm
        to searches for an optimal policy that maximizes the expected cumulative long-term reward.
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
        self.model = ReinforceNetwork(num_actions, num_hidden_units)
        self.gamma = gamma
        self.optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)
        self.eps = np.finfo(np.float32).eps.item()

    def env_step(self, action: np.ndarray):
        """Returns state, reward and done flag given an action."""

        state, reward, done, _ = self.env.step(action)
        return state.astype(np.float32), np.array(reward, np.float32), np.array(done, np.int32)

    def tf_env_step(self, action: tf.Tensor):
        """This would allow env_step to be included in a callable TensorFlow graph"""

        return tf.numpy_function(self.env_step, [action], [tf.float32, tf.float32, tf.int32])

    def run_episode(self, initial_state, max_steps=1000):
        """Runs a single episode to collect training data.

        @param initial_state: environment's initial state
        @param max_steps: max possible step's per episode
        @return: returns log probabilities and rewards in each step of the episode
        """
        action_log_probs = tf.TensorArray(dtype=tf.float32, size=0, dynamic_size=True)
        rewards = tf.TensorArray(dtype=tf.float32, size=0, dynamic_size=True)

        initial_state_shape = initial_state.shape
        state = initial_state

        for t in tf.range(max_steps):
            state = tf.expand_dims(state, 0)
            action_logits_t = self.model(state)
            action = action_logits_t.sample()[0]

            action_log_probs = action_log_probs.write(t, action_logits_t.log_prob(action))

            state, reward, done = self.tf_env_step(action)
            state.set_shape(initial_state_shape)
            rewards = rewards.write(t, reward)

            if tf.cast(done, tf.bool):
                break
        action_log_probs = action_log_probs.stack()
        rewards = rewards.stack()

        return action_log_probs, rewards

    def get_expected_return(self, rewards, standardize=True):
        """Computes the return for each time step

        @param rewards: rewards for each time step
        @param standardize: boolean value whether to do standardization
        @return: returns discounted cumulative rewards at each step
        """
        n = tf.shape(rewards)[0]
        returns = tf.TensorArray(dtype=tf.float32, size=n)

        rewards = tf.cast(rewards[::-1], dtype=tf.float32)
        discounted_sum = tf.constant(0.0)
        discounted_sum_shape = discounted_sum.shape
        for i in tf.range(n):
            reward = rewards[i]
            discounted_sum = reward + self.gamma * discounted_sum
            discounted_sum.set_shape(discounted_sum_shape)
            returns = returns.write(i, discounted_sum)
        returns = returns.stack()[::-1]

        if standardize:
            returns = ((returns - tf.math.reduce_mean(returns)) / (tf.math.reduce_std(returns) + self.eps))

        return returns

    @tf.function
    def train_step(self, initial_state, max_steps_per_episode=1000):
        """Runs a model training step.

        @param initial_state: environment's initial state
        @param max_steps_per_episode: max possible step's per episode
        @return: returns episode reward
        """
        with tf.GradientTape() as tape:
            action_log_probs, rewards = self.run_episode(initial_state, max_steps_per_episode)

            returns = self.get_expected_return(rewards)

            returns = [tf.expand_dims(x, 1) for x in [returns]]

            loss = -tf.math.reduce_sum(action_log_probs * returns)

        grads = tape.gradient(loss, self.model.trainable_variables)

        self.optimizer.apply_gradients(zip(grads, self.model.trainable_variables))

        episode_reward = tf.math.reduce_sum(rewards)

        return episode_reward

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
                episode_reward = int(self.train_step(initial_state))

                episodes_reward.append(episode_reward)
                running_reward = statistics.mean(episodes_reward)

                t.set_description(f'Episode {i}')
                t.set_postfix(episode_reward=episode_reward, running_reward=running_reward)

                if running_reward > reward_threshold and i >= min_episodes_criterion:
                    break

        self.model.save_weights("reinforce_model_weights/")
        print(f'Solved at episode {i}: average reward: {running_reward:.2f}')

    def test_agent(self, display_episode=False):
        """Tests the agent on one episode

        @param display_episode: whether to show the episode
        """
        state = self.env.reset()
        total_reward, done = 0, False
        while not done:
            if display_episode:
                self.env.render()
            action = self.model(state).sample()[0]
            state, reward, done = self.env_step(action)
            total_reward += reward
        if display_episode:
            self.env.close()
        print(f'Episode reward: {total_reward:.2f}!')

