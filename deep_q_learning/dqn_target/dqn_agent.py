import tqdm
import tensorflow as tf
import numpy as np

from dqn_target.dqn_model import DQNNetwork
from dqn_target.replay_buffer import ReplayBuffer
from dqn_target.utils import plot_learning

tf.compat.v1.disable_eager_execution()


class DQNAgent:
    """
        Deep Q-Learning agents use Experience Replay to learn about
        their environment and update the Main and Target networks.
    """
    def __init__(self, env, policy, learning_rate=5e-4, epsilon=1.0, epsilon_decay=0.999, epsilon_min=0.01, gamma=0.99,
                 batch_size=64, fc1_units=256, fc2_units=256):
        """
        @param env: the environment
        @param policy: epsilon-greedy policy
        @param learning_rate: learning rate
        @param epsilon: the probability of choosing to explore
        @param epsilon_decay: the scale at which epsilon decays
        @param epsilon_min: minimal value of epsilon
        @param gamma: discount factor
        @param batch_size: number of samples that will be passed through to the network at one time
        @param fc1_units: number of neurons in the first hidden layer
        @param fc2_units: number of neurons in the second hidden layer
        """

        self.env = env
        self.policy = policy
        self.main_model, self.target_model = self._get_models(fc1_units, fc2_units, learning_rate)
        self.buffer = self._get_buffer()
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.epsilon_min = epsilon_min
        self.gamma = gamma
        self.batch_size = batch_size

    def _get_models(self, fc1_units, fc2_units, learning_rate):
        """ Getter for main and target network

        @param fc1_units: number of neurons in the first hidden layer
        @param fc2_units: number of neurons in the second hidden layer
        @param learning_rate: learning rate
        @return: main and target network
        """

        num_actions = self.env.action_space.n
        main_model = DQNNetwork(num_actions, fc1_units, fc2_units)
        main_model.compile(
            loss=tf.keras.losses.Huber(reduction=tf.keras.losses.Reduction.SUM),
            optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate)
        )
        target_model = DQNNetwork(num_actions, fc1_units, fc2_units)

        return main_model, target_model

    def _get_buffer(self):
        """ Filling the buffer with initial data before training

        @return: returns minimum filled replay buffer
        """

        buffer = ReplayBuffer()
        while True:
            state = self.env.reset()
            done = False
            while not done:
                action = self.env.action_space.sample()
                next_state, reward, done, _ = self.env.step(action)
                buffer.put(state, action, reward, next_state, done)
                if buffer.has_batch():
                    return buffer
                state = next_state

    def _target_update(self):
        """Copying weights of the main network to the target network"""

        self.target_model.set_weights(self.main_model.get_weights())

    def _get_action(self, state):
        """Returns the action calculated using epsilon-greedy policy"""

        return self.policy(self.main_model, self.env, state, self.epsilon)

    def run_episode(self):
        """Runs one episode of environment

        @return: returns episode reward
        """

        state = self.env.reset()
        done, episode_reward, steps = False, 0, 0

        while not done:
            steps += 1
            action = self._get_action(state)
            next_state, reward, done, _ = self.env.step(action)
            self.buffer.put(state, action, reward, next_state, done)
            episode_reward += reward
            state = next_state
            if steps % 100 == 0:
                self._target_update()
            self.replay()
        return episode_reward

    def replay(self):
        """Runs a model training step"""

        states, actions, rewards, next_states, done = self.buffer.sample()
        targets = self.target_model.predict(states, verbose=0)
        next_q_values = self.target_model.predict(next_states, verbose=0).max(axis=1)
        batch_index = np.arange(self.batch_size, dtype=np.int32)
        targets[batch_index, actions] = rewards + (1 - done) * next_q_values * self.gamma
        self.main_model.fit(states, targets, verbose=0)
        self.epsilon = max(self.epsilon * self.epsilon_decay, self.epsilon_min)

    def train_agent(self, reward_threshold=200, max_episodes=1000, min_episodes_criterion=200):
        """Runs agent training

        @param reward_threshold: threshold when the agent is trained
        @param max_episodes: maximum number of agent training episodes
        @param min_episodes_criterion: minimum number of episodes to meet the criteria
        """

        episodes_reward = []

        with tqdm.trange(max_episodes) as t:
            for i in t:
                episode_reward = int(self.run_episode())

                episodes_reward.append(episode_reward)
                running_reward = np.mean(episodes_reward[-100:])

                t.set_description(f'Episode {i}')
                t.set_postfix(episode_reward=episode_reward, running_reward=running_reward)

                if running_reward > reward_threshold and i >= min_episodes_criterion:
                    break

        self.main_model.save_weights("dql_model_weights/")
        print(f'Solved at episode {i}: average reward: {running_reward:.2f}!')
        plot_learning(np.arange(0, i+1), np.array(episodes_reward), "Episodes", "Rewards")

    def test_agent(self, display_episode=False):
        """Tests the agent on one episode

        @param display_episode: whether to show the episode
        """

        state = self.env.reset()
        total_reward, done = 0, False
        while not done:
            if display_episode:
                self.env.render()
            action = self._get_action(state)
            state, reward, done, _ = self.env.step(action)
            total_reward += reward
        if display_episode:
            self.env.close()
        print(f'Episode reward: {total_reward:.2f}!')

    def load_weights(self, filename):
        """Loads weights of the network that is located in filename."""

        self.main_model.load_weights(filename)
        self.target_model.load_weights(filename)


