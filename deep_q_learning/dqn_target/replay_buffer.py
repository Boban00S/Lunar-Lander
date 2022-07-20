import random
import collections
import numpy as np


class ReplayBuffer:
    """
    Reinforcement learning algorithms use replay buffers to store
    trajectories of experience when executing a policy in an environment.
    """

    def __init__(self, batch_size=64, capacity=1000000):
        """
        @param batch_size: number of samples that will be passed through to the network at one time
        @param capacity: buffer capacity
        """
        self.batch_size = batch_size
        self.buffer = collections.deque(maxlen=capacity)

    def put(self, state, action, reward, next_state, done):
        """Adding one element to the buffer"""

        self.buffer.append([state, action, reward, next_state, done])

    def sample(self):
        """Returns batch_size of random elements from buffer."""

        sample = random.sample(self.buffer, self.batch_size)
        states, actions, rewards, next_states, done = map(np.asarray, zip(*sample))
        states = np.array(states).reshape(self.batch_size, -1)
        next_states = np.array(next_states).reshape(self.batch_size, -1)
        return states, actions, rewards, next_states, done

    def size(self):
        return len(self.buffer)

    def capacity(self):
        return self.buffer.maxlens

    def has_batch(self):
        return self.size() == self.batch_size
