import numpy as np


def epsilon_greedy(model, env, state, eps=0.1):
    """ Epsilon-Greedy is a simple method to balance exploration and exploitation.

    @param model: a model that predicts the next most valuable action
    @param env: the environment
    @param state: the state from which the action is calculated
    @param eps: the probability of choosing to explore
    @return: returns next action to take
    """
    p = np.random.random()
    if p < (1 - eps):
        state = np.atleast_2d(state)
        values = model.predict(state, verbose=0)
        return np.argmax(values)
    else:
        return env.action_space.sample()
