import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

sns.set_theme()


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


def plot_learning(x, y, x_name, y_name):
    """Plots agent's rewards during training.

    @param x: episodes of training
    @param y: reward at each episode
    @param x_name: name of x-axis
    @param y_name: name of y-axis
    """
    dims = (10, 6)
    d = {x_name: x, y_name: y}
    df = pd.DataFrame(data=d)
    fig, ax = plt.subplots(figsize=dims)
    sns.lineplot(ax=ax, data=df, x=x_name, y=y_name)
    plt.show()
