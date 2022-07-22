import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

sns.set_theme()


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
