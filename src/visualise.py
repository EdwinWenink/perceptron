'''
Functions to visualize perceptron inputs and decision boundary over time.
'''

import matplotlib.pyplot as plt
import numpy as np
from matplotlib import colors

SCALE = 1


def set_scale(scale: int):
    """For better plotting results, set the scale to the scale of the data."""
    global SCALE
    SCALE = scale


def abline(intercept, slope):
    """Plot a line from slope and intercept"""
    axes = plt.gca()
    axes.set_xbound(-SCALE*1.5, SCALE*1.5)
    x_vals = np.array(axes.get_xlim())
    y_vals = intercept + slope * x_vals
    plt.xlim(-SCALE, SCALE)
    plt.ylim(-SCALE, SCALE)
    plt.plot(x_vals, y_vals, '--')


def plot_2D_decision_boundary(W: np.ndarray):
    """
    Given a fitted weights matrix (bias + 2 inputs) plot the decision boundary.
    """
    intercept = -W[0] / W[2]
    if W[2] and W[1]:
        slope = -(W[0] / W[2]) / (W[0] / W[1])
    else:
        # TODO evaluate if this makes sense
        slope = 0
    abline(intercept, slope)


def plot_2D_inputs(X: np.ndarray, y: np.ndarray):
    """Plot the input data."""
    assert X.shape[1] == 2
    cmap = colors.ListedColormap(['blue', 'red'])
    plt.scatter(X[:, 0], X[:, 1], c=y, cmap=cmap)


def plot_training_history(X: np.ndarray, y: np.ndarray,
                          W_history: list[np.ndarray]):
    """Plot the training history of Perceptron weights."""
    plot_2D_inputs(X, y)
    for i in range(0, len(W_history), len(X)):
        plot_2D_decision_boundary(W_history[i])
    plt.show()
