'''
Module for generating data for the perceptron.
'''

import numpy as np


def get_material_implication():
    '''Get input data corresponding to the material implication A -> B.'''
    # Without bias term perceptron will not converge for orthogonal inputs
    X = np.array([[1, 1],  # True
                  [1, -1],  # False
                  [-1, 1],  # True
                  [-1, -1]])  # True

    # Encode True as +1 and False as -1 such that y - y_pred is always 0 when
    # predicted correctly but non-zero otherwise
    y = np.array([1, -1, 1, 1])
    return X, y


def get_and():
    '''Get input data corresponding to the logical AND operation.'''
    X = np.array([[1, 1],  # True
                  [1, -1],  # False
                  [-1, 1],  # False
                  [-1, -1]])  # False

    # Encode True as +1 and False as -1 such that y - y_pred is always 0 when
    # predicted correctly but non-zero otherwise
    y = np.array([1, -1, -1, -1])
    return X, y


def generate_linearly_separable_dataset(intercept: float, slope: float,
                                        scale: int, N: int = 100):
    """
    Use an intercept and slope to generate separable data points.
    """
    # Generate random points in 2D space
    X = np.random.uniform(-scale, scale, (N, 2))

    # Calculate the corresponding Y labels based on the separating line
    y = (X[:, 1] > (slope * X[:, 0] + intercept)).astype(int)

    # Encode 0 as -1
    y[y == 0] = -1

    return X, y
