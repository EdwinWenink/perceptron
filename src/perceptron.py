'''
Module defining the Perceptron class.
'''

import numpy as np


class Perceptron():
    '''
    Simple single layer perceptron for binary classification.
    '''

    def __init__(self, n_inputs: int, learning_rate: float, max_epochs: int = 100):
        '''
        Initialize Perceptron with zero weights for each input node
        and for the bias node.'''
        self.W = np.zeros((n_inputs+1, ))
        self.W_history = [self.W]
        self.learning_rate = learning_rate
        self.max_epochs = max_epochs

    def forward(self, x: np.ndarray, y: int | None = None):
        '''
        Make a forward pass to make predictions and return a delta.
        The delta will be all zeros unless targets y are provided.

        NOTE that a dummy input of 1 is added for the bias term
        so this does not have to be included in the model input.
        '''
        x = np.hstack((1, x))
        y_pred = 1 if self.W.dot(x) > 0 else -1

        # If y is provided, compute delta
        # otherwise just generate a prediction
        if y:
            delta = self.learning_rate * (y - y_pred) * x
        else:
            delta = np.zeros(self.W.shape)
        return y_pred, delta

    def update(self, x: np.ndarray, y: int) -> bool:
        '''
        Update weights and return whether
        the weights were updated.
        '''
        _, delta = self.forward(x, y)
        self.W += delta

        # Keep track of update history
        self.W_history.append(self.W)
        return delta.any()

    def train(self, X: np.ndarray, y: np.ndarray):
        '''
        Train the perceptron and store the history
        of weights for later plotting.
        '''
        for i in range(self.max_epochs):
            # Loop over all data points
            misclassifications = 0
            for xi, yi in zip(X, y):
                if self.update(xi, yi):
                    misclassifications += 1
                    print("Update", self.W)

            # Check for convergence
            if not misclassifications:
                print("Converged.")
                break

            print(f"{misclassifications} misclassifications this epoch.")

        print("Epochs trained:", i+1)
        return self.W_history
