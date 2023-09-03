import matplotlib.pyplot as plt
import numpy as np
from matplotlib import colors

SCALE = 1


class Perceptron():
    '''
    Simple single layer perceptron for binary classification.
    '''

    def __init__(self, n_inputs: int, learning_rate: float, max_epochs: int = 100):
        # Initialize zero weights for each input node + for the bias node
        self.W = np.zeros((n_inputs+1, ))
        self.W_history = [self.W]
        self.learning_rate = learning_rate
        self.max_epochs = max_epochs

    def forward(self, x: np.ndarray, y: int | None = None):
        # NOTE this adds a dummy input for the bias term
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
        y_pred, delta = self.forward(x, y)
        self.W += delta

        # Keep track of update history
        self.W_history.append(self.W)
        return delta.any()

    def train(self, X: np.ndarray, y: np.ndarray):
        # Store weights for later plotting
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


def abline(intercept, slope):
    """Plot a line from slope and intercept"""
    axes = plt.gca()
    axes.set_xbound(-SCALE, SCALE)
    x_vals = np.array(axes.get_xlim())
    y_vals = intercept + slope * x_vals
    plt.ylim(-SCALE, SCALE)
    plt.plot(x_vals, y_vals, '--')


def plot_decision_boundary(W: np.ndarray):
    """
    Given a fitted weights matrix (bias + 2 inputs)
    plot the decision boundary.
    """
    intercept = -W[0] / W[2]
    if W[2] and W[1]:
        slope = -(W[0] / W[2]) / (W[0] / W[1])
    else:
        # TODO evaluate
        slope = 0
    abline(intercept, slope)


def plot_2D_inputs(X: np.ndarray, y: np.ndarray):
    assert X.shape[1] == 2
    cmap = colors.ListedColormap(['blue', 'red'])
    plt.scatter(X[:, 0], X[:, 1], c=y, cmap=cmap)


def plot_training_history(X: np.ndarray, W_history: list[np.ndarray]):
    plot_2D_inputs(X, y)
    for i in range(0, len(W_history), len(X)):
        plot_decision_boundary(W_history[i])
    plt.show()


def get_material_implication():
    X = np.array([[1, 1],  # True
                  [1, -1],  # False
                  [-1, 1],  # True
                  [-1, -1]])  # True

    # Encode True as +1 and False as -1 such that y - y_pred is always 0 when
    # predicted correctly but non-zero otherwise
    y = np.array([1, -1, 1, 1])
    return X, y


def get_AND():
    X = np.array([[1, 1],  # True
                  [1, -1],  # False
                  [-1, 1],  # False
                  [-1, -1]])  # False

    # Encode True as +1 and False as -1 such that y - y_pred is always 0 when
    # predicted correctly but non-zero otherwise
    y = np.array([1, -1, -1, -1])
    return X, y


def generate_linearly_separable_dataset(intercept: float, slope: float, N: int = 100):
    """
    Use an intercept and slope to generate separable data points.
    """
    # Generate random points in 2D space
    # X = np.random.rand(N, 2) * SCALE
    X = np.random.uniform(-SCALE, SCALE, (N, 2))

    # Calculate the corresponding Y labels based on the separating line
    y = (X[:, 1] > (slope * X[:, 0] + intercept)).astype(int)

    # Encode 0 as -1
    y[y == 0] = -1

    return X, y


if __name__ == '__main__':
    # Training example: material implication A -> B
    # Without bias this will not converge for orthogonal inputs
    # X, y = get_material_implication()
    # X, y = get_AND()
    X, y = generate_linearly_separable_dataset(0.5, 1.55)

    # Learning rate
    learning_rate = .1

    # Train the perceptron
    perceptron = Perceptron(n_inputs=X.shape[1], learning_rate=learning_rate, max_epochs=100)
    perceptron.train(X, y)

    plot_training_history(X, perceptron.W_history)
