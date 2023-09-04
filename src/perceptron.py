import numpy as np


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
