from src.generate_data import (generate_linearly_separable_dataset, get_AND,
                               get_material_implication)
from src.perceptron import Perceptron
from src.visualise import plot_training_history, set_scale

if __name__ == '__main__':
    # Scale for data generation and plotting
    scale = 2
    set_scale(scale)

    # Training example: material implication A -> B
    # X, y = get_material_implication()
    # X, y = get_AND()

    # Or instead generate linearly separable data using intercept and slope
    # We expect that the fitted decision boundary will have similar intercept and slope
    X, y = generate_linearly_separable_dataset(intercept=0.5,
                                               slope=1.55,
                                               scale=scale)

    # Learning rate
    learning_rate = .1

    # Train the perceptron
    perceptron = Perceptron(n_inputs=X.shape[1], learning_rate=learning_rate,
                            max_epochs=100)
    perceptron.train(X, y)

    # Plot training history overlayed on plot of perceptron inputs
    plot_training_history(X, y, perceptron.W_history)
