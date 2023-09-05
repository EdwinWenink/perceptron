"""
Main script for generating input data, training the perceptron
and visualizing the results.
"""

from src.generate_data import (generate_linearly_separable_dataset, get_and,
                               get_material_implication)
from src.perceptron import Perceptron
from src.visualise import plot_training_history, set_scale

if __name__ == '__main__':
    # Scale for data generation and plotting
    scale = 2
    set_scale(scale)

    # Generate input data sets for some simple logical functions.
    # We also generate linearly separable data using an intercept and slope.
    # We expect that the fitted decision boundary will have similar parameters.
    data_sets = [
            get_material_implication(),
            get_and(),
            generate_linearly_separable_dataset(
                intercept=0.5, slope=1.55, scale=scale)
            ]

    # Run the training and visualization for each of the defined inputs
    for X, y in data_sets:
        # Train the perceptron
        perceptron = Perceptron(n_inputs=X.shape[1], learning_rate=.1,
                                max_epochs=100)
        perceptron.train(X, y)

        # Plot training history as an overlay on the plot of perceptron inputs
        plot_training_history(X, y, perceptron.W_history)
