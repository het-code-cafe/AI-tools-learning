"""
MSEExample.py
"""

import matplotlib.pyplot as plt
from sklearn.linear_model import SGDRegressor
from sklearn.metrics import mean_squared_error
from sklearn.datasets import make_regression

from .config import COLOR_PALETTE, OUTPUT_PATH


class MSEExample:
    # Set our colors from the palette
    green = COLOR_PALETTE['base_colors']['medium_green']
    light_gray = COLOR_PALETTE['neutral_colors']['light_gray']

    def __init__(self, n_samples=100, iterations=50, learning_rate=0.01):
        self.n_samples = n_samples
        self.iterations = iterations
        self.learning_rate = learning_rate

        # Generate synthetic data for regression
        self.X, self.y = make_regression(n_samples=self.n_samples, n_features=1, noise=15, random_state=42)

        # Initialize the model
        self.model = SGDRegressor(max_iter=1, tol=None, learning_rate='constant', eta0=self.learning_rate,
                                  random_state=42)

        # Placeholder to store MSE values
        self.mse_values = []

    def train_and_calculate_mse(self):
        """Train the model and calculate MSE for each iteration."""
        for _ in range(self.iterations):
            self.model.partial_fit(self.X, self.y)  # Perform one iteration of training
            y_pred = self.model.predict(self.X)
            mse = mean_squared_error(self.y, y_pred)
            self.mse_values.append(mse)

    def plot_mse(self):
        """Plot the MSE over iterations."""
        plt.figure(figsize=(10, 6))

        # Plotting the MSE values
        plt.plot(range(1, self.iterations + 1), self.mse_values, color=self.green, marker='o')

        # Title and labels
        plt.title('MSE over Iterations', fontsize=16)
        plt.xlabel('Iteration', fontsize=14)
        plt.ylabel('Mean Squared Error', fontsize=14)

        # Light gray grid
        plt.grid(True, color=self.light_gray)

        # Save the plot to the specified output path
        plt.savefig(OUTPUT_PATH + "mse_over_iterations.png")

    def main(self):
        """Main method to run the training and plot the MSE."""
        self.train_and_calculate_mse()
        self.plot_mse()


if __name__ == "__main__":
    mse_plotter = MSEExample()
    mse_plotter.main()
