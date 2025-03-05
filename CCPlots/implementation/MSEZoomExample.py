"""
MSEZoomExample.py
"""

import matplotlib.pyplot as plt
from sklearn.linear_model import SGDRegressor
from sklearn.datasets import make_regression

from CCPlots.PlotExample import PlotExample
from CCPlots.config import COLOR_PALETTE, OUTPUT_PATH


class MSEZoomExample(PlotExample):

    # Set our colors from the palette
    dark_green = COLOR_PALETTE['base_colors']['dark_green']
    green = COLOR_PALETTE['base_colors']['medium_green']
    bright_yellow = COLOR_PALETTE['base_colors']['bright_yellow']
    mint_green = COLOR_PALETTE['accent_colors']['mint_green']
    light_gray = COLOR_PALETTE['neutral_colors']['light_gray']

    def __init__(self, n_samples=100, learning_rate=0.01):
        self.n_samples = n_samples
        self.learning_rate = learning_rate

        # Generate synthetic data for regression
        self.X, self.y = make_regression(n_samples=self.n_samples, n_features=1, noise=15, random_state=42)

        # Initialize the model
        self.model = SGDRegressor(max_iter=1, tol=None, learning_rate='constant', eta0=self.learning_rate,
                                  random_state=42)

    def main(self):
        """Main method to train and plot the MSE zoom."""
        self.train_one_iteration()
        self.plot_mse_zoom()

    def train_one_iteration(self):
        """Train the model for one iteration and get the predictions."""
        self.model.partial_fit(self.X, self.y)  # Perform one iteration of training
        self.y_pred = self.model.predict(self.X)

    def plot_mse_zoom(self):
        """Plot the data points, the fitted line, and the errors."""
        plt.figure(figsize=(10, 6))

        # Plot data points
        plt.scatter(self.X, self.y, color=self.bright_yellow, label='Data points', edgecolor=self.dark_green)

        # Plot the fitted line
        plt.plot(self.X, self.y_pred, color=self.green, label='Fitted line')

        # Plot the errors
        for i in range(len(self.X)):
            plt.plot([self.X[i], self.X[i]], [self.y[i], self.y_pred[i]], color=self.mint_green, linestyle='--')

        # Title and labels
        plt.title('Zoomed-In MSE for one iteration', fontsize=16)
        plt.xlabel('Feature value', fontsize=14)
        plt.ylabel('Target value', fontsize=14)

        # Light gray grid
        plt.grid(True, color=self.light_gray)

        # Add legend
        plt.legend()

        # Save the plot to the specified output path
        plt.savefig(OUTPUT_PATH + "mse_zoom_iteration.png")

if __name__ == "__main__":
    MSEZoomExample().main()
