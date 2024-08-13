"""
LinearRegressionExample.py

Create an animated plot of a linear regression example. The plot demonstrates the concept of linear regression
by visualizing how the regression line evolves as more data points are added. This example uses house prices
based on the size of the house.
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from sklearn.linear_model import LinearRegression

from .config import THEME, OUTPUT_PATH


class LinearRegressionExample:
    def main(self):
        # Generate a simple dataset for linear regression (House Size vs. Price)
        np.random.seed(42)
        house_sizes = np.random.rand(100) * 2000 + 500  # House sizes between 500 and 2500 sq ft
        prices = house_sizes * 200 + (np.random.randn(100) * 10000)  # Price = 200 * size + noise

        # Sort the data for better visualization
        sorted_indices = np.argsort(house_sizes)
        house_sizes = house_sizes[sorted_indices]
        prices = prices[sorted_indices]

        # Create the figure and axis for the animation
        fig, ax = plt.subplots()
        ax.set_xlim(min(house_sizes) - 100, max(house_sizes) + 100)
        ax.set_ylim(min(prices) - 10000, max(prices) + 10000)
        ax.set_title("Linear Regression Example: House Size vs. Price", fontsize=16)
        ax.set_xlabel("House Size (sq ft)", fontsize=14)
        ax.set_ylabel("Price ($)", fontsize=14)

        # Scatter plot of the data points
        scatter = ax.scatter(house_sizes, prices, color='#ACD1AF')

        # Initialize the regression line plot
        line, = ax.plot([], [], color='#459578', linewidth=2)

        # Function to initialize the animation
        def init():
            line.set_data([], [])
            return line,

        # Function to update the animation at each frame
        def update(frame):
            if frame < 2:  # Ensure at least 2 points are available for fitting
                return line,

            # Use data up to the current frame
            X = house_sizes[:frame].reshape(-1, 1)
            y = prices[:frame]

            # Perform linear regression
            regressor = LinearRegression()
            regressor.fit(X, y)

            # Predict values across the full range of house sizes
            X_full = np.linspace(min(house_sizes), max(house_sizes), 100).reshape(-1, 1)
            y_pred = regressor.predict(X_full)

            # Update the line data
            line.set_data(X_full.flatten(), y_pred)

            return line,

        # Create the animation
        ani = FuncAnimation(fig, update, frames=len(house_sizes), init_func=init, blit=True, interval=10)

        # Save the animation as a GIF (I have no idea why
        ani.save(OUTPUT_PATH + "linear_regression_animation.gif", writer='pillow')

        plt.show()


if __name__ == "__main__":
    LinearRegressionExample().main()
