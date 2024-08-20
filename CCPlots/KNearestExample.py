"""
KNNVisualizationExample.py

Create an animated plot of a K-Nearest Neighbors (KNN) classification example. The plot demonstrates the concept of KNN
by visualizing how a new data point is classified based on its nearest neighbors. This example uses house prices
based on the size of the house and the number of rooms.
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from sklearn.neighbors import KNeighborsRegressor

from .config import COLOR_PALETTE, OUTPUT_PATH


class KNearestExample:
    # Set colors for the plot
    green = COLOR_PALETTE['base_colors']['medium_green']
    periwinkle = COLOR_PALETTE['accent_colors']['periwinkle_blue']

    def main(self):
        # Generate a simple dataset (House Size, Number of Rooms vs. Price)
        np.random.seed(42)
        house_sizes = np.random.rand(100) * 2000 + 500  # House sizes between 500 and 2500 sq ft
        num_rooms = np.random.rand(100) * 5 + 1  # Number of rooms between 1 and 6
        prices = house_sizes * 150 + num_rooms * 20000 + (
                    np.random.randn(100) * 10000)  # Price = 150 * size + 20000 * rooms + noise

        # Create the figure and 3D axis for the animation
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        ax.set_xlim(min(house_sizes) - 100, max(house_sizes) + 100)
        ax.set_ylim(min(num_rooms) - 1, max(num_rooms) + 1)
        ax.set_zlim(min(prices) - 10000, max(prices) + 10000)
        ax.set_title("K-Nearest Neighbors for Housing")
        ax.set_xlabel("House Size (sq ft)")
        ax.set_ylabel("Number of Rooms")
        ax.set_zlabel("Price ($)")

        # Scatter plot of the data points
        scatter = ax.scatter(house_sizes, num_rooms, prices, color=self.green)

        # Initialize the KNN model
        knn = KNeighborsRegressor(n_neighbors=5)  # Using 5 nearest neighbors

        # Function to initialize the animation
        def init():
            return scatter,

        # Function to update the animation at each frame
        def update(frame):
            if frame < 5:  # Ensure at least 5 points are available for fitting
                return scatter,

            # Use data up to the current frame
            X = np.column_stack((house_sizes[:frame], num_rooms[:frame]))
            y = prices[:frame]

            # Fit the KNN model
            knn.fit(X, y)

            # New data point (for visualization purposes, we'll make it dynamic)
            new_point = np.array([[house_sizes[frame], num_rooms[frame]]])

            # Predict the price for the new data point
            predicted_price = knn.predict(new_point)

            # Plot the new point and the line connecting it to its predicted price
            ax.scatter(new_point[0, 0], new_point[0, 1], predicted_price[0], color=self.periwinkle,
                       label="New Point")
            ax.plot([new_point[0, 0], new_point[0, 0]], [new_point[0, 1], new_point[0, 1]],
                    [ax.get_zlim()[0], predicted_price[0]], color=self.periwinkle, linestyle="--")

            return scatter,

        # Create the animation
        ani = FuncAnimation(fig, update, frames=len(house_sizes), init_func=init, blit=False, interval=200)

        # Save the animation as a GIF
        ani.save(OUTPUT_PATH + "knn_visualization_animation.gif", writer='pillow')
