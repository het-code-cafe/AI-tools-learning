"""
LogisticRegressionExample.py
"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_classification
from sklearn.linear_model import LogisticRegression
from matplotlib.animation import FuncAnimation
from matplotlib.colors import ListedColormap

from CCPlots.PlotExample import PlotExample
from CCPlots.config import THEME, COLOR_PALETTE, OUTPUT_PATH


class LogisticRegressionExample(PlotExample):
    # Use the specified colormap theme for the plot
    cmap_light = plt.get_cmap(THEME)

    # Define a high-contrast color palette for the classes
    cmap_bold = [
        COLOR_PALETTE["complementary_colors"]["deep_burgundy"],  # Higher contrast color
        COLOR_PALETTE["analogous_colors"]["soft_green"]  # Contrasting but distinct
    ]

    # Use the dark green from our palette instead of black
    dark_green = COLOR_PALETTE['base_colors']['dark_green']

    def __init__(self, n_samples=200):
        self.n_samples = n_samples

        # Create a synthetic dataset
        self.X, self.y = make_classification(
            n_samples=self.n_samples,
            n_features=2,
            n_informative=2,
            n_redundant=0,
            n_clusters_per_class=1,
            n_classes=2,
            random_state=42
        )

        # Initialize logistic regression
        self.model = LogisticRegression(solver='lbfgs')

    def update(self, frame):
        # Train the model incrementally
        self.model.max_iter = frame + 1
        self.model.fit(self.X, self.y)

        # Predict probability for the mesh grid points
        Z = self.model.predict_proba(np.c_[self.xx.ravel(), self.yy.ravel()])[:, 1]
        Z = Z.reshape(self.xx.shape)

        # Remove the old contour and create a new one
        for c in self.contourf.collections:
            c.remove()
        for c in self.contour.collections:
            c.remove()

        self.contourf = self.ax.contourf(self.xx, self.yy, Z, alpha=0.3, cmap=self.cmap_light)
        self.contour = self.ax.contour(self.xx, self.yy, Z, levels=[0.5], linewidths=2, colors=self.dark_green)

        return self.contourf.collections + self.contour.collections + [self.scatter]

    def init_func(self):
        return self.contourf.collections + self.contour.collections + [self.scatter]

    def main(self):
        # Set up the plot
        fig, self.ax = plt.subplots()
        self.ax.set_xlim(self.X[:, 0].min() - 1, self.X[:, 0].max() + 1)
        self.ax.set_ylim(self.X[:, 1].min() - 1, self.X[:, 1].max() + 1)
        self.ax.set_title("Logistic Regression Example: Spam vs. not spam", fontsize=16)
        self.ax.set_xlabel("Number of links in Email", fontsize=14)
        self.ax.set_ylabel("Email length (in characters)", fontsize=14)

        # Mesh grid for the background
        self.xx, self.yy = np.meshgrid(np.arange(self.X[:, 0].min() - 1, self.X[:, 0].max() + 1, 0.1),
                                       np.arange(self.X[:, 1].min() - 1, self.X[:, 1].max() + 1, 0.1))

        # Initial contour plot
        Z = np.zeros_like(self.xx)
        self.contourf = self.ax.contourf(self.xx, self.yy, Z, alpha=0.8, cmap=self.cmap_light)
        self.contour = self.ax.contour(self.xx, self.yy, Z, levels=[0.5], linewidths=2, colors=self.dark_green)

        # Scatter plot of the data points, now with higher contrast colors
        self.scatter = self.ax.scatter(self.X[:, 0], self.X[:, 1], c=self.y, cmap=ListedColormap(self.cmap_bold),
                                       edgecolor=self.dark_green, s=40)

        # Create the animation
        ani = FuncAnimation(fig, self.update, frames=30, init_func=self.init_func, interval=200, repeat=False)

        # Save the animation as a GIF
        ani.save(OUTPUT_PATH + "logistic_regression_animation.gif", writer='pillow')


if __name__ == "__main__":
    LogisticRegressionExample().main()
