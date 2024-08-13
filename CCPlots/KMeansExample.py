"""
KMeansExample.py
"""

from sklearn.datasets import make_blobs
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from matplotlib.colors import to_rgba

from .config import THEME, OUTPUT_PATH


class KMeansExample:

    def __init__(self, n_clusters=4, n_samples=300):
        self.n_clusters = n_clusters
        self.n_samples = n_samples
        self.X, self.y = make_blobs(n_samples=self.n_samples, centers=self.n_clusters, cluster_std=3.0, random_state=42)
        self.kmeans = KMeans(n_clusters=self.n_clusters, init='random', n_init=1, max_iter=1, algorithm='lloyd', random_state=42)

        # Generate colors from the specified colormap
        cmap = plt.get_cmap(THEME)
        self.cluster_colors = [cmap(i / self.n_clusters) for i in range(self.n_clusters)]
        self.center_colors = [self.darken_color(cmap(i / self.n_clusters)) for i in range(self.n_clusters)]

    def darken_color(self, color, factor=0.35):
        """Darken a given RGBA color."""
        r, g, b, a = to_rgba(color)
        return (r * factor, g * factor, b * factor, a)

    def update(self, frame):
        # Fit KMeans incrementally for each frame
        self.kmeans.max_iter = frame + 1
        self.kmeans.fit(self.X)

        # Assign colors to clusters
        labels = self.kmeans.labels_
        scatter_colors = [self.cluster_colors[label] for label in labels]

        self.scatter.set_color(scatter_colors)
        self.centers.set_offsets(self.kmeans.cluster_centers_)
        self.centers.set_color([self.center_colors[label] for label in range(self.n_clusters)])
        return self.scatter, self.centers

    def init_func(self):
        self.scatter.set_offsets(self.X)
        return self.scatter, self.centers

    def main(self):
        fig, ax = plt.subplots()
        ax.set_xlim(-20, 20)
        ax.set_ylim(-20, 20)

        # Initial points are grey
        self.scatter = ax.scatter(self.X[:, 0], self.X[:, 1], s=30, c='grey', edgecolor='k')
        self.centers = ax.scatter([], [], s=100, marker='X')  # Centers will use darker versions of cluster colors

        ax.set_title("KMeans Clustering: Determining Species", fontsize=16)
        ax.set_xlabel("Height of a penguin", fontsize=14)
        ax.set_ylabel("Weight of a penguin", fontsize=14)

        # More frames and a smaller interval to show the process better
        ani = FuncAnimation(fig, self.update, frames=30, init_func=self.init_func, interval=10, repeat=False)
        ani.save(OUTPUT_PATH + "kmeans_animation.gif", writer='pillow')


if __name__ == "__main__":
    KMeansExample().main()
