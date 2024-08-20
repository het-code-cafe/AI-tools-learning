"""
sklearn_clustering.py

Chapter: Algorithms
Topic: Unsupervised Learning, clustering
"""
from sklearn.cluster import KMeans

# Example data
X = [[1, 2], [2, 3], [3, 4], [8, 9], [9, 10]]

# Model
kmeans = KMeans(n_clusters=2)
kmeans.fit(X)

# Prediction
labels = kmeans.predict([[0, 0], [10, 10]])
print(labels)  # Output: [0, 1]
