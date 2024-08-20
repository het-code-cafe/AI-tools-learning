"""
sklearn_pca.py
"""

from sklearn.decomposition import PCA

# Example data
X = [[1, 2], [3, 4], [5, 6], [7, 8]]

# Model
pca = PCA(n_components=1)
X_reduced = pca.fit_transform(X)

print(X_reduced)
