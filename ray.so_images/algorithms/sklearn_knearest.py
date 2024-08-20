"""
sklearn_knearest.py
"""
from sklearn.neighbors import KNeighborsClassifier

# Example data
X = [[0], [1], [2], [3]]
y = [0, 0, 1, 1]

# Model
knn = KNeighborsClassifier(n_neighbors=3)
knn.fit(X, y)

# Prediction
pred = knn.predict([[1.5]])
print(pred)  # Output: [0]
