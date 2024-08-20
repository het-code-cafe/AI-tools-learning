"""
sklearn_gradient_boosting.py
"""
from sklearn.ensemble import GradientBoostingClassifier

# Example data
X = [[1, 2], [3, 4], [5, 6], [7, 8]]
y = [0, 0, 1, 1]

# Model
clf = GradientBoostingClassifier()
clf.fit(X, y)

# Prediction
pred = clf.predict([[4, 5]])
print(pred)  # Output: [1]
