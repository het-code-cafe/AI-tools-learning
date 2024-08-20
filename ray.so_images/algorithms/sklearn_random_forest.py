"""
sklearn_random_forest.py
"""
from sklearn.ensemble import RandomForestClassifier

# Example data
X = [[0, 0], [1, 1], [2, 2], [3, 3]]
y = [0, 0, 1, 1]

# Model
clf = RandomForestClassifier(n_estimators=10)
clf.fit(X, y)

# Prediction
pred = clf.predict([[2, 2]])
print(pred)  # Output: [1]
