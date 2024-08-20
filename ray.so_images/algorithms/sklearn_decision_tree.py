"""
sklearn_decision_tree.py
"""
from sklearn.tree import DecisionTreeClassifier

# Example data
X = [[0, 0], [1, 1], [2, 2], [3, 3]]
y = [0, 0, 1, 1]

# Model
clf = DecisionTreeClassifier()
clf.fit(X, y)

# Prediction
pred = clf.predict([[2, 2]])
print(pred)  # Output: [1]
