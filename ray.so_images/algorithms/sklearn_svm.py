"""
sklearn_svm.py

Support Vector Machine example
"""

from sklearn import svm

# Example data
X = [[0, 0], [1, 1], [2, 2], [3, 3]]
y = [0, 0, 1, 1]

# Model
clf = svm.SVC(kernel='linear')
clf.fit(X, y)

# Prediction
pred = clf.predict([[1.5, 1.5]])
print(pred)  # Output: [1]
