"""
sklearn_gaussianNB.py

Gaussian Naive Bayes classifier
"""
from sklearn.naive_bayes import GaussianNB

# Example data
X = [[1, 2], [2, 3], [3, 4], [4, 5]]
y = [0, 0, 1, 1]

# Model
gnb = GaussianNB()
gnb.fit(X, y)

# Prediction
pred = gnb.predict([[2, 2]])
print(pred)  # Output: [0]