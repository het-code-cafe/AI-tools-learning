"""
sklearn_logistic_regression.py

Logistic Regression model
"""
from sklearn.linear_model import LogisticRegression

# Example data
X = [[1], [2], [3], [4], [5]]
y = [0, 0, 1, 1, 1]

# Model
model = LogisticRegression()
model.fit(X, y)

# Prediction
pred = model.predict([[2.5]])
print(pred)  # Output: [0]
