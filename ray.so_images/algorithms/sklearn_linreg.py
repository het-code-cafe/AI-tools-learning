"""
sklearn_linreg.py

Linear Regression using the sklearn library.
"""
from sklearn.linear_model import LinearRegression

# Example data y = ax + b (with a = 1, b = 0)
X = [[1], [2], [3], [4], [5]]
y = [1, 2, 3, 4, 5]

# Model
model = LinearRegression()
model.fit(X, y)

# Prediction
pred = model.predict([[6]])
print(pred)  # Output: [6.0]
