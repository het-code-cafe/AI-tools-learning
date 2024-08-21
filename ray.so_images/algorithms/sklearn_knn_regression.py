"""
sklearn_knn_regression.py

K-Nearest Neighbors model to predict housing prices based on various features.
"""
from sklearn.datasets import load_boston
from sklearn.neighbors import KNeighborsRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score

# Load the Boston Housing dataset
boston = load_boston()
X = boston.data  # Features: Various factors like crime rate, number of rooms, etc.
y = boston.target  # Labels: Median value of owner-occupied homes (in $1000s)

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Model
knn = KNeighborsRegressor(n_neighbors=5)
knn.fit(X_train, y_train)

# Prediction
y_pred = knn.predict(X_test)

# Evaluate the model
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)
print(f"Mean Squared Error: {mse:.2f}")
print(f"R^2 Score: {r2:.2f}")

# Example prediction
sample = [[0.1, 0.0, 6.2, 1.0, 0.5, 7.5, 60.0, 4.0, 1.0, 296.0, 15.3, 390.0, 2.0]]  # Example feature set
pred = knn.predict(sample)
print(f"Predicted price for the sample: ${pred[0] * 1000:.2f}")
