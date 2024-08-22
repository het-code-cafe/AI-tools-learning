"""
sklearn_knn_regression.py

K-Nearest Neighbors model to predict housing prices based on various features.
"""
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.datasets import load_diabetes
from sklearn.neighbors import KNeighborsRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score

pd.set_option('display.max_columns', None)

diabetes = load_diabetes(as_frame=True)
X = diabetes.data
y = diabetes.target

scaler = StandardScaler()
y = scaler.fit_transform(y.values.reshape(-1, 1))
#print(y)

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
