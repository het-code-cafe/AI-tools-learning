"""
statsmodels_linreg.py

We can also do linear regression with different libraries. The algorithm is the same,
but there can be slight differences due to rounding.

Statsmodels is a good pick for Data Scientists, because it offers such an extensive summary of the model.
"""
from statsmodels.api import OLS, add_constant
import numpy as np

# Example data
X = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
y = np.array([2, 4, 6, 8, 10, 12, 14, 16, 18, 20])

# Add a constant (intercept) to the input data
X = add_constant(X)

# Model
model = OLS(y, X)
results = model.fit()

# Prediction
pred = results.predict([1, 8.5])  # [1, 8] includes the intercept and the new value
print(pred)  # Output: [17.0]

print(results.summary())
