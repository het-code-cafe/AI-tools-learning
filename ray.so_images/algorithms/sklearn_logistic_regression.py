"""
sklearn_logistic_regression.py

Logistic Regression model to predict student pass/fail based on study hours.
"""
from sklearn.linear_model import LogisticRegression

# Example data: [hours studied, assignments completed]
X = [
    [1, 1], [2, 1], [3, 2], [4, 2], [5, 3],
    [6, 3], [7, 4], [8, 4], [9, 5], [10, 5]
]
# Labels: 0 = Fail, 1 = Pass
y = [0, 0, 0, 0, 1, 1, 1, 1, 1, 1]

# Model
model = LogisticRegression()
model.fit(X, y)

# Prediction: Student putting in less effort
pred_less_effort = model.predict([[4, 1]])
print(f"Prediction for less effort: {pred_less_effort[0]}")  # Output: [0] (likely Fail)
prob_less_effort = model.predict_proba([[4, 1]])
print(f"Probability for less effort: {prob_less_effort[0]}")  # Output: [[probability_of_fail, probability_of_pass]]

# Prediction: Student putting in above average effort
pred_above_average = model.predict([[7, 4]])
print(f"Prediction for above average effort: {pred_above_average[0]}")  # Output: [1] (likely Pass)
prob_above_average = model.predict_proba([[7, 4]])
print(f"Probability for above average effort: {prob_above_average[0]}")  # Output: [[probability_of_fail, probability_of_pass]]
