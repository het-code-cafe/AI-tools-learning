import statsmodels.api as sm
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt

# Load the Pearson dataset
df = sm.datasets.get_rdataset("pearson", "HistData").data

# Inspect the first few rows of the dataset
print(df.head())

# Select the features and target variable
X = df[['Father']].values  # Father's height as feature
y = df['Son'].values       # Son's height as target

# Split the data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)

# Initialize and train the Linear Regression model
model = LinearRegression()
model.fit(X_train, y_train)

# Make predictions on the test set
y_pred = model.predict(X_test)

# Calculate performance metrics
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print(f"Mean Squared Error: {mse:.2f}")
print(f"R^2 Score: {r2:.2f}")

# Plot the results
plt.figure()
plt.scatter(X_test, y_test, color='blue', label='Actual heights')
plt.plot(X_test, y_pred, color='red', linewidth=2, label='Predicted heights')
plt.xlabel("Father's Height")
plt.ylabel("Son's Height")
plt.title("Linear Regression: Predicting Son's Height from Father's Height")
plt.legend()
plt.show()
