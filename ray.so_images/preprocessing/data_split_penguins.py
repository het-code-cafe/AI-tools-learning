import pandas as pd
import seaborn as sns
from sklearn.model_selection import train_test_split

# Target column
TARGET = 'species'

# Load the Penguins dataset: in this case we prepare the
# dataset to predict the species
df = sns.load_dataset('penguins')

# Drop rows with any missing values for the demo
df = df.dropna(how="any")

# Select features and target variable
y = df[TARGET]
X = df.drop([TARGET], axis=1)

# Drop rows with missing values in the selected features
X = X.dropna()

# Handle categorical variables using one-hot encoding
X = pd.get_dummies(X, drop_first=True)

# Split the data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Output the shapes of the resulting datasets
print(f"X_train shape: {X_train.shape}")
print(f"X_test shape: {X_test.shape}")
print(f"y_train shape: {y_train.shape}")
print(f"y_test shape: {y_test.shape}")
