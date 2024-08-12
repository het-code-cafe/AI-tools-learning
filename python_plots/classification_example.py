from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt
import numpy as np

output_path = "./output/"
theme = "ocean"

# Generate a simple dataset for classification
# Let's consider Feature 1 as 'Age' and Feature 2 as 'Blood Pressure'
X, y = make_classification(
    n_samples=200,
    n_features=2,
    n_informative=2,
    n_redundant=0,
    n_clusters_per_class=1,
    n_classes=2,
    random_state=42
)

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Train a simple logistic regression classifier
classifier = LogisticRegression()
classifier.fit(X_train, y_train)

# Plot the decision boundary
plt.figure(figsize=(8, 6))

# Create a mesh to plot the decision boundary
x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.1),
                     np.arange(y_min, y_max, 0.1))

Z = classifier.predict(np.c_[xx.ravel(), yy.ravel()])
Z = Z.reshape(xx.shape)

plt.contourf(xx, yy, Z, alpha=0.4, cmap=theme)
plt.scatter(X[:, 0], X[:, 1], c=y, cmap=theme, edgecolor='k', s=100)

plt.title("Classification Example with Decision Boundary", fontsize=16)
plt.xlabel("Age", fontsize=14)
plt.ylabel("Blood Pressure", fontsize=14)
plt.grid(True)
plt.savefig(output_path + "classification_decision_boundary.png")

# Generate predictions and create a confusion matrix
y_pred = classifier.predict(X_test)
cm = confusion_matrix(y_test, y_pred)

# Plot the confusion matrix
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=classifier.classes_)
disp.plot(cmap=theme)
plt.title("Confusion Matrix", fontsize=16)
plt.savefig(output_path + "classification_confusion_matrix.png")
