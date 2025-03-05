"""
ClassificationExample.py

Plot an example of a classification problem. Generates a plot for the slides including a decision boundary
and a confusion matrix to help illustrate the concept.
"""

from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt
import numpy as np

from CCPlots.PlotExample import PlotExample
from CCPlots.config import THEME, COLOR_PALETTE, OUTPUT_PATH


class ClassificationExample(PlotExample):
    def main(self):
        # Generate a simple dataset for classification, where feature 1 is 'Age' and
        # feature 2 is 'Blood Pressure'
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

        plt.contourf(xx, yy, Z, alpha=0.4, cmap=THEME)
        plt.scatter(X[:, 0], X[:, 1], c=y, cmap=THEME, edgecolor=COLOR_PALETTE['base_colors']['dark_green'], s=100)

        plt.title("Classification Example with Decision Boundary", fontsize=16)
        plt.xlabel("Age", fontsize=14)
        plt.ylabel("Blood Pressure", fontsize=14)
        plt.grid(True)
        plt.savefig(OUTPUT_PATH + "classification_decision_boundary.png")

        # Generate predictions and create a confusion matrix
        y_pred = classifier.predict(X_test)
        cm = confusion_matrix(y_test, y_pred)

        # Plot the confusion matrix
        disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=classifier.classes_)
        disp.plot(cmap=THEME)
        plt.title("Confusion Matrix", fontsize=16)
        plt.savefig(OUTPUT_PATH + "classification_confusion_matrix.png")


if __name__ == "__main__":
    ClassificationExample().main()
