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

RANDOM_SEED = 42

class ClassificationExample(PlotExample):

    # We will hold our data and classifier later
    X, y = None, None
    X_train, X_test, y_train, y_test = None, None, None, None
    classifier: LogisticRegression = None

    def main(self):
        """
        Execute a LR on synth data and plot a decision boundary and confusion matrix
        in English and Dutch
        """
        # Generate dummy data
        self.generate_data()

        # Split the data into training and testing sets
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            self.X, self.y, test_size=0.3, random_state=RANDOM_SEED)

        # Train LR classification model
        self.train_classifier()

        # Plot the decision boundary with English text
        self.plot_decision_boundary(
            fname="classification_decision_boundary.png",
            title="Classification Example with Decision Boundary",
            xlabel="Age",
            ylabel="Blood Pressure"
        )

        # En nu in het Nederlands hoppa
        self.plot_decision_boundary(
            fname="classification_decision_boundary_NL.png",
            title="Diabetes classificatievoorbeeld",
            xlabel="Leeftijd",
            ylabel="Bloeddruk"
        )

        # Plot the confusion matrix with English text
        self.plot_confusion_matrix(
            fname="classification_confusion_matrix.png",
            title="Confusion Matrix",
            xlabel="Predicted label",
            ylabel="True label"
        )

        # En nu weer in het nederlands hatsa
        self.plot_confusion_matrix(
            fname="classification_confusion_matrix_NL.png",
            title="Verwarringsmatrix",
            xlabel="Voorspeld label",
            ylabel="Werkelijk label"
        )

    def plot_confusion_matrix(self, fname: str, title: str,
                              xlabel: str = "Predicted label",
                              ylabel: str = "True label") -> None:
        # Generate predictions and create a confusion matrix
        y_pred = self.classifier.predict(self.X_test)
        cm = confusion_matrix(self.y_test, y_pred)

        # Plot the confusion matrix
        disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=self.classifier.classes_)
        disp.plot(cmap=THEME)
        plt.title(title, fontsize=16)
        plt.xlabel(xlabel, fontsize=14)
        plt.ylabel(ylabel, fontsize=14)
        plt.savefig(OUTPUT_PATH + fname)

    def plot_decision_boundary(self, fname: str, title: str,
                               xlabel: str = "Age",
                               ylabel: str = "Blood Pressure") -> None:
        """ Plot the decision boundary and save the file with the desired text. """
        # Create a mesh to plot the decision boundary
        x_min, x_max = self.X[:, 0].min() - 1, self.X[:, 0].max() + 1
        y_min, y_max = self.X[:, 1].min() - 1, self.X[:, 1].max() + 1
        xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.1),
                             np.arange(y_min, y_max, 0.1))

        Z = self.classifier.predict(np.c_[xx.ravel(), yy.ravel()])
        Z = Z.reshape(xx.shape)

        # Plot the decision boundary
        plt.figure(figsize=(8, 6))

        plt.contourf(xx, yy, Z, alpha=0.4, cmap=THEME)
        plt.scatter(self.X[:, 0], self.X[:, 1],
                    c=self.y,
                    cmap=THEME,
                    edgecolor=COLOR_PALETTE['base_colors']['dark_green'],
                    s=100)

        plt.title(title, fontsize=16)
        plt.xlabel(xlabel, fontsize=14)
        plt.ylabel(ylabel, fontsize=14)
        plt.grid(True)
        plt.savefig(OUTPUT_PATH + fname)

    def generate_data(self) -> None:
        """ Generate some dummy data to simulate a diabetes classification problem. """
        # Generate a simple dataset for classification, where feature 1 is 'Age' and
        # feature 2 is 'Blood Pressure'
        self.X, self.y = make_classification(
            n_samples=200,
            n_features=2,
            n_informative=2,
            n_redundant=0,
            n_clusters_per_class=1,
            n_classes=2,
            random_state=RANDOM_SEED
        )

    def train_classifier(self) -> None:
        """ Train a Logistic Regression model on the data """
        # Train a simple logistic regression classifier
        self.classifier = LogisticRegression()
        self.classifier.fit(self.X_train, self.y_train)


if __name__ == "__main__":
    # Maintain this exactly, or we won't be able to regenerate every plot
    # resulting from this code.
    ClassificationExample().main()
