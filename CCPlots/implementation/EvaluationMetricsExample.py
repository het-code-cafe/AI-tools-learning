import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

from CCPlots.PlotExample import PlotExample
from CCPlots.config import OUTPUT_PATH, CMAP_WHITE


class EvaluationMetricsExample(PlotExample):

    def main(self):
        # Define the confusion matrix values again
        conf_matrix = np.array([[50, 10],  # 50 True Negatives, 10 False Positives
                                [5, 35]])  # 5 False Negatives, 35 True Positives

        # Labels for the plot including counts
        group_labels = np.array([
            f"True Negative\n(Healthy correctly diagnosed)\n{conf_matrix[0, 0]}",
            f"False Positive\n(Healthy misdiagnosed as sick)\n{conf_matrix[0, 1]}",
            f"False Negative\n(Sick misdiagnosed as healthy)\n{conf_matrix[1, 0]}",
            f"True Positive\n(Sick correctly diagnosed)\n{conf_matrix[1, 1]}"
        ]).reshape(2, 2)

        # Create the heatmap
        plt.figure(figsize=(8, 6))
        ax = sns.heatmap(conf_matrix, annot=group_labels, fmt="", cmap=CMAP_WHITE, cbar=True,
                         xticklabels=["Predicted Healthy", "Predicted Sick"],
                         yticklabels=["Is Healthy", "Is Sick"])

        # Add labels and title
        plt.xlabel("Predicted Label")
        plt.ylabel("Actual Label")
        plt.title("Evaluating Results for Disease Diagnosis")

        # Show the plot
        plt.savefig(OUTPUT_PATH + "confusion_matrix.png")

if __name__ == "__main__":
    EvaluationMetricsExample().main()