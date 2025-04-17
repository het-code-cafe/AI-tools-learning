"""
ContinuousDiscreteExample.py

This example illustrates the difference between categorical and continuous variables.
The intention is to also illustrate the concept of binning, which can turn a regression
problem into a classification problem.

@TODO: Finish up this example, it can be made clearer.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from CCPlots.PlotExample import PlotExample
from CCPlots.config import COLOR_PALETTE


class ContinuousDiscreteExample(PlotExample):

    # Set colours for the plot
    green = COLOR_PALETTE['base_colors']['medium_green']
    mint = COLOR_PALETTE['accent_colors']['mint_green']

    def main(self) -> None:
        # Simulated continuous data: ages (with some exceptionally high ages for realism)
        np.random.seed(42)
        ages = np.concatenate([
            np.random.normal(loc=35, scale=10, size=450),   # Most people between 25-45
            np.random.normal(loc=75, scale=8, size=40),     # Older age group
            np.random.normal(loc=95, scale=3, size=10)      # Exceptionally old
        ])
        ages = np.clip(ages, 0, 100)  # Keep all values between 0 and 100

        # Bin the continuous data into categories
        bins = [0, 18, 25, 35, 50, 65, 80, 100]
        labels = ['<18', '18–24', '25–34', '35–49', '50–64', '65–79', '80+']
        age_bins = pd.cut(ages, bins=bins, labels=labels, right=False)

        df = pd.DataFrame({
            'Age': ages,
            'Age Group': age_bins
        })

        # Plot
        fig, axs = plt.subplots(1, 2, figsize=(14, 5))

        # Categorical version: bar plot
        sns.countplot(x='Age Group', data=df, ax=axs[1], color=self.mint, order=labels)
        axs[1].set_title("Categorical Variable: Age Bins")
        axs[1].set_xlabel("Age Group")

        # Continuous data: histogram
        sns.histplot(df['Age'], kde=True, ax=axs[0], color=self.green)
        axs[0].set_title("Continuous Variable: Age Distribution")
        axs[0].set_xlabel("Age")

        plt.tight_layout()
        plt.show()


if __name__ == "__main__":
    ContinuousDiscreteExample().main()
