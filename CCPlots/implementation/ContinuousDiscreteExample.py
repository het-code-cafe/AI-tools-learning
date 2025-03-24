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
        # Simulated data
        df = pd.DataFrame({
            'Categorical': np.random.choice(['Red', 'Green', 'Blue', 'Yellow', 'Purple', 'Orange'], size=200),
            'Continuous': np.random.normal(loc=70, scale=10, size=200)
        })

        # Plot
        fig, axs = plt.subplots(1, 2, figsize=(12, 4))

        sns.countplot(x='Categorical', data=df, ax=axs[0], color=self.green)
        axs[0].set_title("Categorical Variable (Bar Plot)")

        sns.histplot(df['Continuous'], kde=True, ax=axs[1], color=self.green)
        axs[1].set_title("Continuous Variable (Histogram)")

        plt.tight_layout()
        plt.show()

if __name__ == "__main__":
    ContinuousDiscreteExample().main()