"""
RegressionExample.py

Plot an example of a distribution we would use for a regression problem. Generates a plot for the slides.
"""

import matplotlib.pyplot as plt
import numpy as np
import matplotlib.ticker as ticker

from CCPlots.PlotExample import PlotExample
from CCPlots.config import OUTPUT_PATH, COLOR_PALETTE


class RegressionExample(PlotExample):

    # Configure the colours for the plot
    dark_green = COLOR_PALETTE['base_colors']['dark_green']
    green = COLOR_PALETTE['base_colors']['medium_green']
    yellow = COLOR_PALETTE['base_colors']['bright_yellow']

    def main(self):

        # Generate data for the normal distribution curve
        mean = 159
        std_dev = 6.1
        x = np.linspace(mean - 4*std_dev, mean + 4*std_dev, 1000)
        y = (1/(std_dev * np.sqrt(2 * np.pi))) * np.exp(-0.5 * ((x - mean) / std_dev)**2)

        # Plot the normal distribution
        plt.figure(figsize=(10, 6))
        plt.plot(x, y, label="Female Height Distribution World-wide", color=COLOR_PALETTE['base_colors']['medium_green'])

        # Shade the regions under the curve
        plt.fill_between(x, y, where=(x <= mean - 2*std_dev), color=self.green, alpha=0.4)
        plt.fill_between(x, y, where=((x > mean - 2*std_dev) & (x <= mean - std_dev)), color=self.yellow, alpha=0.4)
        plt.fill_between(x, y, where=((x > mean - std_dev) & (x < mean + std_dev)), color=self.green, alpha=0.4)
        plt.fill_between(x, y, where=((x >= mean + std_dev) & (x < mean + 2*std_dev)), color=self.yellow, alpha=0.4)
        plt.fill_between(x, y, where=(x >= mean + 2*std_dev), color=self.green, alpha=0.4)

        # Add labels for the IQ scores and standard deviations
        plt.axvline(mean, color=self.dark_green, linestyle='dashed', linewidth=1)
        plt.axvline(mean - std_dev, color=self.dark_green, linestyle='dashed', linewidth=1)
        plt.axvline(mean + std_dev, color=self.dark_green, linestyle='dashed', linewidth=1)
        plt.axvline(mean - 2*std_dev, color=self.dark_green, linestyle='dashed', linewidth=1)
        plt.axvline(mean + 2*std_dev, color=self.dark_green, linestyle='dashed', linewidth=1)

        plt.text(mean, max(y)*0.9, '100', ha='center', color='#113428', fontsize=12)
        plt.text(mean - std_dev, max(y)*0.9, '85', ha='center', color='#113428', fontsize=12)
        plt.text(mean + std_dev, max(y)*0.9, '115', ha='center', color='#113428', fontsize=12)
        plt.text(mean - 2*std_dev, max(y)*0.9, '70', ha='center', color='#113428', fontsize=12)
        plt.text(mean + 2*std_dev, max(y)*0.9, '130', ha='center', color='#113428', fontsize=12)

        # Set plot labels and title
        plt.title("Female Height Distribution World-wide", fontsize=16)
        plt.xlabel("Height in cms (mean=159, std=6.1)", fontsize=14)
        plt.ylabel("Probability Density", fontsize=14)
        plt.grid(True)

        # Customize the x-axis
        ax = plt.gca()
        ax.xaxis.set_major_locator(ticker.MultipleLocator(15))
        ax.xaxis.set_minor_locator(ticker.MultipleLocator(5))
        ax.grid(True, which='both', linestyle='--', linewidth=0.5)

        # Show the plot
        plt.savefig(OUTPUT_PATH + "regression_example.png")


if __name__ == "__main__":
    RegressionExample().main()
