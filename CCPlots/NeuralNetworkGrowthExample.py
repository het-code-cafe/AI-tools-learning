"""
NeuralNetworkGrowthExample.py
"""

import matplotlib.pyplot as plt

from CCPlots.config import OUTPUT_PATH, COLOR_PALETTE


class NeuralNetworkGrowthExample:

    # Plot colours
    deep_teal = COLOR_PALETTE['analogous_colors']['deep_teal']
    mint_green = COLOR_PALETTE['accent_colors']['mint_green']
    light_gray = COLOR_PALETTE['neutral_colors']['light_gray']

    def main(self):
        # Data for the models
        years = [2012, 2014, 2018, 2020]
        models = ["AlexNet", "VGG-16", "BERT", "GPT-3"]
        parameters = [60e6, 138e6, 340e6, 175e9]

        # Plotting the line figure
        plt.figure(figsize=(10, 6))
        plt.plot(years, parameters, marker='o', color=self.deep_teal)
        plt.fill_between(years, parameters, color=self.mint_green, alpha=0.6)
        for i, txt in enumerate(models):
            plt.text(years[i], parameters[i], txt, fontsize=10, ha='right')

        # Plot y logarithmically (exponential growth)
        plt.yscale('log')

        # Plot labeling
        plt.title("Growth in Neural Network Parameters Over Time", fontsize=14)
        plt.xlabel("Year")
        plt.xlim(min(years), max(years))
        plt.ylabel("Parameters (log scale)")

        plt.grid(True, color=self.light_gray)
        plt.savefig(OUTPUT_PATH + "neural_network_growth_line_log.png")


if __name__ == "__main__":
    NeuralNetworkGrowthExample().main()
