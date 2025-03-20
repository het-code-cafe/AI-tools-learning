import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from CCPlots import PlotExample
from CCPlots.config import OUTPUT_PATH, COLOR_PALETTE

# Implement the noise illustration
class NoiseIllustration(PlotExample):

    green: str = COLOR_PALETTE['base_colors']['medium_green']
    mint: str = COLOR_PALETTE["accent_colors"]["mint_green"]

    def main(self) -> None:
        """ Generates and plots a noisy sine wave with the actual function """

        # Set seaborn style
        sns.set_style("whitegrid")

        # Generate x values
        x = np.linspace(0, 10, 100)

        # Generate actual function (sine wave)
        y_actual = np.sin(x)

        # Generate noisy data
        noise = np.random.normal(0, 0.3, size=x.shape)  # Adding Gaussian noise
        y_noisy = y_actual + noise

        # Create the plot
        plt.figure(figsize=(8, 5))
        plt.plot(x, y_actual, label="True Function (sin(x))", color=self.green, linewidth=2)  # True function
        plt.scatter(x, y_noisy, label="Noisy Observations", color=self.mint, alpha=0.6)  # Noisy points

        # Labels and legend
        plt.xlabel("X Values")
        plt.ylabel("Y Values")
        plt.title("Illustration of Noise in Data")
        plt.legend()

        plt.savefig(OUTPUT_PATH + "noisy_data_example.png")


if __name__ == "__main__":
    NoiseIllustration().main()