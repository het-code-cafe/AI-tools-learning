"""
NeuralNetworkActivationFunctionsExample.py
"""
import numpy as np
import matplotlib.pyplot as plt

from CCPlots.PlotExample import PlotExample
from CCPlots.config import OUTPUT_PATH, COLOR_PALETTE


class NeuralNetworkActivationFunctionsExample(PlotExample):

    # Line colors
    dark_green = COLOR_PALETTE["base_colors"]["dark_green"]
    rusty_red = COLOR_PALETTE["complementary_colors"]["rusty_red"]
    medium_green = COLOR_PALETTE["base_colors"]["medium_green"]
    deep_teal = COLOR_PALETTE["analogous_colors"]["deep_teal"]
    deep_burgundy = COLOR_PALETTE["complementary_colors"]["deep_burgundy"]
    periwinkle_blue = COLOR_PALETTE["accent_colors"]["periwinkle_blue"]

    # Grid
    light_gray = COLOR_PALETTE['neutral_colors']['light_gray']

    def main(self):
        # Define the range of inputs
        x = np.linspace(-10, 10, 400)

        # Plotting the activation functions
        plt.figure(figsize=(12, 8))

        plt.subplot(2, 3, 1)
        plt.plot(x, self.sigmoid(x), color=self.dark_green)
        plt.title('Sigmoid')
        plt.grid(True, color=self.light_gray)

        plt.subplot(2, 3, 2)
        plt.plot(x, self.tanh(x), color=self.rusty_red)
        plt.title('Tanh')
        plt.grid(True, color=self.light_gray)

        plt.subplot(2, 3, 3)
        plt.plot(x, self.relu(x), color=self.medium_green)
        plt.title('ReLU')
        plt.grid(True, color=self.light_gray)

        plt.subplot(2, 3, 4)
        plt.plot(x, self.leaky_relu(x), color=self.deep_teal)
        plt.title('Leaky ReLU')
        plt.grid(True, color=self.light_gray)

        plt.subplot(2, 3, 5)
        plt.plot(x, self.swish(x), color=self.deep_burgundy)
        plt.title('Swish')
        plt.grid(True, color=self.light_gray)

        plt.subplot(2, 3, 6)
        plt.plot(x, self.softplus(x), color=self.periwinkle_blue)
        plt.title('Softplus')
        plt.grid(True, color=self.light_gray)

        plt.tight_layout()
        plt.savefig(OUTPUT_PATH + "neural_network_activation_functions.png")

    def sigmoid(self, x):
        return 1 / (1 + np.exp(-x))

    def tanh(self, x):
        return np.tanh(x)

    def relu(self, x):
        return np.maximum(0, x)

    def leaky_relu(self, x, alpha=0.01):
        return np.where(x > 0, x, alpha * x)

    def swish(self, x):
        return x * self.sigmoid(x)

    def softplus(self, x):
        return np.log(1 + np.exp(x))

if __name__ == "__main__":
    NeuralNetworkActivationFunctionsExample().main()
