import matplotlib.pyplot as plt
import numpy as np

from CCPlots.PlotExample import PlotExample
from CCPlots.config import COLOR_PALETTE, OUTPUT_PATH


class TokenizationExample(PlotExample):

    sentence: str
    subword_tokens: str

    green = COLOR_PALETTE['base_colors']['medium_green']

    def __init__(self, sentence = "Tokenization is essential for NLP models!", tokens = ["Tok", "en", "ization", "is", "es", "sen", "tial", "for", "NLP", "mod", "els", "!"]):
        # Example sentence
        self.sentence = sentence

        # Example of subword tokenization (simulated BPE-style tokens)
        self.subword_tokens = tokens

    def main(self):

        # Define positions for the tokens in the visualization
        x_positions = np.arange(len(self.subword_tokens))

        # Create the plot
        plt.figure(figsize=(12, 3))
        plt.scatter(x_positions, [1]*len(self.subword_tokens), color=self.green, s=100)

        # Add subword tokens as labels
        for i, token in enumerate(self.subword_tokens):
            plt.text(x_positions[i], 1.05, token, ha='center', fontsize=12, fontweight='bold')

        # Add connecting arrows to show the subword tokenization process
        for i in range(len(self.subword_tokens) - 1):
            plt.arrow(x_positions[i], 1, x_positions[i+1] - x_positions[i] - 0.2, 0,
                      head_width=0.05, head_length=0.1, fc='black', ec='black')

        # Formatting the visualization
        plt.xticks([])
        plt.yticks([])
        plt.title("Subword Tokenization Process", fontsize=14, fontweight='bold')
        plt.ylim(0.8, 1.2)
        plt.grid(False)
        plt.savefig(OUTPUT_PATH + "tokenization_example.png")

if __name__ == "__main__":
    TokenizationExample().main()
