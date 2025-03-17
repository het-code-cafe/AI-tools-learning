import matplotlib.pyplot as plt
import matplotlib.colors as mcolors

from CCPlots.PlotExample import PlotExample
from CCPlots.config import COLOR_PALETTE, OUTPUT_PATH


class LLMPredictExample(PlotExample):

    # Define the colormap using the given colors
    custom_colormap = mcolors.ListedColormap([
        COLOR_PALETTE['base_colors']['dark_green'],
        COLOR_PALETTE['base_colors']['medium_green'],
        COLOR_PALETTE['accent_colors']['mint_green'],
        COLOR_PALETTE['base_colors']['bright_yellow'],
        COLOR_PALETTE['accent_colors']['periwinkle_blue']
    ], name="custom_colormap")

    def main(self):
        # Re-import necessary libraries after execution state reset
        import matplotlib.pyplot as plt

        # Example sentence leading to next-word prediction
        prompt = "The cat sat on the"

        # Simulated probability distribution for the next word
        next_word_probs = {
            "mat": 0.5,
            "floor": 0.2,
            "chair": 0.15,
            "roof": 0.1,
            "dog": 0.05
        }

        # Extract words and their probabilities
        words = list(next_word_probs.keys())
        probabilities = list(next_word_probs.values())

        # Create bar chart
        plt.figure(figsize=(6, 4))
        plt.bar(words, probabilities, color=self.custom_colormap.colors)

        # Formatting the chart
        plt.ylim(0, 1)
        plt.ylabel("Probability")
        plt.xlabel("Predicted Next Word")
        plt.title(f"Predicting the Next Word for: '{prompt}'")

        # Display the chart
        plt.savefig(OUTPUT_PATH + "llm_predict_next.png")

if __name__ == "__main__":
    LLMPredictExample().main()