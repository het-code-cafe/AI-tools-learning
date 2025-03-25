from CCPlots.PlotExample import PlotExample
from CCPlots.config import OUTPUT_PATH, COLOR_PALETTE


class PerceptronExample(PlotExample):

    PERCEPTRON_COLOR = COLOR_PALETTE['base_colors']['bright_yellow']

    def main(self) -> None:
        import matplotlib.pyplot as plt
        import matplotlib.patches as patches

        # Set up the figure
        fig, ax = plt.subplots(figsize=(8, 6))
        ax.set_xlim(0, 10)
        ax.set_ylim(0, 10)
        ax.axis('off')

        # Draw inputs with adjusted arrow destinations
        inputs = ['x1', 'x2', 'x3']
        weights = ['w1', 'w2', 'w3']
        input_positions = [6.5, 5.0, 3.5]  # Y positions for input lines
        for i, (x, w, y) in enumerate(zip(inputs, weights, input_positions)):
            ax.annotate(x, (1, y), fontsize=12, ha='right')
            ax.annotate(w, (2.2, y + 0.4), fontsize=10, ha='left')
            ax.arrow(1.2, y, 2.3, 5 - y, head_width=0.1, head_length=0.2, fc='black', ec='black',
                     length_includes_head=True)

        # Draw perceptron circle
        circle = patches.Circle((4.5, 5), 1.2, edgecolor='black', facecolor=self.PERCEPTRON_COLOR, lw=2)
        ax.add_patch(circle)
        ax.text(4.5, 5.4, r'$\sum (w_i x_i) + b$', fontsize=12, ha='center')
        ax.text(4.5, 4.3, 'Activation', fontsize=10, ha='center')

        # Draw output
        ax.arrow(5.7, 5, 2.5, 0, head_width=0.1, head_length=0.2, fc='black', ec='black', length_includes_head=True)
        ax.text(8.5, 5, 'Output', fontsize=12, ha='left')

        plt.tight_layout()
        plt.savefig(OUTPUT_PATH + "perceptron_schematic.png")

if __name__ == "__main__":
    PerceptronExample().main()