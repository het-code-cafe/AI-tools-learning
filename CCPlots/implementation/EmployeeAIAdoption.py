import matplotlib.pyplot as plt

from CCPlots.PlotExample import PlotExample
from CCPlots.config import COLOR_PALETTE

# Colour scheme
BAR_C = [
        COLOR_PALETTE["base_colors"]["dark_green"],
        COLOR_PALETTE["base_colors"]["medium_green"],
        COLOR_PALETTE["base_colors"]["bright_teal"],
        COLOR_PALETTE["base_colors"]["bright_yellow"],
    ]

class EmployeeAIAdoption(PlotExample):

    def main(self):
        # Data for the bar chart
        categories = ['Tijd bespaard', 'Focus verbeterd', 'Creativiteit verhoogd', 'Meer werkplezier']
        values = [90, 85, 84, 83]

        # Create the bar chart
        plt.figure(figsize=(10, 5))
        plt.bar(categories, values, color=BAR_C)
        plt.title('Impact van AI op werkervaring (% van de medewerkers eens)')
        plt.ylabel('Percentage')
        plt.ylim(0, 100)

        # Show the plot
        plt.show()

if __name__ == '__main__':
    EmployeeAIAdoption().main()