import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from sklearn.model_selection import KFold
import seaborn as sns

from CCPlots.PlotExample import PlotExample
from CCPlots.config import OUTPUT_PATH, COLOR_PALETTE


class KFoldExample(PlotExample):

    # Define a list of colors, in this case we specifically need two!
    colors = [COLOR_PALETTE['base_colors']['medium_green'], COLOR_PALETTE['base_colors']['bright_yellow']]

    def __init__(self):
        # Initiate the color map
        self.cmap = ListedColormap(self.colors)

    def main(self):
        # Sample dataset
        df = sns.load_dataset('penguins').dropna(subset=['species'])
        X = df.drop(['species', 'island', 'sex'], axis=1).dropna()

        # Initialize KFold cross-validation
        kf = KFold(n_splits=5, shuffle=True, random_state=42)

        # Create an empty array to mark training and testing samples
        n_samples = len(X)
        cv_splits = np.zeros((kf.get_n_splits(), n_samples))

        # Create an empty array to mark training and testing samples
        n_samples = len(X)
        cv_splits = np.zeros((kf.get_n_splits(), n_samples))

        # Fill the array with indices
        for i, (train_index, test_index) in enumerate(kf.split(X)):
            cv_splits[i, train_index] = 1  # Mark training indices as 1
            cv_splits[i, test_index] = 2  # Mark test indices as 2

        # Plot the K-Fold cross-validation splits
        plt.figure()
        plt.imshow(cv_splits, aspect='auto', cmap=self.cmap, interpolation='nearest')
        plt.colorbar(label='Fold number')
        plt.xlabel('Index in the dataset')
        plt.ylabel('Fold number')
        plt.title('Visualization of K-Fold Cross-Validation')
        plt.savefig(OUTPUT_PATH + "kfold_validation.png")
