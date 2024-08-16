"""
OverfittingUnderfittingExample.py
"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

from CCPlots.config import OUTPUT_PATH, COLOR_PALETTE


class OverfittingUnderfittingExample:

    medium_green = COLOR_PALETTE['base_colors']['medium_green']
    mint_green = COLOR_PALETTE['accent_colors']['mint_green']
    periwinkle_blue = COLOR_PALETTE['accent_colors']['periwinkle_blue']
    light_gray = COLOR_PALETTE['neutral_colors']['light_gray']

    def main(self):
        # Generate synthetic data
        np.random.seed(0)
        X = np.sort(np.random.rand(40, 1) * 10, axis=0)
        y = np.sin(X).ravel() + np.random.normal(0, 0.2, X.shape[0])

        # Split the data into training and testing sets
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.5, random_state=0)

        # Create polynomial features for underfitting and overfitting
        poly_under = PolynomialFeatures(degree=2)
        poly_over = PolynomialFeatures(degree=11)

        X_train_under = poly_under.fit_transform(X_train)
        X_test_under = poly_under.transform(X_test)

        X_train_over = poly_over.fit_transform(X_train)
        X_test_over = poly_over.transform(X_test)

        # Fit linear regression models
        model_under = LinearRegression().fit(X_train_under, y_train)
        model_over = LinearRegression().fit(X_train_over, y_train)

        # Generate predictions for the entire range of X for plotting
        # using both the overfitting and underfitting model
        X_range = np.linspace(X.min(), X.max(), 500).reshape(-1, 1)
        X_range_under = poly_under.transform(X_range)
        X_range_over = poly_over.transform(X_range)

        # Generate a continuous line of predictions for plotting
        y_range_pred_under = model_under.predict(X_range_under)
        y_range_pred_over = model_over.predict(X_range_over)

        # Get training and test predictions for both models
        y_pred_under_train = model_under.predict(X_train_under)
        y_pred_under_test = model_under.predict(X_test_under)

        y_pred_over_train = model_over.predict(X_train_over)
        y_pred_over_test = model_over.predict(X_test_over)

        # Calculate MSE
        mse_under_train = mean_squared_error(y_train, y_pred_under_train)
        mse_under_test = mean_squared_error(y_test, y_pred_under_test)

        mse_over_train = mean_squared_error(y_train, y_pred_over_train)
        mse_over_test = mean_squared_error(y_test, y_pred_over_test)

        # Plotting
        plt.figure(figsize=(14, 6))

        # Underfitting
        plt.subplot(1, 2, 1)
        plt.scatter(X_train, y_train, color=self.mint_green, label='Training Data')
        plt.scatter(X_test, y_test, color=self.periwinkle_blue, label='Test Data')
        plt.plot(X_range, y_range_pred_under, color=self.medium_green, label='Model (Underfitting)')
        plt.title(f'Underfitting\nTrain MSE: {mse_under_train:.2f}, Test MSE: {mse_under_test:.2f}')
        plt.xlabel('X')
        plt.ylabel('y')
        plt.legend()
        plt.grid(True, c=self.light_gray)

        # Overfitting
        plt.subplot(1, 2, 2)
        plt.scatter(X_train, y_train, color=self.mint_green, label='Training Data')
        plt.scatter(X_test, y_test, color=self.periwinkle_blue, label='Test Data')
        plt.plot(X_range, y_range_pred_over, color=self.medium_green, label='Model (Overfitting)')
        plt.title(f'Overfitting\nTrain MSE: {mse_over_train:.2f}, Test MSE: {mse_over_test:.2f}')
        plt.xlabel('X')
        plt.ylabel('y')
        plt.legend()
        plt.grid(True, c=self.light_gray)

        plt.tight_layout()
        plt.savefig(OUTPUT_PATH + "overfitting_underfitting.png")