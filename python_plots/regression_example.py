import matplotlib.pyplot as plt
import numpy as np
import matplotlib.ticker as ticker

output_path = "./output/"
theme = "ocean"

# Generate data for the normal distribution curve
mean = 100
std_dev = 15
x = np.linspace(mean - 4*std_dev, mean + 4*std_dev, 1000)
y = (1/(std_dev * np.sqrt(2 * np.pi))) * np.exp(-0.5 * ((x - mean) / std_dev)**2)

# Plot the normal distribution
plt.figure(figsize=(10, 6))
plt.plot(x, y, label="Normal Distribution", color="#459578")

# Shade the regions under the curve
plt.fill_between(x, y, where=(x <= mean - 2*std_dev), color='#37EDAB', alpha=0.4)
plt.fill_between(x, y, where=((x > mean - 2*std_dev) & (x <= mean - std_dev)), color='#FED53C', alpha=0.4)
plt.fill_between(x, y, where=((x > mean - std_dev) & (x < mean + std_dev)), color='#37EDAB', alpha=0.4)
plt.fill_between(x, y, where=((x >= mean + std_dev) & (x < mean + 2*std_dev)), color='#FED53C', alpha=0.4)
plt.fill_between(x, y, where=(x >= mean + 2*std_dev), color='#37EDAB', alpha=0.4)

# Add labels for the IQ scores and standard deviations
plt.axvline(mean, color='#113428', linestyle='dashed', linewidth=1)
plt.axvline(mean - std_dev, color='#113428', linestyle='dashed', linewidth=1)
plt.axvline(mean + std_dev, color='#113428', linestyle='dashed', linewidth=1)
plt.axvline(mean - 2*std_dev, color='#113428', linestyle='dashed', linewidth=1)
plt.axvline(mean + 2*std_dev, color='#113428', linestyle='dashed', linewidth=1)

plt.text(mean, max(y)*0.9, '100', ha='center', color='#113428', fontsize=12)
plt.text(mean - std_dev, max(y)*0.9, '85', ha='center', color='#113428', fontsize=12)
plt.text(mean + std_dev, max(y)*0.9, '115', ha='center', color='#113428', fontsize=12)
plt.text(mean - 2*std_dev, max(y)*0.9, '70', ha='center', color='#113428', fontsize=12)
plt.text(mean + 2*std_dev, max(y)*0.9, '130', ha='center', color='#113428', fontsize=12)

# Set plot labels and title
plt.title("Normal Distribution of IQ Scores", fontsize=16)
plt.xlabel("IQ Score", fontsize=14)
plt.ylabel("Probability Density", fontsize=14)
plt.grid(True)

# Customize the x-axis
ax = plt.gca()
ax.xaxis.set_major_locator(ticker.MultipleLocator(15))
ax.xaxis.set_minor_locator(ticker.MultipleLocator(5))
ax.grid(True, which='both', linestyle='--', linewidth=0.5)

# Show the plot
plt.savefig(output_path + "regression_example.png")
