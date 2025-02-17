import numpy as np
import matplotlib.pyplot as plt

# Example data
x_data = np.linspace(0, 10, 100)
y_data = np.sin(x_data) + 0.5 * np.random.normal(size=len(x_data))


# Function to calculate RSS
def calculate_rss(y_true, y_pred):
    return np.sum((y_true - y_pred) ** 2)


# List of polynomial degrees to try
degrees = [2, 3, 4, 5]

# Dictionary to store RSS for each degree
rss_values = {}

# Fit polynomials, calculate RSS, and plot them
plt.scatter(x_data, y_data, label='Data')

for degree in degrees:
    coeffs = np.polyfit(x_data, y_data, degree)
    poly = np.poly1d(coeffs)
    y_pred = poly(x_data)

    rss = calculate_rss(y_data, y_pred)
    rss_values[degree] = rss

    plt.plot(x_data, y_pred, label=f'Degree {degree}')

plt.legend()
plt.show()

# Print RSS values for each degree
for degree, rss in rss_values.items():
    print(f'Degree {degree}: RSS = {rss}')

# Find the degree with the minimum RSS
best_degree = min(rss_values, key=rss_values.get)
print(f'Best degree: {best_degree}')
