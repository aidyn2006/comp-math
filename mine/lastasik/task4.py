import numpy as np

# Function to calculate divided differences
def divided_differences(x, f):
    n = len(x)
    dd = np.zeros((n, n))  # Create a table for divided differences
    dd[:, 0] = f  # First column is the f(x) values

    # Calculate the divided differences for higher order derivatives
    for j in range(1, n):
        for i in range(n-j):
            dd[i, j] = (dd[i+1, j-1] - dd[i, j-1]) / (x[i+j] - x[i])
    return dd

# Function to calculate the first and second derivatives at x=1.1
def calculate_derivatives_at_1_1(x_values, f_values):
    # Calculate the divided differences table
    dd_table = divided_differences(x_values, f_values)

    # Extract the first and second derivatives at x=1.1
    first_derivative = dd_table[0, 1]  # First derivative at x=1.1
    second_derivative = dd_table[0, 2]  # Second derivative at x=1.1

    return first_derivative, second_derivative

# Example data (from the task)
x_values_task_4 = np.array([1.0, 1.2, 1.4, 1.6, 1.8, 2.0])  # x values
f_values_task_4 = np.array([0, 0.128, 0.544, 1.296, 2.432, 4.000])  # f(x) values

# Calculate the derivatives and output the result
first_derivative, second_derivative = calculate_derivatives_at_1_1(x_values_task_4, f_values_task_4)

print(f"First derivative at x=1.1: {first_derivative}")
print(f"Second derivative at x=1.1: {second_derivative}")
