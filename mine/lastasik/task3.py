import numpy as np

# Function to calculate the divided differences
def divided_differences(x, f):
    n = len(x)
    dd = np.zeros((n, n))  # Create a table for divided differences
    dd[:, 0] = f  # First column is the f(x) values

    # Calculate the divided differences for higher order derivatives
    for j in range(1, n):
        for i in range(n-j):
            dd[i, j] = (dd[i+1, j-1] - dd[i, j-1]) / (x[i+j] - x[i])
    return dd

# Function to calculate first, second, and third derivatives at x=1.5
def calculate_derivatives_at_1_5(x_values, f_values):
    # Calculate the divided differences table
    dd_table = divided_differences(x_values, f_values)

    # Extract the first, second, and third derivatives at x=1.5
    first_derivative = dd_table[0, 1]  # First derivative at x=1.5
    second_derivative = dd_table[0, 2]  # Second derivative at x=1.5
    third_derivative = dd_table[0, 3]  # Third derivative at x=1.5

    return first_derivative, second_derivative, third_derivative

# Example data (from the task)
x_values_task_3 = np.array([1.5, 2.0, 2.5, 3.0, 3.5, 4.0])  # x values
f_values_task_3 = np.array([3.375, 7.000, 13.625, 24.000, 38.875, 59.000])  # f(x) values

# Calculate the derivatives and output the result
first_derivative, second_derivative, third_derivative = calculate_derivatives_at_1_5(x_values_task_3, f_values_task_3)

print(f"First derivative at x=1.5: {first_derivative}")
print(f"Second derivative at x=1.5: {second_derivative}")
print(f"Third derivative at x=1.5: {third_derivative}")
