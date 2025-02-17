import numpy as np

# Function to calculate divided differences
def divided_differences(x, f):
    n = len(x)
    # Create a 2D array to store divided differences
    dd = np.zeros((n, n))
    dd[:, 0] = f  # First column is the f(x) values

    # Fill the divided difference table
    for j in range(1, n):
        for i in range(n-j):
            dd[i, j] = (dd[i+1, j-1] - dd[i, j-1]) / (x[i+j] - x[i])
    return dd

# Function to calculate f'(10) using Newton's Divided Difference Method
def calculate_derivative_at_10(x_values, f_values):
    # Calculate the divided differences table
    dd_table = divided_differences(x_values, f_values)

    # Finding the interval that contains 10 (between 5 and 11 in this case)
    x_target = 10
    i = 1  # Start with the interval between 5 and 11

    # Newton's interpolation formula for the first derivative at x = 10
    f_prime_10 = dd_table[0, 1]  # The first derivative at 5

    for j in range(2, i+1):
        f_prime_10 += dd_table[0, j] * np.prod(x_target - x_values[:j])

    return f_prime_10

# Example data (from the task)
x_values = np.array([3, 5, 11, 27, 34])  # x values
f_values = np.array([-13, 23, 899, 17315, 35606])  # f(x) values

# Calculate and output the result
derivative_at_10 = calculate_derivative_at_10(x_values, f_values)

print(f"The derivative at x = 10 is: {derivative_at_10}")
