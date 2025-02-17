import numpy as np
from scipy.interpolate import interp1d


def derivative_forward(x, y, h):
    """Calculate forward difference"""
    return (y[1] - y[0]) / h


def derivative_backward(x, y, h):
    """Calculate backward difference"""
    return (y[-1] - y[-2]) / h


def derivative_central(x, y, h):
    """Calculate central difference"""
    return (y[2] - y[0]) / (2 * h)


def second_derivative(x, y, h):
    """Calculate second derivative using central difference"""
    return (y[2] - 2 * y[1] + y[0]) / (h ** 2)


def third_derivative(x, y, h):
    """Calculate third derivative using central difference"""
    return (y[3] - 3 * y[2] + 3 * y[1] - y[0]) / (h ** 3)

def lagrange_interpolation(x_vals, y_vals, x_target):
    """Interpolates the y value at x_target using Lagrange polynomial."""
    n = len(x_vals)
    result = 0.0

    for i in range(n):
        term = y_vals[i]
        for j in range(n):
            if i != j:
                term *= (x_target - x_vals[j]) / (x_vals[i] - x_vals[j])
        result += term

    return result

def lagrange_derivative(x_vals, y_vals, x_target):
    """Computes the derivative of the Lagrange interpolating polynomial at x_target."""
    n = len(x_vals)
    derivative_result = 0.0

    for i in range(n):
        term = y_vals[i]
        sum_term = 0.0

        for j in range(n):
            if i != j:
                product_term = 1.0
                for k in range(n):
                    if k != i and k != j:
                        product_term *= (x_target - x_vals[k]) / (x_vals[i] - x_vals[k])
                sum_term += product_term / (x_vals[i] - x_vals[j])

        derivative_result += term * sum_term

    return derivative_result



# Problem 1: Initial acceleration
print("Problem 1: Initial acceleration")
t = np.array([0, 5, 10, 15, 20])
v = np.array([3, 14, 69, 228, None])

# Interpolating v(20) using the known points
known_t = t[:-1]
known_v = v[:-1]

v_20 = lagrange_interpolation(known_t, known_v, 20)
v[-1] = v_20

# Using forward difference for initial acceleration
h = t[1] - t[0]
initial_acceleration = derivative_forward(t[:2], v[:2], h)
print(f"Initial acceleration = {initial_acceleration} m/s²")


# Given data points
x2 = np.array([3, 5, 11, 27, 34])
fx2 = np.array([-13, 23, 899, 17315, 35606])

# Compute f'(10) using Lagrange interpolation
derivative_at_10 = lagrange_derivative(x2, fx2, 10)
print("\nProblem 2: f'(10)")
print(f"f'(10) ≈ {derivative_at_10}")

# Problem 3: First, second, and third derivatives at x=1.5
print("\nProblem 3: Derivatives at x=1.5")
x3 = np.array([1.5, 2.0, 2.5, 3.0, 3.5, 4.0])
fx3 = np.array([3.375, 7.000, 13.625, 24.000, 38.875, 59.000])
h3 = 0.5  # Step size

# Calculate derivatives using the first few points
first_deriv = derivative_forward(x3[:2], fx3[:2], h3)
second_deriv = second_derivative(x3[:3], fx3[:3], h3)
third_deriv = third_derivative(x3[:4], fx3[:4], h3)

print(f"First derivative at x=1.5: {first_deriv}")
print(f"Second derivative at x=1.5: {second_deriv}")
print(f"Third derivative at x=1.5: {third_deriv}")

# Problem 4: First and second derivatives at x=1.1
print("\nProblem 4: Derivatives at x=1.1")
x4 = np.array([1.0, 1.2, 1.4, 1.6, 1.8, 2.0])
fx4 = np.array([0, 0.128, 0.544, 1.296, 2.432, 4.00])

# Find closest points for x=1.1
idx = np.searchsorted(x4, 1.1)
if idx > 0:
    # Use forward difference for first derivative
    h4 = x4[idx] - x4[idx - 1]
    first_deriv_1_1 = (fx4[idx] - fx4[idx - 1]) / h4

    # Use central difference for second derivative if possible
    if idx < len(x4) - 1:
        second_deriv_1_1 = (fx4[idx + 1] - 2 * fx4[idx] + fx4[idx - 1]) / (h4 ** 2)

    print(f"First derivative at x=1.1: {first_deriv_1_1}")
    print(f"Second derivative at x=1.1: {second_deriv_1_1}")

# Problem 5: First and second derivatives at multiple points
print("\nProblem 5: Derivatives at multiple points")
x5 = np.array([1.00, 1.05, 1.10, 1.15, 1.20, 1.25, 1.30])
y5 = np.array([1.000, 1.025, 1.049, 1.072, 1.095, 1.118, 1.140])
h5 = 0.05  # Step size

points = [1.05, 1.25, 1.15]
for x in points:
    idx = np.searchsorted(x5, x)
    if 0 < idx < len(x5) - 1:
        # Central difference for first derivative
        first_deriv = (y5[idx + 1] - y5[idx - 1]) / (2 * h5)
        # Central difference for second derivative
        second_deriv = (y5[idx + 1] - 2 * y5[idx] + y5[idx - 1]) / (h5 ** 2)
        print(f"\nAt x = {x}:")
        print(f"First derivative (dy/dx): {first_deriv}")
        print(f"Second derivative (d²y/dx²): {second_deriv}")