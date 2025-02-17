import numpy as np

# 1. Initial Acceleration
# Data
time = np.array([0, 5, 10, 15, 20])
velocity = np.array([3, 14, 69, 228, 0])  # Last value unknown, not used here
h = 5

def initial_acceleration(v, h):
    return (-3 * v[0] + 4 * v[1] - v[2]) / (2 * h)

acceleration_0 = initial_acceleration(velocity, h)

# 2. First Derivative at x = 10 (Unequally spaced points)
x2 = np.array([3, 5, 11, 27, 34])
y2 = np.array([-13, 23, 899, 17315, 35606])

def divided_differences(x, y):
    n = len(y)
    coef = np.copy(y)
    for j in range(1, n):
        coef[j:n] = (coef[j:n] - coef[j - 1]) / (x[j:n] - x[j - 1])
    return coef

# Using Newton's interpolation formula for the first derivative
coef = divided_differences(x2, y2)

# Approximate derivative at x = 10
# (Using first two points for simplicity)
fx_prime_10 = coef[1] + coef[2] * (10 - x2[0])

# 3. First, Second, Third Derivatives at x = 1.5
x3 = np.array([1.5, 2.0, 2.5, 3.0, 3.5, 4.0])
y3 = np.array([3.375, 7.000, 13.625, 24.000, 38.875, 59.000])
h3 = 0.5

def central_differences(y, h):
    first_derivative = (y[2] - y[0]) / (2 * h)
    second_derivative = (y[2] - 2 * y[1] + y[0]) / (h ** 2)
    third_derivative = (y[3] - 3 * y[2] + 3 * y[1] - y[0]) / (h ** 3)
    return first_derivative, second_derivative, third_derivative

f1, f2, f3 = central_differences(y3[:4], h3)

# 4. First and Second Derivatives at x = 1.1
x4 = np.array([1.0, 1.2, 1.4, 1.6, 1.8, 2.0])
y4 = np.array([0, 0.128, 0.544, 1.296, 2.432, 4.000])
h4 = 0.2

f_prime_1_1 = (y4[1] - y4[0]) / h4
f_double_prime_1_1 = (y4[2] - 2 * y4[1] + y4[0]) / (h4 ** 2)

# 5. Derivatives at specific points
x5 = np.array([1.00, 1.05, 1.10, 1.15, 1.20, 1.25, 1.30])
y5 = np.array([1.000, 1.025, 1.049, 1.072, 1.095, 1.118, 1.140])
h5 = 0.05

def central_diff(y, h):
    first_deriv = (y[2] - y[0]) / (2 * h)
    second_deriv = (y[2] - 2 * y[1] + y[0]) / (h ** 2)
    return first_deriv, second_deriv

# At x = 1.05
dy_dx_1_05, d2y_dx2_1_05 = central_diff(y5[:3], h5)
# At x = 1.25
dy_dx_1_25, d2y_dx2_1_25 = central_diff(y5[4:], h5)
# At x = 1.15
dy_dx_1_15, d2y_dx2_1_15 = central_diff(y5[2:5], h5)

# Output results
print("1. Initial acceleration at t=0:", acceleration_0)
print("2. f'(10):", fx_prime_10)
print("3. Derivatives at x=1.5:", f1, f2, f3)
print("4. Derivatives at x=1.1:", f_prime_1_1, f_double_prime_1_1)
print("5. Derivatives at specific points:")
print("   At x=1.05:", dy_dx_1_05, d2y_dx2_1_05)
print("   At x=1.25:", dy_dx_1_25, d2y_dx2_1_25)
print("   At x=1.15:", dy_dx_1_15, d2y_dx2_1_15)
