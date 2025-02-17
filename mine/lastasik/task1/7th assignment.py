import numpy as np
from scipy.interpolate import lagrange

def forward_difference(f_values, index, h):
    return (f_values[index + 1] - f_values[index]) / h

def backward_difference(f_values, index, h):
    return (f_values[index] - f_values[index - 1]) / h

def central_difference(f_values, index, h):
    return (f_values[index + 1] - f_values[index - 1]) / (2 * h)

def second_derivative(f_values, index, h):
    return (f_values[index + 1] - 2 * f_values[index] + f_values[index - 1]) / (h ** 2)

# Задача 1
velocities = [0, 3, 14, 69, 228]
times = [0, 5, 10, 15, 20]
initial_acceleration = forward_difference(velocities, 0, 5)
print("-------------------------\nTask 1:")
print("Initial Acceleration:", initial_acceleration)

# Задача 2
x_vals = [3, 5, 11, 27, 34]
f_vals = [-13, 23, 899, 17315, 35606]
index = 2  # x = 11
h_forward = x_vals[index + 1] - x_vals[index]
h_backward = x_vals[index] - x_vals[index - 1]

poly = lagrange(x_vals[:4], f_vals[:4])
f_prime_10_interp = np.polyder(poly)(10)

f_prime_10 = (forward_difference(f_vals, index, h_forward) + backward_difference(f_vals, index, h_backward)) / 2
print("-------------------------\nTask 2:")
print("f'(10) (без интерполяции):", f_prime_10)
print("f'(10) (с интерполяцией Лагранжа):", f_prime_10_interp)

# Задача 3
x_vals3 = [1.5, 2.0, 2.5, 3.0, 3.5, 4.0]
f_vals3 = [3.375, 7.000, 13.625, 24.000, 38.875, 59.000]
h3 = x_vals3[1] - x_vals3[0]

f_prime_1_5 = forward_difference(f_vals3, 0, h3)
f_double_prime_1_5 = second_derivative(f_vals3, 1, h3)
f_triple_prime_1_5 = (f_vals3[3] - 3*f_vals3[2] + 3*f_vals3[1] - f_vals3[0]) / (h3 ** 3)

print("-------------------------\nTask 3:")
print("f'(1.5):", f_prime_1_5)
print("f''(1.5):", f_double_prime_1_5)
print("f'''(1.5):", f_triple_prime_1_5)

# Задача 4
x_vals2 = [1.0, 1.2, 1.4, 1.6, 1.8, 2.0]
f_vals2 = [0, 0.128, 0.544, 1.296, 2.432, 4.00]
index2 = 1  # для x = 1.2 (ближайшая к 1.1)
h2 = x_vals2[1] - x_vals2[0]

f_prime_1_1 = central_difference(f_vals2, index2, h2)
f_double_prime_1_1 = second_derivative(f_vals2, index2, h2)

f_prime_interp_1_1 = np.interp(1.1, [1.0, 1.2], [forward_difference(f_vals2, 0, h2), forward_difference(f_vals2, 1, h2)])

print("-------------------------\nTask 4:")
print("f'(1.1) (без интерполяции):", f_prime_1_1)
print("f'(1.1) (с линейной интерполяцией):", f_prime_interp_1_1)
print("f''(1.1):", f_double_prime_1_1)

# Задача 5
x_vals5 = [1.00, 1.05, 1.10, 1.15, 1.20, 1.25, 1.30]
y_vals5 = [1.000, 1.025, 1.049, 1.072, 1.095, 1.118, 1.140]
h5 = x_vals5[1] - x_vals5[0]

dy_dx_1_05 = forward_difference(y_vals5, 1, h5)
d2y_dx2_1_05 = second_derivative(y_vals5, 1, h5)


dy_dx_1_25 = backward_difference(y_vals5, 5, h5)
d2y_dx2_1_25 = second_derivative(y_vals5, 5, h5)


dy_dx_1_15 = central_difference(y_vals5, 3, h5)
d2y_dx2_1_15 = second_derivative(y_vals5, 3, h5)

print("-------------------------\nTask 5:")
print("At x = 1.05:")
print("dy/dx:", dy_dx_1_05)
print("d^2y/dx^2:", d2y_dx2_1_05)

print("At x = 1.25:")
print("dy/dx:", dy_dx_1_25)

print("d^2y/dx^2:", d2y_dx2_1_25)

print("At x = 1.15:")
print("dy/dx:", dy_dx_1_15)
print("d^2y/dx^2:", d2y_dx2_1_15)
