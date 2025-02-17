import numpy as np
from scipy.interpolate import lagrange
import sympy as sp

def forward_difference(f_values, index, h):
    return (-3 * f_values[index] + 4 * f_values[index + 1] - f_values[index + 2]) / (2 * h)


def backward_difference(f_values, index, h):
    return (3 * f_values[index] - 4 * f_values[index - 1] + f_values[index - 2]) / (2 * h)


def central_difference(f_values, index, h):
    return (f_values[index + 1] - f_values[index - 1]) / (2 * h)


def second_derivative(f_values, index, h):
    return (f_values[index - 1] - 2 * f_values[index] + f_values[index + 1]) / (h ** 2)


# Задача 1
times = np.array([0, 5, 10, 15, 20], dtype=float)
velocities = np.array([0, 3, 14, 69, 228], dtype=float)
h1 = 5.0
initial_acceleration = (-25 * velocities[0] + 48 * velocities[1] - 36 * velocities[2] + 16 * velocities[3] - 3 * velocities[4]) / (12 * h1)

print("-------------------------\nTask 1:")
print("Initial Acceleration:", initial_acceleration)

# Задача 2
x_vals = np.array([3, 5, 11, 27, 34], dtype=float)
f_vals = np.array([-13, 23, 899, 17315, 35606], dtype=float)

# Интерполяция Лагранжа для производной в x = 10
# Интерполяция Лагранжа для производной в x = 10
lagrange_poly = lagrange(x_vals, f_vals)

# Определение символьной переменной
x = sp.symbols('x')

# Преобразование полинома в символьное выражение
lagrange_poly_sym = sum(coef * x**i for i, coef in enumerate(lagrange_poly.coefficients[::-1]))

# Нахождение производной
poly_prime = sp.diff(lagrange_poly_sym, x)
f_prime_10_interp = poly_prime.subs(x, 10)

print("-------------------------\nTask 2:")
print("f'(10) (с интерполяцией Лагранжа):", f_prime_10_interp)

# Задача 3
x_vals3 = np.array([1.5, 2.0, 2.5, 3.0, 3.5, 4.0], dtype=float)
f_vals3 = np.array([3.375, 7.000, 13.625, 24.000, 38.875, 59.000], dtype=float)
h3 = 0.5

# Разности
delta1 = np.diff(f_vals3)
delta2 = np.diff(delta1)
delta3 = np.diff(delta2)

f_prime_1_5 = (1 / h3) * (delta1[0] - 0.5 * delta2[0] + (1/3) * delta3[0])
f_double_prime_1_5 = (1 / h3**2) * (delta2[0] - delta3[0])
f_triple_prime_1_5 = (1 / h3**3) * delta3[0]

print("-------------------------\nTask 3:")
print("f'(1.5):", f_prime_1_5)
print("f''(1.5):", f_double_prime_1_5)
print("f'''(1.5):", f_triple_prime_1_5)

# Задача 4
x_vals4 = np.array([1.0, 1.2, 1.4, 1.6, 1.8, 2.0], dtype=float)
f_vals4 = np.array([0, 0.128, 0.544, 1.296, 2.432, 4.00], dtype=float)
h4 = 0.2

u = (1.1 - 1.0) / h4

delta1_4 = np.diff(f_vals4)
delta2_4 = np.diff(delta1_4)
delta3_4 = np.diff(delta2_4)

f_prime_1_1 = (1 / h4) * (delta1_4[0] + ((2*u - 1)/2) * delta2_4[0] + ((3*u**2 - 6*u + 2)/6) * delta3_4[0])
f_double_prime_1_1 = (1 / h4**2) * (delta2_4[0] + (u - 1) * delta3_4[0])

print("-------------------------\nTask 4:")
print("f'(1.1):", f_prime_1_1)
print("f''(1.1):", f_double_prime_1_1)

# Задача 5
x_vals5 = np.array([1.00, 1.05, 1.10, 1.15, 1.20, 1.25, 1.30], dtype=float)
y_vals5 = np.array([1.000, 1.025, 1.049, 1.072, 1.095, 1.118, 1.140], dtype=float)
h5 = 0.05

# Для x = 1.05 (вперед)
dy_dx_1_05 = (-3 * y_vals5[0] + 4 * y_vals5[1] - y_vals5[2]) / (2 * h5)
d2y_dx2_1_05 = (y_vals5[0] - 2 * y_vals5[1] + y_vals5[2]) / (h5 ** 2)

# Для x = 1.25 (назад)
dy_dx_1_25 = (3 * y_vals5[5] - 4 * y_vals5[4] + y_vals5[3]) / (2 * h5)
d2y_dx2_1_25 = (y_vals5[4] - 2 * y_vals5[5] + y_vals5[6]) / (h5 ** 2)

# Для x = 1.15 (центральные разности)
dy_dx_1_15 = (y_vals5[4] - y_vals5[2]) / (2 * h5)
d2y_dx2_1_15 = (y_vals5[2] - 2 * y_vals5[3] + y_vals5[4]) / (h5 ** 2)

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