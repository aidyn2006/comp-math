import numpy as np
import scipy.integrate as spi


def f(x):
    return np.sin(x)

def trapezoidal_rule(a, b, n):
    h = (b - a) / n
    sum_terms = 0

    for i in range(1, n):
        sum_terms += f(a + i * h)

    I = (h / 2) * (f(a) + 2 * sum_terms + f(b))
    return I

def simpsons_one_third_rule(a, b, n):
    if n % 2 == 1:
        n += 1
    h = (b - a) / n
    result = f(a) + f(b)
    for i in range(1, n, 2):
        result += 4 * f(a + i * h)
    for i in range(2, n-1, 2):
        result += 2 * f(a + i * h)
    return result * h / 3

def simpsons_three_eighth_rule(a, b, n):
    if n % 3 != 0:
        n += 3 - (n % 3)
    h = (b - a) / n
    result = f(a) + f(b)
    for i in range(1, n, 3):
        result += 3 * f(a + i * h) + 3 * f(a + (i + 1) * h)
    for i in range(3, n-2, 3):
        result += 2 * f(a + i * h)
    return result * 3 * h / 8

a = 0
b = np.pi
n = 6

trapezoidal_result = trapezoidal_rule(a, b, n)
simpsons_1_3_result = simpsons_one_third_rule(a, b, n)
simpsons_3_8_result = simpsons_three_eighth_rule(a, b, n)
analytic_result = spi.quad(f, a, b)

print(f"Trapezional: {trapezoidal_result}")
print(f"Simpson's 1/3: {simpsons_1_3_result}")
print(f"Simpson's 3/8: {simpsons_3_8_result}")
print(f"Analytical : {analytic_result[1]}")