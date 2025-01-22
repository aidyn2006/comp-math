import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

import math
x1 = np.array([0, 1, 2, 3, 4])
y1 = np.array([1, 1.8, 1.3, 2, 6.3])

# y = ax^2 + bx + c
coeffs1 = np.polyfit(x1, y1, 2)

poly1 = np.poly1d(coeffs1)
y1_fit = poly1(x1)

residuals = y1 - y1_fit
rss = math.sqrt(np.sum(residuals**2))

print("Residuals:", residuals)
print("RSS:", rss)


plt.figure()
plt.scatter(x1, y1, color='red', label='Data points')
plt.plot(x1, y1_fit, label=f'Fitted parabola: {coeffs1[0]:.2f}x^2 + {coeffs1[1]:.2f}x + {coeffs1[2]:.2f}')
plt.legend()
plt.title('Second-degree Parabola Fit')
plt.show()

print(f"Second-degree parabola coefficients: {coeffs1}")

x2 = np.array([6, 7, 8, 8.5, 9, 10])
y2 = np.array([5, 5, 4, 4.5, 4, 3.3])

# y = mx + c
coeffs2 = np.polyfit(x2, y2, 1)
poly2 = np.poly1d(coeffs2)
y2_fit = poly2(x2)

plt.figure()
plt.scatter(x2, y2, color='red', label='Data points')
plt.plot(x2, y2_fit, label=f'Fitted line: {coeffs2[0]:.2f}x + {coeffs2[1]:.2f}')
plt.legend()
plt.title('Straight Line Fit')
plt.show()

print(f"Straight line coefficients: {coeffs2}")

x3 = np.array([0, 1, 2, 3])
y3 = np.array([1.05, 2.10, 3.85, 8.30])

# y = ae^(bx)
def exp_func(x, a, b):
    return a * np.exp(b * x)

coeffs3, _ = curve_fit(exp_func, x3, y3)
y3_fit = exp_func(x3, *coeffs3)

print(curve_fit(exp_func, x3, y3))

plt.figure()
plt.scatter(x3, y3, color='red', label='Data points')
plt.plot(x3, y3_fit, label=f'Fitted curve: {coeffs3[0]:.2f}e^({coeffs3[1]:.2f}x)')
plt.legend()
plt.title('Exponential Curve Fit')
plt.show()

print(f"Exponential curve coefficients: {coeffs3}")

x4 = np.array([1, 2, 3, 4, 5])
y4 = np.array([1.8, 5.1, 8.9, 14.1, 19.8])

# y = ax + bx^2
def quad_func(x, a, b):
    return a * x + b * x**2

coeffs4, _ = curve_fit(quad_func, x4, y4)
y4_fit = quad_func(x4, *coeffs4)

plt.figure()
plt.scatter(x4, y4, color='red', label='Data points')
plt.plot(x4, y4_fit, label=f'Fitted curve: {coeffs4[0]:.2f}x + {coeffs4[1]:.2f}x^2')
plt.legend()
plt.title('Quadratic Curve Fit')
plt.show()

print(f"Quadratic curve coefficients: {coeffs4}")

x5 = np.array([1, 2, 3, 4, 5, 6, 7, 8])
y5 = np.array([5.4, 6.3, 8.2, 10.3, 12.6, 14.9, 17.3, 19.5])

# y = ax + b/x
def rational_func(x, a, b):
    return a * x + b / x

coeffs5, _ = curve_fit(rational_func, x5, y5)
y5_fit = rational_func(x5, *coeffs5)

plt.figure()
plt.scatter(x5, y5, color='red', label='Data points')
plt.plot(x5, y5_fit, label=f'Fitted curve: {coeffs5[0]:.2f}x + {coeffs5[1]:.2f}/x')
plt.legend()
plt.title('Rational Curve Fit')
plt.show()

print(f"Rational curve coefficients: {coeffs5}")