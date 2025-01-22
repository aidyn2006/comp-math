import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
import math




x1 = np.array([0, 1, 2, 3, 4])
y1 = np.array([1, 1.8, 1.3, 2, 6.3])


coeffs1 = np.polyfit(x1, y1, 2)
poly1 = np.poly1d(coeffs1)
y1_fit_poly = poly1(x1)

errors1 = y1 - y1_fit_poly

plt.figure()
plt.scatter(x1, y1, color='red', label='Data points')
plt.plot(x1, y1_fit_poly, label=f'Fitted parabola: {coeffs1[0]:.2f}x^2 + {coeffs1[1]:.2f}x + {coeffs1[2]:.2f}')
plt.vlines(x1, y1, y1_fit_poly, color='black', linestyle='dotted', label='Errors')
plt.legend()
plt.title('Second-degree Parabola Fit with Errors')
plt.show()


print(f"Errors for polynomial fit: {errors1}")

x2 = np.array([6, 7, 8, 8.5, 9, 10])
y2 = np.array([5, 5, 4, 4.5, 4, 3.3])

coeffs2 = np.polyfit(x2, y2, 1)
poly2 = np.poly1d(coeffs2)
y2_fit = poly2(x2)

errors2 = y2 - y2_fit

plt.figure()
plt.scatter(x2, y2, color='red', label='Data points')
plt.plot(x2, y2_fit, label=f'Fitted line: {coeffs2[0]:.2f}x + {coeffs2[1]:.2f}')
plt.vlines(x2, y2, y2_fit, color='black', linestyle='dotted', label='Errors')
plt.legend()
plt.title('Straight Line Fit with Errors')
plt.show()

x3 = np.array([0, 1, 2, 3])
y3 = np.array([1.05, 2.10, 3.85, 8.30])


def exp_func(x, a, b):
    return a * np.exp(b * x)


coeffs3_exp, _ = curve_fit(exp_func, x3, y3)
y3_fit = exp_func(x3, *coeffs3_exp)

errors3 = y3 - y3_fit

plt.figure()
plt.scatter(x3, y3, color='red', label='Data points')
plt.plot(x3, y3_fit, label=f'Fitted curve: {coeffs3_exp[0]:.2f}e^({coeffs3_exp[1]:.2f}x)')
plt.vlines(x3, y3, y3_fit, color='black', linestyle='dotted', label='Errors')
plt.legend()
plt.title('Exponential Curve Fit with Errors')
plt.show()

print(f"Exponential curve coefficients: {coeffs3_exp}")
print(f"Errors: {errors3}")

x4 = np.array([1, 2, 3, 4, 5])
y4 = np.array([1.8, 5.1, 8.9, 14.1, 19.8])


def quad_func(x, a, b):
    return a * x + b * x ** 2


coeffs4, _ = curve_fit(quad_func, x4, y4)
y4_fit = quad_func(x4, *coeffs4)

errors4 = y4 - y4_fit

plt.figure()
plt.scatter(x4, y4, color='red', label='Data points')
plt.plot(x4, y4_fit, label=f'Fitted curve: {coeffs4[0]:.2f}x + {coeffs4[1]:.2f}x^2')
plt.vlines(x4, y4, y4_fit, color='black', linestyle='dotted', label='Errors')
plt.legend()
plt.title('Quadratic Curve Fit with Errors')
plt.show()

print(f"Quadratic curve coefficients: {coeffs4}")
print(f"Errors: {errors4}")

x5 = np.array([1, 2, 3, 4, 5, 6, 7, 8])
y5 = np.array([5.4, 6.3, 8.2, 10.3, 12.6, 14.9, 17.3, 19.5])


def rational_func(x, a, b):
    return a * x + b / x


coeffs5, _ = curve_fit(rational_func, x5, y5)
y5_fit = rational_func(x5, *coeffs5)

errors5 = y5 - y5_fit

plt.figure()
plt.scatter(x5, y5, color='red', label='Data points')
plt.plot(x5, y5_fit, label=f'Fitted curve: {coeffs5[0]:.2f}x + {coeffs5[1]:.2f}/x')
plt.vlines(x5, y5, y5_fit, color='black', linestyle='dotted', label='Errors')
plt.legend()
plt.title('Rational Curve Fit with Errors')
plt.show()

print(f"Rational curve coefficients: {coeffs5}")
print(f"Errors: {errors5}")











# plt.figure()
# plt.scatter(x1_avg, y1_avg, color='blue', label='Averaged Data points')
# plt.plot(x1_avg, y1_fit,
#          label=f'Fitted curve with 3 constants: {coeffs3[0]:.2f}e^({coeffs3[1]:.2f}x) + {coeffs3[2]:.2f}')
# plt.legend()
# plt.title('Three Constants Law Fit')
# plt.show()
#
# print(f"Three constants law coefficients: {coeffs3}")
# print(f"Method of Group Averages:\n Averaged x: {x1_avg}\n Averaged y: {y1_avg}")
# initial_guess = [1, 0.5, 0]
#
# coeffs3, _ = curve_fit(three_constants_func, x1_avg, y1_avg, p0=initial_guess, maxfev=100000)
# y1_fit = three_constants_func(x1_avg, *coeffs3)

# def method_of_group_averages(x, y, num_groups):
#     min_x, max_x = min(x), max(x)
#     group_edges = np.linspace(min_x, max_x, num_groups + 1)
#
#     avg_x = []
#     avg_y = []
#
#     for i in range(num_groups):
#         group_mask = (x >= group_edges[i]) & (x < group_edges[i + 1])
#         avg_x.append(np.mean(x[group_mask]))
#         avg_y.append(np.mean(y[group_mask]))
#
#     return np.array(avg_x), np.array(avg_y)
# x1_avg, y1_avg = method_of_group_averages(x1, y1, num_groups=3)
#
#
# def three_constants_func(x, a, b, c):
#     return a * np.exp(b * x) + c
