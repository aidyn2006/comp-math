import numpy as np
import matplotlib.pyplot as plt

x = np.array([1, 2, 3, 4, 5])
y = np.array([1.8, 5.1, 8.9, 14.1, 19.8])

n = len(x)
sum_x = np.sum(x)
sum_x2 = np.sum(x**2)
sum_x3 = np.sum(x**3)
sum_x4 = np.sum(x**4)
sum_y = np.sum(y)
sum_xy = np.sum(x * y)
sum_x2y = np.sum(x**2 * y)

A = np.array([
    [sum_x2, sum_x3],
    [sum_x3, sum_x4]
])
B = np.array([sum_xy, sum_x2y])

coefficients = np.linalg.solve(A, B)
a, b = coefficients

print(f"Коэффициенты: a = {a:.4f}, b = {b:.4f}")

def quadratic_fit(x):
    return a * x + b * x**2

x_fit = np.linspace(min(x), max(x), 500)
y_fit = quadratic_fit(x_fit)

plt.scatter(x, y, color='red', label='Исходные данные')
plt.plot(x_fit, y_fit, color='blue', label=f'Функция: y = {a:.2f}x + {b:.2f}x²')
plt.title('Аппроксимация методом наименьших квадратов')
plt.xlabel('X')
plt.ylabel('Y')
plt.legend()
plt.grid()
plt.show()