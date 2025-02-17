import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit


def linear_func(x, a, b):
    return a * x + b


def quadratic_func(x, a, b, c):
    return a * x ** 2 + b * x + c


def exponential_func(x, a, b):
    return a * np.exp(b * x)


def rational_func(x, a, b):
    return a * x + b / x


def fit_and_find_best_model(x, y):
    models = [
        (linear_func, "Linear", [1, 1]),
        (quadratic_func, "Quadratic", [1, 1, 1]),
        (exponential_func, "Exponential", [1, 0.1]),
        (rational_func, "Rational", [1, 1]),
    ]

    min_error = float('inf')
    best_model = None
    best_coeffs = None
    best_fit = None

    for func, name, p0 in models:
        coeffs, _ = curve_fit(func, x, y, p0=p0)
        y_fit = func(x, *coeffs)
        error = np.sum((y - y_fit) ** 2)

        if error < min_error:
            min_error = error
            best_model = name
            best_coeffs = coeffs
            best_fit = y_fit

    return best_model, best_coeffs, best_fit, min_error


x = np.array([0, 1, 2, 4])
y = np.array([1.05, 3.10, 3.85, 8.30])

best_model, best_coeffs, best_fit, min_error = fit_and_find_best_model(x, y)

print(f"Лучшая модель: {best_model}")
print(f"Коэффициенты: {best_coeffs}")
print(f"Минимальная ошибка: {min_error}")

plt.scatter(x, y, color='red', label='Исходные данные')
plt.plot(x, best_fit, label=f'Лучшая подгонка: {best_model}')
plt.legend()
plt.title('Подгонка с минимальной ошибкой')
plt.show()
