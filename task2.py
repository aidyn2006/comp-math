import numpy as np
import matplotlib
import math

#matplotlib.use('TkAgg')
import matplotlib.pyplot as plt

# Исходная функция
func1 = lambda x: x ** 3 - 3*x - 5

# Производная функции для метода Ньютона-Рафсона
derivative_func1 = lambda x: 3 * x ** 2 - 1


# Метод бисекции
def bisection_method(func, a, b, tol=1e-6, max_iter=100):
    x_vals = []
    for _ in range(max_iter):
        c = (a + b) / 2
        x_vals.append(c)
        if abs(func(c)) < tol or (b - a) / 2 < tol:
            break
        if func(a) * func(c) < 0:
            b = c
        else:
            a = c
    return c, x_vals


# Метод ложного положения
def false_position_method(func, a, b, tol=1e-6, max_iter=100):
    x_vals = []
    for _ in range(max_iter):
        c = b - (func(b) * (b - a)) / (func(b) - func(a))
        x_vals.append(c)
        if abs(func(c)) < tol:
            break
        if func(a) * func(c) < 0:
            b = c
        else:
            a = c
    # print(c,x_vals)
    return c, x_vals


# Метод простой итерации
def iteration_method(func, x0, g, tol=1e-6, max_iter=100):
    x_vals = [x0]
    for _ in range(max_iter):
        x1 = g(x_vals[-1])
        x_vals.append(x1)
        if abs(x1 - x_vals[-2]) < tol:
            break
    print(x1,x_vals)
    return x1, x_vals


# Метод Ньютона-Рафсона
def newton_raphson_method(func, derivative, x0, tol=1e-6, max_iter=100):
    x_vals = [x0]
    for _ in range(max_iter):
        x1 = x_vals[-1] - func(x_vals[-1]) / derivative(x_vals[-1])
        x_vals.append(x1)
        if abs(x1 - x_vals[-2]) < tol:
            break
    print(x1)
    return x1, x_vals


# Главная программа
if __name__ == "__main__":
    a, b = 2, 6  # Интервал
    x0 = 2  # Начальное приближение для итерационных методов

    # Функция g(x) для метода итерации (на основе преобразования исходного уравнения)
    g = lambda x: (x + 1) ** (1 / 3)

    # Решение уравнения различными методами
    root_bisection, bisection_vals = bisection_method(func1, a, b)
    root_false_position, false_position_vals = false_position_method(func1, a, b)
    root_iteration, iteration_vals = iteration_method(func1, x0, g)
    root_newton, newton_vals = newton_raphson_method(func1, derivative_func1, x0)

    # Построение графиков
    plt.figure(figsize=(10, 6))
    plt.plot(range(len(bisection_vals)), bisection_vals, label="Bisection Method", marker='o')
    plt.plot(range(len(false_position_vals)), false_position_vals, label="False Position Method", marker='x')
    plt.plot(range(len(iteration_vals)), iteration_vals, label="Iteration Method", marker='^')
    plt.plot(range(len(newton_vals)), newton_vals, label="Newton-Raphson Method", marker='s')

    plt.xlabel("Iteration")
    plt.ylabel("x Value")
    plt.title("Convergence of Root-Finding Methods")
    plt.legend()
    plt.grid()
    plt.show()
