import numpy as np
import pandas as pd

def forward_difference_table(x, y):
    n = len(x)
    table = np.zeros((n, n))
    table[:, 0] = [val if val is not None else 0 for val in y]

    for j in range(1, n):
        for i in range(n - j):
            table[i][j] = table[i + 1][j - 1] - table[i][j - 1]

    return table

def lagrange_interpolation(x, y, value):
    n = len(x)
    result = 0
    for i in range(n):
        term = y[i]
        for j in range(n):
            if j != i:
                term *= (value - x[j]) / (x[i] - x[j])
        result += term
    return result

def problem_1():
    x = [10, 20, 30, 40]
    y = [1.1, 2.0, 4.4, 7.9]

    forward_table = forward_difference_table(x, y)
    forward_df = pd.DataFrame(forward_table, columns=[f"Δ^{i}f" for i in range(len(x))])
    forward_df.insert(0, "x", x)

    print("Задача 1: Таблица прямых разностей")
    print(forward_df.fillna(''))

def problem_2():
    x = [0, 1, 2, 3, 4]
    y = [1.0, 1.5, 2.2, 3.1, 4.6]

    forward_table = forward_difference_table(x, y)
    forward_df = pd.DataFrame(forward_table, columns=[f"Δ^{i}f" for i in range(len(x))])
    forward_df.insert(0, "x", x)

    print("\nЗадача 2: Таблица разностей")
    print(forward_df.fillna(''))

    delta_f_2 = forward_table[2][1]
    print(f"Δf(2) = {delta_f_2}")

def problem_3():
    def y_function(x):
        return x**3 + x**2 - 2*x + 1

    x = [0, 1, 2, 3, 4, 5]
    y = [y_function(xi) for xi in x]

    x_extended = x + [6]
    y_extended = y + [y_function(6)]

    forward_table = forward_difference_table(x_extended, y_extended)
    forward_df = pd.DataFrame(forward_table, columns=[f"Δ^{i}f" for i in range(len(x_extended))])
    forward_df.insert(0, "x", x_extended)

    print("\nЗадача 3: Таблица разностей с расширением до x = 6")
    print(forward_df.fillna(''))

    y_6 = y_function(6)
    print(f"y(6) из таблицы: {forward_table[0][-1]}")
    print(f"y(6) по подстановке: {y_6}")

def problem_4():
    def f_function(x):
        return x**3 + 5*x - 7

    x = [-1, 0, 1, 2, 3, 4, 5]
    y = [f_function(xi) for xi in x]

    x_extended = x + [6]
    y_extended = y + [f_function(6)]

    forward_table = forward_difference_table(x_extended, y_extended)
    forward_df = pd.DataFrame(forward_table, columns=[f"Δ^{i}f" for i in range(len(x_extended))])
    forward_df.insert(0, "x", x_extended)

    print("\nЗадача 4: Таблица разностей с расширением до x = 6")
    print(forward_df.fillna(''))

    f_6 = f_function(6)
    print(f"f(6) из таблицы: {forward_table[0][-1]}")
    print(f"f(6) по подстановке: {f_6}")

def problem_5():
    x = [-0.2, 0.0, 0.2, 0.4, 0.6, 0.8, 1.0]
    y = [2.6, 3.0, 3.4, 4.28, 7.08, 14.2, 29.0]

    x_extended = [x[0] - 0.2, x[0] - 0.1] + x + [x[-1] + 0.1, x[-1] + 0.2]
    y_extended = [lagrange_interpolation(x, y, xi) for xi in x_extended]

    forward_table = forward_difference_table(x_extended, y_extended)
    forward_df = pd.DataFrame(forward_table, columns=[f"Δ^{i}f" for i in range(len(x_extended))])
    forward_df.insert(0, "x", x_extended)

    print("\nЗадача 5: Расширенная таблица разностей")
    print(forward_df.fillna(''))

def problem_6():
    x = [10, 20, 30, 40]
    y = [1.1, None, 4.266666666666667, 7.9]

    missing_index = y.index(None)
    missing_x = x[missing_index]
    missing_y = lagrange_interpolation(x[:missing_index] + x[missing_index+1:],
                                       y[:missing_index] + y[missing_index+1:],
                                       missing_x)

    y[missing_index] = missing_y

    print(f"\nЗадача 6: Пропущенное значение y({missing_x}) = {missing_y}")

def problem_7():
    def f_function(x):
        return (2*x +1) * (2*x + 3) * (2*x + 5) * (2*x + 7) * (2*x + 9) * (2*x + 11) * (2*x + 13) * (2*x + 15)

    x = [0, 1, 2, 3, 4, 5]
    y = [f_function(xi) for xi in x]

    forward_table = forward_difference_table(x, y)
    delta_4_f = forward_table[0][4]

    print(f"\nЗадача 7: Δ^4 f(x) = {delta_4_f}")

def problem_8():
    x = [0, 1, 2, 3, 4, 5, 6, 7]
    y = [1, -1, 1, -1, 1, None, None, None]

    # Вычисляем следующие три значения
    forward_table = forward_difference_table(x, y)
    for i in range(4, len(x)):
        y[i] = y[i-1] + forward_table[i-1][1]

    print("\nЗадача 8: Следующие три значения:")
    print(y[5:])

def problem_9():
    def f_function(x):
        return (2*x + 1) * (2*x + 3) * (2*x + 5) * (2*x + 7) * (2*x + 9) * (2*x + 11) * (2*x + 13) * (2*x + 15) * (2*x + 17) * (2*x + 19)

    x = [0, 1, 2, 3, 4, 5]
    y = [f_function(xi) for xi in x]

    forward_table = forward_difference_table(x, y)
    delta_4_f = forward_table[0][4]

    print(f"\nЗадача 9: Δ^4 f(x) = {delta_4_f}")

def problem_10():
    x = [2, 3, 4, 5, 6]
    y = [45.0, 49.2, 54.1, None, 67.4]

    missing_index = y.index(None)
    missing_x = x[missing_index]
    missing_y = lagrange_interpolation(x[:missing_index] + x[missing_index+1:],
                                       y[:missing_index] + y[missing_index+1:],
                                       missing_x)

    y[missing_index] = missing_y

    print(f"\nЗадача 10: Пропущенное значение y({missing_x}) = {missing_y}")

if __name__ == "__main__":
    problem_1()
    problem_2()
    problem_3()
    problem_4()
    problem_5()
    problem_6()
    problem_7()
    problem_8()
    problem_9()
    problem_10()
