import operator
from functools import reduce
from typing import Dict
import numpy as np


def calculate_forward_differences(x: np.ndarray, y: np.ndarray) -> Dict:
    """
    Calculate forward differences table for given x and y values.
    """
    n = len(x)
    if len(y) != n:
        raise ValueError("x and y must have the same length")

    diff_table = np.zeros((n, n))
    diff_table[:, 0] = y

    for j in range(1, n):
        for i in range(n - j):
            diff_table[i, j] = diff_table[i + 1, j - 1] - diff_table[i, j - 1]

    return {
        'diff_table': diff_table,
        'x': x,
        'y': y
    }


def newton_forward_interpolation(x: float, data: Dict) -> float:
    """
    Compute interpolated value using Newton's Forward Interpolation.
    """
    x_data = data['x']
    diff_table = data['diff_table']
    h = x_data[1] - x_data[0]
    u = (x - x_data[0]) / h

    result = diff_table[0, 0]
    u_term = 1

    for i in range(1, len(x_data)):
        u_term *= (u - i + 1) / i
        result += u_term * diff_table[0, i]

    return result


def calculate_backward_differences(x: np.ndarray, y: np.ndarray) -> Dict:
    """
    Calculate backward differences table for given x and y values.
    """
    n = len(x)
    if len(y) != n:
        raise ValueError("x and y must have the same length")

    diff_table = np.zeros((n, n))
    diff_table[:, 0] = y

    for j in range(1, n):
        for i in range(n - 1, j - 1, -1):
            diff_table[i, j] = diff_table[i, j - 1] - diff_table[i - 1, j - 1]

    return {
        'diff_table': diff_table,
        'x': x,
        'y': y
    }


def newton_backward_interpolation(x: float, data: Dict) -> float:
    """
    Compute interpolated value using Newton's Backward Interpolation.
    """
    x_data = data['x']
    diff_table = data['diff_table']
    n = len(x_data)
    h = x_data[1] - x_data[0]
    u = (x - x_data[n - 1]) / h

    result = diff_table[n - 1, 0]
    u_term = 1

    for i in range(1, n):
        u_term *= (u + i - 1) / i
        result += u_term * diff_table[n - i - 1, i]

    return result


def central_difference_interpolation(x: float, x_data: np.ndarray, y_data: np.ndarray) -> float:
    """
    Compute interpolated value using Central Difference Interpolation.
    """
    h = x_data[1] - x_data[0]

    k = len(x_data) // 2

    delta_yk = (y_data[k + 1] - y_data[k - 1]) / 2

    result = y_data[k] + ((x - x_data[k]) / h) * delta_yk
    return result


def lagrange_interpolation(x: float, x_data: np.ndarray, y_data: np.ndarray) -> float:
    """
    Compute interpolated value using Lagrange Interpolation.
    """
    n = len(x_data)
    result = 0.0

    for i in range(n):
        term = y_data[i]
        for j in range(n):
            if j != i:
                term *= (x - x_data[j]) / (x_data[i] - x_data[j])
        result += term

    return result


def calculate_forward_differences(x: np.ndarray, y: np.ndarray) -> Dict:
    """
    Calculate forward differences table for given x and y values.
    Returns both the difference matrix and formatted table.
    """
    n = len(x)
    if len(y) != n:
        raise ValueError("x and y must have the same length")

    diff_table = np.zeros((n, n))
    diff_table[:, 0] = y

    for j in range(1, n):
        for i in range(n - j):
            diff_table[i, j] = diff_table[i + 1, j - 1] - diff_table[i, j - 1]

    table = []
    for i in range(n):
        row = {
            'x': x[i],
            'f': y[i]
        }
        for j in range(1, n - i):
            row[f'Δ{j}f'] = diff_table[i, j]
        table.append(row)

    return {
        'table': table,
        'diff_matrix': diff_table
    }


def print_difference_table(result: Dict, method: str = 'forward') -> None:
    """Print the difference table in a formatted way."""
    table = result['table']
    symbol = 'Δ' if method == 'forward' else '∇'

    max_diff = max(len(row) - 2 for row in table)
    headers = ['x', 'f'] + [f'{symbol}{i}f' for i in range(1, max_diff + 1)]
    header_str = '|'.join(f'{h:^12}' for h in headers)
    print('-' * len(header_str))
    print(header_str)
    print('-' * len(header_str))

    for row in table:
        row_values = []
        row_values.append(f"{row['x']:^12.4f}")
        row_values.append(f"{row['f']:^12.4f}")

        for i in range(1, max_diff + 1):
            key = f'{symbol}{i}f'
            if key in row:
                row_values.append(f"{row[key]:^12.4f}")
            else:
                row_values.append(' ' * 12)

        print('|'.join(row_values))
    print('-' * len(header_str))


def product_function(x: float, terms: range) -> float:
    """Calculate product function of form (2x + 1)(2x + 3)...(2x + n)"""
    return reduce(operator.mul, ((2 * x + i) for i in terms))


def newton_forward_interpolation(x: float, data: Dict) -> float:
    """Compute interpolated value using Newton's Forward Interpolation."""
    x_data = np.array([row['x'] for row in data['table']])
    diff_table = data['diff_matrix']
    h = x_data[1] - x_data[0]
    u = (x - x_data[0]) / h

    result = diff_table[0, 0]
    u_term = 1

    for i in range(1, len(x_data)):
        u_term *= (u - i + 1) / i
        result += u_term * diff_table[0, i]

    return result


def solve():
    print("\n=== Tasks for Forward and Backward difference ===")

    print("\nTask 1: Forward Difference Table")
    x1 = np.array([10, 20, 30, 40])
    y1 = np.array([1.1, 2.0, 4.4, 7.9])
    result1 = calculate_forward_differences(x1, y1)
    print_difference_table(result1)

    print("\nTask 2: Forward Difference Table and Δ³f(2)")
    x2 = np.array([0, 1, 2, 3, 4])
    y2 = np.array([1.0, 1.5, 2.2, 3.1, 4.6])
    result2 = calculate_forward_differences(x2, y2)
    print_difference_table(result2)
    print(f"Δ³f(2) = {result2['diff_matrix'][2, 3]}")

    print("\nTask 3: Polynomial y = x³ + x² - 2x + 1")

    def f3(x): return x ** 3 + x ** 2 - 2 * x + 1

    x3 = np.array([0, 1, 2, 3, 4, 5])
    y3 = np.array([f3(x) for x in x3])
    result3 = calculate_forward_differences(x3, y3)
    print_difference_table(result3)
    actual_y6 = f3(6)
    interpolated_y6 = newton_forward_interpolation(6, result3)

    print(f"Actual y(6) = {actual_y6}")
    print(f"Interpolated y(6) = {interpolated_y6}")

    print("\nTask 4: f(x) = x³ + 5x - 7")
    def f4(x): return x ** 3 + 5 * x - 7
    x4 = np.array([-1, 0, 1, 2, 3, 4, 5])
    y4 = np.array([f4(x) for x in x4])
    result4 = calculate_forward_differences(x4, y4)
    print_difference_table(result4)
    actual_y6 = f4(6)
    interpolated_y6 = newton_forward_interpolation(6, result4)

    print(f"Actual y(6) = {actual_y6}")
    print(f"Interpolated y(6) = {interpolated_y6}")

    print("\nTask 5: Extended Table")
    x5 = np.array([-0.2, 0.0, 0.2, 0.4, 0.6, 0.8, 1.0])
    y5 = np.array([2.6, 3.0, 3.4, 4.28, 7.08, 14.2, 29.0])
    result5 = calculate_forward_differences(x5, y5)
    first_x = -0.4
    last_x = 1.2
    first_y = newton_forward_interpolation(first_x, result5)
    last_y = newton_forward_interpolation(last_x, result5)

    x_extended = np.insert(x5, 0, first_x)
    x_extended = np.append(x_extended, last_x)
    y_extended = np.insert(y5, 0, first_y)
    y_extended = np.append(y_extended, last_y)

    result_extended = calculate_forward_differences(x_extended, y_extended)

    print_difference_table(result_extended)

    print("\nTask 6: find missing term")
    x6 = np.array([2, 3, 5])  # Skip 5
    y6 = np.array([45.0, 49.2, 59.6999])
    result6 = calculate_forward_differences(x6, y6)
    missing_value = lagrange_interpolation(4, x6, y6)
    print(f"Missing value = {missing_value}")


    print("\nTask 7: Product Function")
    def f7(x): return product_function(x, range(1, 16, 2))
    x7 = np.array([0, 1, 2, 3, 4])
    y7 = np.array([f7(x) for x in x7])
    result7 = calculate_forward_differences(x7, y7)
    print_difference_table(result7)

    print("\nTask 8: Polynomial of Degree 4")
    x8 = np.array([0, 1, 2, 3, 4])
    y8 = np.array([1, -1, 1, -1, 1])
    result8 = calculate_forward_differences(x8, y8)

    y_5 = newton_forward_interpolation(5, result8)
    x8 = np.append(x8, 5)
    y8 = np.append(y8, y_5)

    y_6 = lagrange_interpolation(6, x8, y8)
    x8 = np.append(x8, 6)
    y8 = np.append(y8, y_6)

    y_7 = lagrange_interpolation(7, x8, y8)
    x8 = np.append(x8, 7)
    y8 = np.append(y8, y_7)
    print("After adding x=5, 6, 7:", x8, y8)


    print("\nTask 9: Another Product Function")
    def f9(x): return product_function(x, range(1, 20, 2))
    x9 = np.array([0, 1, 2, 3, 4])
    y9 = np.array([f9(x) for x in x9])
    result9 = calculate_forward_differences(x9, y9)
    print_difference_table(result9)


if __name__ == "__main__":
    solve()