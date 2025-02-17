import numpy as np
from functools import reduce
import operator


def calculate_forward_differences(x, f):
    """
    Calculate forward differences table for given x values and their corresponding f(x) values.
    """
    n = len(x)
    if len(f) != n:
        raise ValueError("x and f must have the same length")

    diff_table = np.zeros((n, n))
    diff_table[:, 0] = f

    for j in range(1, n):
        for i in range(n - j):
            diff_table[i, j] = diff_table[i + 1, j - 1] - diff_table[i, j - 1]

    table = []
    for i in range(n):
        row = {
            'x': x[i],
            'f': f[i]
        }
        for j in range(1, n - i):
            row[f'Δ{j}f'] = diff_table[i, j]
        table.append(row)

    return {
        'table': table,
        'diff_matrix': diff_table
    }


def print_difference_table(result, method='forward'):
    """
    Print the difference table in a formatted way.
    """
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


def product_function(x, terms):
    """Calculate product function of form (2x + 1)(2x + 3)...(2x + n)"""
    return reduce(operator.mul, ((2 * x + i) for i in terms))


def solve_all_tasks():
    """Solve all numerical difference tasks"""

    print("Task 1: Forward Difference Table")
    x1 = np.array([10, 20, 30, 40])
    y1 = np.array([1.1, 2.0, 4.4, 7.9])
    result1 = calculate_forward_differences(x1, y1)
    print_difference_table(result1)

    print("\nTask 2: Forward Difference Table and Δ³f(2)")
    x2 = np.array([0, 1, 2, 3, 4])
    f2 = np.array([1.0, 1.5, 2.2, 3.1, 4.6])
    result2 = calculate_forward_differences(x2, f2)
    print_difference_table(result2)
    print(f"Δ³f(2) = {result2['diff_matrix'][2, 3]}")

    print("\nTask 3: Polynomial y = x³ + x² - 2x + 1")

    def f3(x): return x ** 3 + x ** 2 - 2 * x + 1

    x3 = np.array([0, 1, 2, 3, 4, 5])
    y3 = np.array([f3(x) for x in x3])
    result3 = calculate_forward_differences(x3, y3)
    print_difference_table(result3)
    print(f"y(6) = {f3(6)}")

    print("\nTask 4: f(x) = x³ + 5x - 7")

    def f4(x): return x ** 3 + 5 * x - 7

    x4 = np.array([-1, 0, 1, 2, 3, 4, 5])
    y4 = np.array([f4(x) for x in x4])
    result4 = calculate_forward_differences(x4, y4)
    print_difference_table(result4)
    print(f"f(6) = {f4(6)}")

    print("\nTask 5: Extended Table")
    x5 = np.array([-0.2, 0.0, 0.2, 0.4, 0.6, 0.8, 1.0])
    y5 = np.array([2.6, 3.0, 3.4, 4.28, 7.08, 14.2, 29.0])
    result5 = calculate_forward_differences(x5, y5)
    print_difference_table(result5)

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
    print_difference_table(result8)

    print("\nTask 9: Another Product Function")

    def f9(x): return product_function(x, range(1, 20, 2))

    x9 = np.array([0, 1, 2, 3, 4])
    y9 = np.array([f9(x) for x in x9])
    result9 = calculate_forward_differences(x9, y9)
    print_difference_table(result9)


    print("\nTask 10: Find Missing Term")
    x10 = np.array([2, 3, 4, 5, 6])
    y10 = np.array([45.0, 49.2, 54.1, None, 67.4])
    # Calculate differences using available points to estimate missing value
    x10_known = np.array([2, 3, 4, 6])
    y10_known = np.array([45.0, 49.2, 54.1, 67.4])
    result10 = calculate_forward_differences(x10_known, y10_known)
    print_difference_table(result10)


if __name__ == "__main__":
    solve_all_tasks()
