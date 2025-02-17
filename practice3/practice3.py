import numpy as np
import pandas as pd
from tabulate import tabulate

define_matrix = lambda: (
    np.array([
        [10, -7, 3, 5],
        [-6, 8, -1, -4],
        [3, 1, 4, 11],
        [5, -18, -4, 8]
        ], dtype=float),
    np.array([6, 5, 3, 14], dtype=float),
)


def cramer(coeff_matrix, constants):
    det_main = np.linalg.det(coeff_matrix)
    if np.isclose(det_main, 0):
        return "No unique solution (determinant is zero)"

    solution = []
    for col in range(len(constants)):
        temp_matrix = coeff_matrix.copy()
        temp_matrix[:, col] = constants
        solution.append(np.linalg.det(temp_matrix) / det_main)
    return np.array(solution)


def gauss_elimination(coeff_matrix, constants):
    n = len(constants)
    augmented_matrix = np.hstack((coeff_matrix, constants.reshape(-1, 1)))

    for i in range(n):
        if augmented_matrix[i][i] == 0.0:
            raise ZeroDivisionError("Divide by zero detected during forward elimination!")

        for j in range(i + 1, n):
            ratio = augmented_matrix[j][i] / augmented_matrix[i][i]
            for k in range(n + 1):
                augmented_matrix[j][k] -= ratio * augmented_matrix[i][k]

    solution = np.zeros(n)
    solution[n - 1] = augmented_matrix[n - 1][n] / augmented_matrix[n - 1][n - 1]
    for i in range(n - 2, -1, -1):
        solution[i] = augmented_matrix[i][n]
        for j in range(i + 1, n):
            solution[i] -= augmented_matrix[i][j] * solution[j]
        solution[i] /= augmented_matrix[i][i]

    return solution


def jacobi_method(coeff_matrix, constants, tolerance=1e-10, max_iter=100):
    size = len(constants)
    current_guess = np.zeros(size)
    results = []

    for iteration in range(max_iter):
        next_guess = np.zeros_like(current_guess)

        for i in range(size):
            sum_before = np.dot(coeff_matrix[i, :i], current_guess[:i])
            sum_after = np.dot(coeff_matrix[i, i + 1:], current_guess[i + 1:])
            next_guess[i] = (constants[i] - sum_before - sum_after) / coeff_matrix[i, i]

        error = np.abs(next_guess - current_guess)
        results.append([iteration] + list(next_guess) + list(error))

        if np.all(error < tolerance):
            break

        current_guess = next_guess

    columns = ["Iteration"] + [f"x_{i}" for i in range(size)] + [f"Error_x{i}" for i in range(size)]
    return pd.DataFrame(results, columns=columns)


def gauss_seidel_method(coeff_matrix, constants, tolerance=1e-10, max_iter=100):
    size = len(constants)
    current_guess = np.zeros(size)
    results = []

    for iteration in range(max_iter):
        next_guess = current_guess.copy()

        for i in range(size):
            sum_before = np.dot(coeff_matrix[i, :i], next_guess[:i])
            sum_after = np.dot(coeff_matrix[i, i + 1:], current_guess[i + 1:])
            next_guess[i] = (constants[i] - sum_before - sum_after) / coeff_matrix[i, i]

        error = np.abs(next_guess - current_guess)
        results.append([iteration] + list(next_guess) + list(error))

        if np.all(error < tolerance):
            break

        current_guess = next_guess

    columns = ["Iteration"] + [f"x_{i}" for i in range(size)] + [f"Error_x{i}" for i in range(size)]
    return pd.DataFrame(results, columns=columns)

if __name__ == "__main__":
    A, B = define_matrix()

    cramer_result = cramer(A, B)
    gaussian_result = gauss_elimination(A, B)

    jacobi_result = jacobi_method(A, B)
    gauss_seidel_result = gauss_seidel_method(A, B)

    print("Cramer's Rule Result:", cramer_result)
    print("Gaussian Elimination Result:", gaussian_result)

    print("\nJacobi Method Iterations:")
    print(tabulate(jacobi_result, headers='keys', tablefmt='grid'))

    print("\nGauss-Seidel Method Iterations:")
    print(tabulate(gauss_seidel_result, headers='keys', tablefmt='grid'))

    jacobi_result.to_csv('jacobi_results.csv', index=False)
    gauss_seidel_result.to_csv('gauss_seidel_results.csv', index=False)

    print("\nResults exported to 'jacobi.csv' and 'gauss_seidel.csv'.")
