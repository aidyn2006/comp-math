import pandas as pd
from tabulate import tabulate


# Initialize matrix A and vector B for the system of equations
def initialize_system():
    A = [
        [5, 2, 1],
        [-1, 4, 2],
        [2, -3, 10]
    ]
    B = [7, 3, -1]
    return A, B


# Manually calculate the determinant of a matrix using recursion
def determinant(matrix):
    n = len(matrix)
    if n == 1:
        return matrix[0][0]
    if n == 2:
        return matrix[0][0] * matrix[1][1] - matrix[0][1] * matrix[1][0]

    det = 0
    for i in range(n):
        sub_matrix = [row[:i] + row[i+1:] for row in matrix[1:]]  # Remove row 0 and column i
        det += ((-1) ** i) * matrix[0][i] * determinant(sub_matrix)
    return det


# Solve the system using Cramer's rule
def solve_cramer(A, B):
    det_A = determinant(A)
    if det_A == 0:
        return "No unique solution (det(A) = 0)"
    solutions = []
    for i in range(len(B)):
        Ai = [row[:] for row in A]  # Create a copy of matrix A
        for j in range(len(A)):  # Replace the i-th column of A with B
            Ai[j][i] = B[j]
        solutions.append(determinant(Ai) / det_A)
    return solutions


# Solve the system using Gaussian elimination
def solve_gauss_elimination(A, B):
    n = len(A)
    augmented_matrix = [row + [B[i]] for i, row in enumerate(A)]  # Augment A with B

    # Forward elimination
    for i in range(n):
        if augmented_matrix[i][i] == 0:
            return "No unique solution"
        for j in range(i + 1, n):
            ratio = augmented_matrix[j][i] / augmented_matrix[i][i]
            for k in range(i, n + 1):
                augmented_matrix[j][k] -= ratio * augmented_matrix[i][k]

    # Backward substitution
    x = [0] * n
    for i in range(n - 1, -1, -1):
        x[i] = augmented_matrix[i][n] / augmented_matrix[i][i]
        for j in range(i - 1, -1, -1):
            augmented_matrix[j][n] -= augmented_matrix[j][i] * x[i]

    return x


# Solve the system using iterative methods (Jacobi and Gauss-Seidel)
def iterative_solver(A, B, method="jacobi", tol=1e-10, max_iterations=100):
    if method not in ["jacobi", "gauss-seidel"]:
        raise ValueError("Invalid method. Choose either 'jacobi' or 'gauss-seidel'.")

    n = len(B)
    x = [0.0] * n  # Initial guess for solution
    results = []

    for k in range(max_iterations):
        x_new = x[:]
        for i in range(n):
            s1 = sum(A[i][j] * x_new[j] for j in range(i))  # Summing for Jacobi or Gauss-Seidel
            s2 = sum(A[i][j] * x[j] for j in range(i + 1, n))
            x_new[i] = (B[i] - s1 - s2) / A[i][i]

        errors = [abs(x_new[i] - x[i]) for i in range(n)]
        results.append([k] + x_new + errors)
        if all(e < tol for e in errors):  # Check for convergence
            break
        x = x_new

    columns = ["Iteration"] + [f"x_{i + 1}" for i in range(n)] + [f"Error x_{i + 1}" for i in range(n)]
    return pd.DataFrame(results, columns=columns)


# Display the results of various methods
def display_results(cramer_sol, gauss_sol, jacobi_res, gauss_seidel_res):
    print("Cramer's Rule Solution:", cramer_sol)
    print("Gauss Elimination Solution:", gauss_sol)

    print("\nJacobi Method Iterations:")
    print(tabulate(jacobi_res, headers='keys', tablefmt='grid'))

    print("\nGauss-Seidel Method Iterations:")
    print(tabulate(gauss_seidel_res, headers='keys', tablefmt='grid'))

    jacobi_res.to_csv('jacobi_iterations.csv', index=False)
    gauss_seidel_res.to_csv('gauss_seidel_iterations.csv', index=False)
    print("\nResults saved to 'jacobi_iterations.csv' and 'gauss_seidel_iterations.csv'.")


# Main execution block
if __name__ == "__main__":
    pd.set_option('display.max_columns', None)
    pd.set_option('display.width', 1000)

    A, B = initialize_system()
    cramer_solution = solve_cramer(A, B)
    gauss_solution = solve_gauss_elimination(A, B)
    jacobi_results = iterative_solver(A, B, method="jacobi")
    gauss_seidel_results = iterative_solver(A, B, method="gauss-seidel")

    display_results(cramer_solution, gauss_solution, jacobi_results, gauss_seidel_results)
