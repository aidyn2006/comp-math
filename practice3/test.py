import numpy as np

def cramer(A, B):
    det_A = np.linalg.det(A)
    if det_A == 0:
        raise ValueError("The system has no unique solution.")
    n = len(B)
    solutions = []
    for i in range(n):
        A_copy = A.copy()
        A_copy[:, i] = B
        solutions.append(np.linalg.det(A_copy) / det_A)
    return solutions

def gaussian_elimination(A, B):
    A = A.astype(float)
    B = B.astype(float)
    n = len(B)
    for i in range(n):
        for j in range(i + 1, n):
            factor = A[j, i] / A[i, i]
            A[j] -= factor * A[i]
            B[j] -= factor * B[i]
    x = np.zeros(n)
    for i in range(n - 1, -1, -1):
        x[i] = (B[i] - np.dot(A[i, i + 1:], x[i + 1:])) / A[i, i]
    return x

def gauss_jordan(A, B):
    A = A.astype(float)
    B = B.astype(float)
    n = len(B)
    augmented = np.hstack([A, B.reshape(-1, 1)])
    for i in range(n):
        augmented[i] /= augmented[i, i]
        for j in range(n):
            if i != j:
                augmented[j] -= augmented[j, i] * augmented[i]
    return augmented[:, -1]

def jacobi_iteration(A, B, x0, tol=1e-6, max_iter=100):
    n = len(B)
    x = x0.copy()
    for _ in range(max_iter):
        x_new = x.copy()
        for i in range(n):
            sum_j = np.dot(A[i, :], x) - A[i, i] * x[i]
            x_new[i] = (B[i] - sum_j) / A[i, i]
        if np.linalg.norm(x_new - x, ord=np.inf) < tol:
            return x_new
        x = x_new
    raise ValueError("Jacobi iteration did not converge.")

def gauss_seidel(A, B, x0, tol=1e-6, max_iter=100):
    n = len(B)
    x = x0.copy()
    for _ in range(max_iter):
        x_new = x.copy()
        for i in range(n):
            sum_j = np.dot(A[i, :i], x_new[:i]) + np.dot(A[i, i + 1:], x[i + 1:])
            x_new[i] = (B[i] - sum_j) / A[i, i]
        if np.linalg.norm(x_new - x, ord=np.inf) < tol:
            return x_new
        x = x_new
    raise ValueError("Gauss-Seidel iteration did not converge.")

def relaxation_method(A, B, x0, omega=1.25, tol=1e-6, max_iter=100):
    n = len(B)
    x = x0.copy()
    for _ in range(max_iter):
        x_new = x.copy()
        for i in range(n):
            sum_j = np.dot(A[i, :i], x_new[:i]) + np.dot(A[i, i + 1:], x[i + 1:])
            x_new[i] = (1 - omega) * x[i] + omega * (B[i] - sum_j) / A[i, i]
        if np.linalg.norm(x_new - x, ord=np.inf) < tol:
            return x_new
        x = x_new
    raise ValueError("Relaxation method did not converge.")

A = np.array([
    [3, -5, 47, 20],
    [11, 16, 17, 10],
    [56, 22, 11, -18],
    [17, 66, -12, 7]
])
B = np.array([18, 26, 34, 82])
x0 = np.zeros(len(B))

try:
    print("Cramer's Solution:", cramer(A, B))
except ValueError as e:
    print("Cramer's Method Error:", e)

print("Gaussian Elimination Solution:", gaussian_elimination(A.copy(), B.copy()))
print("Gauss-Jordan Solution:", gauss_jordan(A.copy(), B.copy()))
print("Jacobi Iteration Solution:", jacobi_iteration(A, B, x0))
print("Gauss-Seidel Solution:", gauss_seidel(A, B, x0))
print("Relaxation Solution:", relaxation_method(A, B, x0))
