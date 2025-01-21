import matplotlib.pyplot as plt
from sympy import symbols, diff, lambdify

x = symbols('x')
func_sym = x ** 3 - x - 1
derivative_func_sym = diff(func_sym, x)

func1 = lambdify(x, func_sym)
derivative_func1 = lambdify(x, derivative_func_sym)
def bisection_method(func, a, b, tol, max_iter=100):
    x_vals = []
    for _ in range(max_iter):
        c = (a + b) / 2
        x_vals.append(c)
        if abs(func(c)) < tol or abs((b - a)) < tol:
            return c, x_vals
        if func(a) * func(c) < 0:
            b = c
        else:
            a = c
    return c, x_vals

def false_position_method(func, a, b, tol, max_iter=100):
    x_vals = []
    for _ in range(max_iter):
        c = b - (func(b) * (b - a)) / (func(b) - func(a))
        x_vals.append(c)
        if abs(func(c)) < tol:
            return c, x_vals
        if func(a) * func(c) < 0:
            b = c
        else:
            a = c
    return c, x_vals

def iteration_method(func, x0, g, tol, max_iter=100):
    x_vals = [x0]
    for _ in range(max_iter):
        x1 = g(x_vals[-1])
        x_vals.append(x1)
        if abs(x1 - x_vals[-2]) < tol:
            return x1, x_vals
    return x1, x_vals

def newton_raphson_method(func, derivative, x0, tol, max_iter=100):
    x_vals = [x0]
    for _ in range(max_iter):
        x1 = x_vals[-1] - func(x_vals[-1]) / derivative(x_vals[-1])
        x_vals.append(x1)
        if abs(x1 - x_vals[-2]) < tol:
            return x1, x_vals
    return x1, x_vals

if __name__ == "__main__":
    a, b = 1, 2
    x0 = 1.5

    g = lambda x: (x + 1) ** (1 / 3)

    tol = 0.0000001

    root_bisection, bisection_vals = bisection_method(func1, a, b, tol=tol)
    root_false_position, false_position_vals = false_position_method(func1, a, b, tol=tol)
    root_iteration, iteration_vals = iteration_method(func1, x0, g, tol=tol)
    root_newton, newton_vals = newton_raphson_method(func1, derivative_func1, x0, tol=tol)

    plt.figure(figsize=(10, 6))
    plt.plot(bisection_vals, label="Bisection Method", marker='o')
    plt.plot(false_position_vals, label="False Position Method", marker='x')
    plt.plot(iteration_vals, label="Iteration Method", marker='^')
    plt.plot(newton_vals, label="Newton-Raphson Method", marker='s')

    plt.xlabel("Iteration")
    plt.ylabel("x Value")
    plt.title("Convergence of Root-Finding Methods")
    plt.legend()
    plt.grid()
    plt.show()

