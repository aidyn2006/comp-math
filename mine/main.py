

from matplotlib import pyplot as plt
import sympy as sp
from math import *


x = sp.symbols('x')
equation = input("Enter equation (in terms of x): ")
tolerance = float(input("Enter tolerance: "))
f_expr = sp.sympify(equation)
f_prime_expr = sp.diff(f_expr, x)


def f(x_val):
    return float(f_expr.subs(x, x_val))

def f_prime(x_val):
    return float(f_prime_expr.subs(x, x_val))

def find_initial_guess(f, start=-10, end=10, step=0.1):
    x = start
    while x <= end:
        if f(x) * f(x + step) < 0:
            return (x + x + step) / 2
        x += step
    return find_initial_guess(f, start=start - 10, end=end + 10, step=step)

def find_bisection_interval(f, start=-10, end=10, step=0.1):
    x = start
    while x <= end:
        if f(x) * f(x + step) < 0:
            return x, x + step
        x += step
    return find_bisection_interval(f, start=start - 10, end=end + 10, step=step)

def bisection_method(f, x1, x2, tolerance):
    x_values, y_values, epsilon_values = [], [], []
    count = 0

    while x2 - x1 > tolerance:
        mid = (x1 + x2) / 2
        x_values.append(mid)
        y_values.append(count)
        epsilon_values.append(abs(x2 - x1))

        if f(mid) == 0:
            break
        elif f(mid) * f(x1) < 0:
            x2 = mid
        else:
            x1 = mid

        count += 1

    return x_values, y_values, epsilon_values

def iteration_method(x0, tolerance, max_iter=100):
    x_values = [x0]
    count = 0

    while count < max_iter:
        try:
            x_next = x0 - f(x0) / f_prime(x0)
            x_values.append(x_next)
            if abs(x_next - x0) < tolerance:
                break

            x0 = x_next
        except ZeroDivisionError:
            print("Division by zero encountered in iteration (f'(x) = 0).")
            break

        count += 1

    return x_values, list(range(len(x_values)))

def false_position_method(f, x1, x2, tolerance):
    x_values, y_values, epsilon_values = [], [], []
    count = 0

    while abs(x2 - x1) > tolerance:
        x_root = x2 - (f(x2) * (x2 - x1)) / (f(x2) - f(x1))
        x_values.append(x_root)
        y_values.append(count)
        epsilon_values.append(abs(x2 - x1))

        if abs(f(x_root)) < tolerance:
            break
        elif f(x1) * f(x_root) < 0:
            x2 = x_root
        else:
            x1 = x_root

        count += 1

    return x_values, y_values, epsilon_values

def newton_raphson_method(f, f_prime, x0, tolerance):
    x_values, y_values = [x0], [0]
    count = 1

    while True:
        x_next = x0 - f(x0) / f_prime(x0)
        x_values.append(x_next)
        y_values.append(count)

        if abs(x_next - x0) < tolerance:
            break

        x0 = x_next
        count += 1

    return x_values, y_values

x1, x2 = find_bisection_interval(f, start=-10, end=10, step=0.1)

bisection_x, bisection_y, bisection_e = bisection_method(f, x1, x2, tolerance)
print("Bisection Method: \n approximately root:", bisection_x[-1], "number of iterations:", bisection_y[-1])

newton_raphson_x, newton_raphson_y = newton_raphson_method(f, f_prime, 2, tolerance)
print("Newton-Raphson Method: \n approximately root:", newton_raphson_x[-1], "number of iterations:", newton_raphson_y[-1])

iteration_x, iteration_y = iteration_method((x1 + x2) / 2, tolerance)
print("Iteration Method: \n approximately root:", iteration_x[-1], "number of iterations:", iteration_y[-1])

false_position_x, false_position_y, false_position_e = false_position_method(f, x1, x2, tolerance)
print("False Position Method: \n approximately root:", false_position_x[-1], "number of iterations:", false_position_y[-1])

plt.figure(figsize=(12, 6))

plt.plot(bisection_y, bisection_x, marker='o', label='Bisection Method')
plt.plot(iteration_y, iteration_x, marker='^', label='Iteration Method')
plt.plot(false_position_y, false_position_x, marker='s', label='False Position Method')
plt.plot(newton_raphson_y, newton_raphson_x, marker='x', label='Newton-Raphson Method')

plt.xlabel("Iteration")
plt.ylabel("x Value")
plt.title("Convergence of Root-Finding Methods")
plt.legend()
plt.grid()
plt.show()
