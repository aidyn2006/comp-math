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

a = find_initial_guess(f)
print(f"Initial guess for the root: {a}")
x_values = [a]
epsilon_values = []

b = a - (f(a) / f_prime(a))
epsilon_values.append(abs(b - a))
x_values.append(b)

while abs(b - a) > tolerance:
    a = b
    b = a - (f(a) / f_prime(a))
    epsilon_values.append(abs(b - a))
    x_values.append(b)

print(f"Root found: {b}")

iterations = list(range(len(x_values)))

plt.figure(figsize=(12, 6))

plt.subplot(1, 2, 1)
plt.plot(iterations, x_values, marker='o', label='Approximations (xj)')
plt.xlabel("Number of Iterations")
plt.ylabel("Root Approximation (xj)")
plt.title("Convergence of the Newton-Raphson Method")
plt.grid(color="hotpink", linestyle='--')
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(iterations[1:], epsilon_values, marker='o', color='r', label='Error |xj+1 - xj|')
plt.xlabel("Number of Iterations")
plt.ylabel("Error (Log Scale)")
plt.title("Error Reduction in Newton-Raphson Method")
plt.yscale('log')
plt.grid(color="hotpink", linestyle='--')
plt.legend()

plt.tight_layout()
plt.show()
